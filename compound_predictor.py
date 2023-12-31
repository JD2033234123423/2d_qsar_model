#!/usr/bin/env python3
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt


class CompoundPredictor:
    def __init__(self,training_dataset, prediction_method='random forest', save_prefix=None) -> None:
        self.training_dataset = training_dataset 
        self.x_data = None        
        self.prediction_method = prediction_method
        self.target_name = self.training_dataset['target_pref_name'][0]
        self.save_prefix = save_prefix
        
    @staticmethod
    def generate_fingerprint(smiles: str) -> np.ndarray:
        """
        Takes a standardized canonical smiles and returns an ECFP4 fingerprint.

        Parameters:
        
        'smiles' (str): A string of a compound SMILES
        
        Returns:
        
        np.ndarray: Morgan fingerprint as a NumPy array
        """
        molecule: rdkit.Chem.rdchem.Mol = Chem.rdmolfiles.MolFromSmiles(smiles)
        fp: rdkit.DataStructs.cDataStructs.ExplicitBitVect = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)
        ary = [np.array(x) for x in fp]
        return ary

    def data_preprocessing(self) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Takes an instance of the class CompoundPredictor and returns a tuple of datasets to use in training a regression model and testing a regression model.
        
        Returns:
        
        n_estimators
        
        tuple of data for training, testing. 
        
        """
        if self.x_data is None:
            self.x_data:np.ndarray = np.vstack([self.generate_fingerprint(smiles) for smiles in self.training_dataset['canonical_smiles']])
        self.y_data:pd.Series = self.training_dataset['pchembl_value']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y_data, test_size=0.25)

        return self.x_train, self.x_test, self.y_train, self.y_test
    
    def plot_scatter(self, true_values: pd.Series, predicted_values:np.ndarray, r2:float, rmse:np.ndarray) -> None:
        """ 
        
        Takes an instance of class CompoundPredictor and generates an x y scatter plot. Annotated with rmse and r2 values,
        also displays a line of x = y, where predicted and true PIC50 values would be equal
        
        Parameters: 
        
        'true_values': (pd.Series) Series of true PIC50 values for a single protein target.
        
        'predicted_values': (np.ndarray) Array of predicted PIC50 values generated from a regress.ion model
        
        'r2': (float): Numerical value representing the coefficient of determination of the model.
        
        'rmse' (np.ndarray): Numerical value representing the root of Mean Squared error of the model.
        
        Returns:
        
        None
        
        Saved scatter plot is saved within the current working directory
        
        """
        
        data = pd.DataFrame({'True_Values': true_values, 'Predicted_Values': predicted_values})
        data['True_Values'] = pd.to_numeric(data['True_Values'], errors='coerce')
        data.sort_values(by='True_Values', inplace=True)
        plt.figure(dpi=1200)
        print(f'\nPlotting Graph for: \n\n{data}\n')
        max_value = data.to_numpy().max()
        min_value = data.to_numpy().min()
        min_value -= 1 
        max_value += 1

        plt.scatter(data['True_Values'], data['Predicted_Values'], alpha=0.7)
        
        plt.axline((0, 0), slope=1)
        plt.xlim(min_value, max_value)
        plt.ylim(min_value, max_value)
        
        plt.title(f'{self.target_name} - {self.prediction_method}')
        plt.xlabel('Real PIC50 (M)')
        plt.ylabel('Predicted PIC50 (M)')

        text_x = min_value + 0.5
        text_y = max_value - 0.9 

        plt.text(text_x, text_y, f'R^2 = {r2:.3f}\nRMSE = {rmse:.3f}', bbox = dict(facecolor = 'red', alpha = 0.5))
        

        if self.save_prefix is None:
            plt.savefig(f'{self.target_name}_{self.prediction_method}_scatter.pdf')
        elif self.save_prefix:
            plt.savefig(f'{self.save_prefix}_scatter.pdf')


    def rfmodel(self) -> tuple[RandomForestRegressor, np.ndarray, pd.Series, np.ndarray, float]:
        """
        Builds and trains a Random Forest regression model and returns relevant information.

        Returns:
        
        tuple[RandomForestRegressor, np.ndarray, pd.Series, np.ndarray, float]: A tuple containing the trained model,
        root mean squared error (rmse), true target values (y_test), predicted target values (y_pred), and R-squared (r2).
        
        """
        reg = RandomForestRegressor(n_estimators=300)
        self.data_preprocessing()
        print(f'\nBuilding model for {self.target_name}\n')
        reg.fit(self.x_train, self.y_train)
        y_pred:np.ndarray = reg.predict(self.x_test)
        rmse: np.ndarray = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2:float = r2_score(self.y_test, y_pred)
        return reg, rmse, self.y_test, y_pred,  r2
        
    def xgboost_regression(self) -> tuple[xgb.XGBRegressor, np.ndarray, pd.Series, np.ndarray, float]:
        """
        Builds and trains an XGBoost regression model and returns relevant information.

        Returns:
        
        tuple[xgb.XGBRegressor, np.ndarray, pd.Series, np.ndarray, float]: A tuple containing the trained XGBoost model,
        root mean squared error (rmse), true target values (y_test), predicted target values (y_pred), and R-squared (r2).
        
        """
        
        xgb_r = xgb.XGBRegressor(objective = 'reg:squarederror', n_estimators = 300)
        self.data_preprocessing()
        print(f'\nBuilding model for {self.target_name}\n')
        xgb_r.fit(self.x_train, self.y_train)
        y_pred:np.ndarray = xgb_r.predict(self.x_test)
        rmse:np.ndarray = np.sqrt(mean_squared_error(self.y_test, pred))
        r2:float = r2_score(self.y_test, pred)
        return xgb_r, rmse, self.y_test, y_pred,  r2
        
    def predict(self) -> tuple:
        """
        Chooses and executes the appropriate prediction method based on the specified prediction method.

        Returns:
        
        tuple: A tuple containing the relevant information based on the chosen prediction method.
        
        """
        if self.prediction_method == 'random forest':
            return self.rfmodel()
        elif self.prediction_method == 'xgboost':
            return self.xgboost_regression()
    @staticmethod
    def predict_pIC50(model:object, novel_compound:pd.DataFrame) -> pd.DataFrame:
        """
        Predicts pIC50 values for compounds to a single target using a given model.

        Parameters:
        
        'model' (object): The machine learning regression model used for prediction.
        
        'novel_compound' (pd.DataFrame): DataFrame containing information about novel compounds.

        Returns:
        pd.DataFrame: A DataFrame containing the predicted pIC50 values.
        
        """
        x_pred: np.ndarray = np.vstack([CompoundPredictor.generate_fingerprint(smiles) for smiles in novel_compound['canonical_smiles']])
        pred_pic50:np.ndarray = model.predict(x_pred)
        novel_compound['pred_pic50'] = pred_pic50
        return novel_compound
