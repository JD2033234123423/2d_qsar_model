#!/usr/bin/env python3
from numpy import ndarray
from database_search import search_query
from compound_predictor import CompoundPredictor
import argparse
import pandas as pd
import sys 
from joblib import dump, load

def main() -> None:
    
    
    parser = argparse.ArgumentParser()
    # Optional arguments to running
    parser.add_argument("-t", "--target_name", metavar='', help='(str): Takes a name of a SINGLE PROTEIN target.', type=str)
    parser.add_argument('-c', '--chembl_id', metavar= '', help='(str): Takes a chembl id of a SINGLE PROTEIN target.', type=str)
    parser.add_argument('-r', '--regression_method', metavar= '', help='(str): Choice of regression model for 2D training, can be either random forest or xgboost.', type=str, choices=['random forest', 'xgboost'], default='random forest')
    parser.add_argument('-f', '--training_file', metavar= '', help="""(str): Takes a location of a csv file used for training dataset, csv must have the headers:
'canonical_smiles' ,'pchembl_value, and, 'target_pref_name'. """, type=str)
    parser.add_argument('-s', '--save_prefix', metavar= '', help='(str): Takes a string of what you want your files named.', type=str)
    parser.add_argument('-l', '--load_model', metavar= '', help='(str): Takes a string of the location of your model you want parsing for predicting PIC50 values, requires a training dataset to be loaded.', type=str)

    args = parser.parse_args()
    
    def training(dataset, regression_method) -> object:
        predictor = CompoundPredictor(dataset, regression_method, args.save_prefix)
    
        def retrain_model() -> object:
            model, rmse, y_test, pred, r2 = predictor.predict()
            print(f"Root Mean Squared Error: {rmse}\nR^2: {r2}")
            retrain:str = ''
            while retrain.lower() != 'y' and retrain.lower() != 'n':
                retrain = input("\nWould you like to use this model, type n to retrain (y/n): ")
            if retrain.lower() == 'n':
                return retrain_model()
            elif retrain.lower() == 'y':
                predictor.plot_scatter(y_test, pred, r2, rmse)
                if args.save_prefix:
                    dump(model, args.save_prefix + '.joblib')
                else:
                    dump(model, 'trained_model.joblib')

            return model
        return retrain_model()

    
    if args.load_model is not None and args.training_file is not None:
        model: object = load(args.load_model)
        training_dataset: pd.DataFrame = pd.read_csv(args.training_file)
    
    elif args.load_model is not None and args.training_file is None:
        print('\nLoad model requires training file to be supplied too.\n')
        sys.exit(1)
    
    elif args.load_model is None:

        if args.training_file is None:
            training_dataset:pd.DataFrame = search_query(user_search=args.target_name, chembl_id_search=args.chembl_id, save_prefix=args.save_prefix)
        else:
            training_dataset: pd.DataFrame = pd.read_csv(args.training_file)

        model: object = training(training_dataset, args.regression_method)

    def make_prediction() -> None:
        predict: str = input(f'\nWould you like to predict the PIC50 of novel compounds to {training_dataset["target_pref_name"][0]} (y/n)?: ')

        if predict.lower() == 'y':
            predict_mol_path: str = input("\nEnter a path to a csv file where a column has the header 'canonical_smiles': ")

            try:
                mol_dataset: pd.DataFrame = pd.read_csv(predict_mol_path)
            except FileNotFoundError:
                print(f"\nError: File not found at {predict_mol_path}")
                return make_prediction()

            if 'canonical_smiles' in mol_dataset.columns:
                out_df: pd.DataFrame = CompoundPredictor.predict_pIC50(model, mol_dataset)
                
                if args.save_prefix:
                    out_df.to_csv(args.save_prefix+'_prediction.csv', index=False)
                    print('\nPrediction is done and saved.\n')
                else:
                    out_df.to_csv('predicted_pic50.csv', index=False)
                    print('\nPrediction is done and saved.\n')
            else:
                print("\nError: The specified column 'canonical_smiles' does not exist in the provided CSV file.\n")
                return make_prediction()
            
        elif predict.lower() == 'n':
            pass
        else:
            return make_prediction()

    make_prediction()
    
    print('\nProgramme has ended\n')
    

if __name__ == "__main__":
    main()
    
    sys.exit(0)
