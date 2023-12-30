#!/usr/bin/env python3
import pandas as pd
from chembl_webresource_client.new_client import new_client
from pandas import DataFrame



def search_query(user_search:str=None, chembl_id_search:str=None, save_prefix=None) -> DataFrame():
    '''Function to locate a biological target within the Chembl database,
    returns a pandas dataframe with SMILES and associated PIC50 for single target binding assays.
    
    Parameters: 
        'user_search' (str): a string of a biological target.
        
        'chembl_id_search' (str): a string of a Chembl ID for a SINGLE PROTEIN target.
    '''

    target:chembl_webresource_client.query_set.QuerySet = new_client.target
    if chembl_id_search is None:
        if user_search is None:
            user_search: str = input("\nName the Biological target you want to search CHEMBL for: ")
            if user_search == "":
                return search_query()
        try:
            res:dict = target.filter(pref_name__icontains=user_search, target_type='SINGLE PROTEIN').only(['organism','pref_name','target_chembl_id','target_type']).order_by(['pref_name','organism'])
            df: DataFrame = DataFrame(res).reset_index(drop=True)
            print(f"\n{df}\n")
            if df.empty == True:
                return search_query()
        except ValueError as e:
            print(f"Input Error: {str(e)}, try again")
            return search_query()

        try:
            user_choice: int = int(input("""\nSelect the dataframe row ID that you wish to use as a target,
hit ENTER to search again: """))
            chembl_id_search:str = df['target_chembl_id'][user_choice]
            print(f"\nYour search will be for {chembl_id_search}\n")

        except ValueError:
            return search_query()
    else:
        print(f"\nYour search is for {chembl_id_search}\n")
        
    activity:chembl_webresource_client.query_set.QuerySet = new_client.activity
    activity_dict: dict = activity.filter(target_chembl_id=chembl_id_search, standard_type="IC50", assay_type='B', pchembl_value__isnull=False)
    activity_df:DataFrame = DataFrame(activity_dict)
    if len(activity_df) < 20:
        print("Dataset is too small, choose another target")
        return search_query()
    out_df:pd.Dataframe = activity_df[['canonical_smiles', 'pchembl_value', 'target_pref_name']]
    out_df:pd.DataFrame = out_df.dropna()
    out_df:pd.DataFrame = out_df.sort_values(by='pchembl_value', ascending=False)
    out_df = out_df.drop_duplicates(subset='canonical_smiles', keep='first').reset_index(drop=True)
    print(f'\n{out_df}\n')
    use_dataset: str = input("\nDo you wish to use this dataset for model training? (y/n): ")
    if use_dataset.lower() == 'n':
        return search_query()
    elif use_dataset.lower() == 'y':
        if save_prefix is not None:
            out_df.to_csv(save_prefix+'_training.csv', index=False)    
        elif save_prefix is None:
            out_df.to_csv('training_dataset.csv', index=False)
        return out_df
    else:
        print("Invalid response")
        return search_query(chembl_id_search=chembl_id_search)