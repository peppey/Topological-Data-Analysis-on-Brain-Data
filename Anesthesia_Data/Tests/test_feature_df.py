import pytest
import pandas as pd
import numpy as np
import os

# For being able to access folders from "grandparent" directory "Anesthesia_Data"
import sys
sys.path.append('../') 

import Utils.Time_Series_Classification_Helpers as ts_helpers
import Utils.Brain_Imaging_Classification_Helpers as bi_helpers
import Utils.Classification_Helpers as helpers



def has_duplicates(df):
    """
    Check if DataFrame contains duplicate rows.

    Args:
    - df (pd.DataFrame): DataFrame to check.

    Returns:
    - bool: True if DataFrame contains duplicates, False otherwise.
    """
    return df.duplicated().any()



def create_test_features():

    subject_list = ["m292", "m294"]

    # Create two different dataframes with duplicates for the two subjects
    df_with_duplicates  = {}
    df_with_duplicates["m292"] = pd.DataFrame({'A': [1, 2, 3, 3], 'B': [4, 5, 6, 6]})
    df_with_duplicates["m294"] = pd.DataFrame({'A': [4, 5, 6, 6], 'B': [7, 8, 9, 9]})


    # Create DataFrame with duplicate rows
    df_no_duplicates  = {}
    df_no_duplicates["m292"] = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df_no_duplicates["m294"] = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})


    # Test Time Series Features
    for subject in subject_list:
        for data_type in ["EEG", "EMG"]:
            # With Duplicates
            path = os.path.join("Features", str(subject), str(data_type), "TS_Test_Features_with_Duplicates.csv")
            df_with_duplicates[subject].to_csv(path)
            # Without Duplicates
            path = os.path.join("Features", str(subject), str(data_type), "TS_Test_Features_no_Duplicates.csv")
            df_no_duplicates[subject].to_csv(path)


    # Brain Imaging Features
    for subject in subject_list:
        # With Duplicates
        path = os.path.join("Features", str(subject), "BI_Test_Features_with_Duplicates.csv")
        df_with_duplicates[subject].to_csv(path)

        # Without Duplicates
        path = os.path.join("Features", str(subject), "BI_Test_Features_no_Duplicates.csv")
        df_no_duplicates[subject].to_csv(path)

    pass



def test_feature_dfs_have_duplicates():

    subject_list = ["m292", "m294"]

    # Create Test Features
    create_test_features()

    # Test Time Series Features
    filename = "TS_Test_Features_with_Duplicates.csv"
    feature_df_with_duplicates = ts_helpers.import_and_concatenate_datasets(subject_list, [filename], parent_directory = "")
    
    assert has_duplicates(feature_df_with_duplicates) == True


    filename = "TS_Test_Features_no_Duplicates.csv"
    feature_df_no_duplicates = ts_helpers.import_and_concatenate_datasets(subject_list, [filename], parent_directory = "")

    assert has_duplicates(feature_df_no_duplicates) == False

