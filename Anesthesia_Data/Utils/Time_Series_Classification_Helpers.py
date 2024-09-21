import import_ipynb
import os
import pandas as pd
import numpy as np
import sys

# For being able to access folders from "grandparent" directory "Anesthesia_Data"
import sys
sys.path.append('../') 

import Utils.Classification_Helpers as helpers



def import_and_concatenate_datasets(subject_list, list_of_filenames, parent_directory):
    """
    Import and concatenate feature datasets for each subject.

    Args:
    - subject_list (list): List of subject names.

    Returns:
    - pd.DataFrame: Concatenated feature DataFrame.
    - list: List of all labels.
    """
    subject_feature_dfs = {}

    for subject_idx, subject in enumerate(subject_list):
        subject_feature_dfs[subject] = pd.DataFrame()

        for data_type in ["EEG", "EMG"]:
            data_frames = []

            for file in list_of_filenames:
                path = os.path.join(str(parent_directory), "Features", str(subject), str(data_type), file)
                if os.path.exists(path):
                    data_frames.append(pd.read_csv(path))

            df_both_data_types = pd.concat(data_frames, axis=1)

            if not subject_feature_dfs[subject].empty:
                subject_feature_dfs[subject] = pd.concat([subject_feature_dfs[subject], df_both_data_types], axis=1)
            else:
                subject_feature_dfs[subject] = df_both_data_types
            
            # For duplicate columns, only keep one
            subject_feature_dfs[subject] = helpers.keep_first_duplicate_columns(subject_feature_dfs[subject])


        subject_feature_dfs[subject]["Subject"] = subject_idx


    feature_df = pd.concat(subject_feature_dfs.values(), ignore_index=True)


    feature_df.drop(columns=['Unnamed: 0'], inplace=True)
    
    return feature_df


def create_time_series_feature_dfs(subject_list, time_series_filenames, parent_directory="Time_Series"):
    """
    Create the time series feature dataframe by importing and concatenating datasets.

    Parameters:
    subject_list (list): List of subjects for data import.
    list_of_filenames (list): List of filenames for time series features.
    ts_helpers (module): Module containing helper functions for time series data processing.

    Returns:
    pd.DataFrame: DataFrame containing the concatenated time series features.
    """

    all_dataframes = []
    
    for list_of_filenames in time_series_filenames:
        # Import and concatenate time series datasets
        time_series_feature_df = import_and_concatenate_datasets(
        subject_list, list_of_filenames, parent_directory=parent_directory
        )

        all_dataframes.append(time_series_feature_df)

    return all_dataframes
