import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
import os

# For being able to access folders from "grandparent" directory "Anesthesia_Data"
import sys
sys.path.append('../') 


import Utils.Classification_Helpers as helpers



def import_and_concatenate_datasets(subjects, list_of_filenames, parent_directory = "", label_list = [0, 1, 2, 3, 4]):
    
    """
    Load feature DataFrames for specified subjects.
    
    Args:
    - subjects (list): List of subject names.
    
    Returns:
    - dict: Dictionary containing subject feature DataFrames.
    - list: List of all labels across subjects.
    """
    subject_feature_dfs = {}

    for subject in subjects:
        subject_feature_dfs[subject] = pd.DataFrame()
        data_frames = []

        # Topological Features
        for file in list_of_filenames:
            path = os.path.join(str(parent_directory), "Features", str(subject), file)
            if os.path.exists(path):
                data_frames.append(pd.read_csv(path))
        
        for df_idx, df in enumerate(data_frames):
            df.drop(df.columns[df.columns.str.contains('unnamed',case=False)], axis=1, inplace=True)

            if len(subject_feature_dfs[subject].index) > 0:
                subject_feature_dfs[subject] = pd.concat([subject_feature_dfs[subject], df], axis=1)
            else:
                subject_feature_dfs[subject] = pd.concat([subject_feature_dfs[subject], df], ignore_index=True)
                subject_feature_dfs[subject].drop(subject_feature_dfs[subject].columns[subject_feature_dfs[subject].columns.str.contains('_left',case=False)], axis=1, inplace=True)

        
        subject_feature_dfs[subject] = helpers.keep_first_duplicate_columns(subject_feature_dfs[subject])
            
        
        subject_feature_dfs[subject]["Subject"] = subjects.index(subject)


    brain_imaging_feature_df = pd.concat([subject_feature_dfs[subject] for subject in subjects], ignore_index=True)

    return brain_imaging_feature_df, subject_feature_dfs



def cut_dataframe_to_same_length_as_TS(subject_feature_dfs, subject_list, label_list = [0, 1, 2, 3, 4]):

    brain_imaging_feature_df = pd.DataFrame()

    for subject in subject_list:
        for label in label_list:
            subject_feature_dfs[subject] = subject_feature_dfs[subject].drop(subject_feature_dfs[subject][subject_feature_dfs[subject]["Label"]==label].index[-1])
            
        brain_imaging_feature_df = pd.concat([brain_imaging_feature_df, subject_feature_dfs[subject]])

    brain_imaging_feature_df.reset_index(inplace=True, drop=True)


    return brain_imaging_feature_df


def create_brain_imaging_feature_df(subject_list, brain_imagining_filenames):
    """
    Create the brain imaging feature dataframe, including fold-dependent features.

    Parameters:
    subject_list (list): List of subjects for data import.
    atol_vectorization_filename_brain_imaging (list): List of filenames for brain imaging vectorization features.
    bi_helpers (module): Module containing helper functions for brain imaging data processing.

    Returns:
    pd.DataFrame: DataFrame containing brain imaging features with the same length as time series data.
    pd.DataFrame: DataFrame containing focreate_time_series_feature_df(subject_list, list_of_filenames):ld-dependent brain imaging features with the same length as time series data.
    """

    all_dataframes = []

    for list_of_filenames in brain_imagining_filenames:
        # Import and concatenate brain imaging datasets
        _, subject_feature_df = import_and_concatenate_datasets(
            subject_list, list_of_filenames, parent_directory="Brain_Imaging"
        )
    
        # Cut the dataframe to the same length as the time series data
        feature_df = cut_dataframe_to_same_length_as_TS(
            subject_feature_df, subject_list
        )

        feature_df.fillna(0, inplace=True)

        all_dataframes.append(feature_df)

    return all_dataframes
