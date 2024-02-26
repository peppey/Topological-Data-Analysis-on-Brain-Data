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

        subject_feature_dfs[subject]["Subject"] = subject_idx

    feature_df = pd.concat(subject_feature_dfs.values(), ignore_index=True)

    # For duplicate columns, only keep one
    feature_df = helpers.keep_first_duplicate_columns(feature_df)

    feature_df.drop(columns=['Unnamed: 0'], inplace=True)
    
    return feature_df