import pytest
import pandas as pd
import os
import math

# For being able to access folders from "grandparent" directory "Anesthesia_Data"
import sys
sys.path.append('../') 

import Utils.Time_Series_Classification_Helpers as ts_helpers
import Utils.Brain_Imaging_Classification_Helpers as bi_helpers
import Utils.Classification_Helpers as helpers



def has_duplicate_rows(df):
    """
    Check if DataFrame contains duplicate rows.

    Args:
    - df (pd.DataFrame): DataFrame to check.

    Returns:
    - bool: True if DataFrame contains duplicate rows, False otherwise.
    """
 
    return df.duplicated().any()


def create_test_features():

    subject_list = ["m292", "m294"]
    data_types_list = ["EEG", "EMG"]

    df_with_duplicates  = {}
    df_no_duplicates  = {}

    for subject in subject_list:
        df_with_duplicates[subject] = {}
        df_no_duplicates[subject] = {}


    for data_type in data_types_list:
        # Create  different dataframes with duplicates for the two subjects
        df_with_duplicates["m292"][data_type]  = pd.DataFrame({str(data_type)+'_A': [1, 2, 3, 13, 4, 4], str(data_type)+'_B': [4, 5, 8, 5, 4, 4], 'Label': [0, 1, 2, 3, 4, 4]})
        df_with_duplicates["m294"][data_type]  = pd.DataFrame({str(data_type)+'_A': [7, 8, 9, 8, 5, 5], str(data_type)+'_B': [10, 11, 4, 3, 5, 5], 'Label': [0, 1, 2, 3, 4, 4]})

        df_no_duplicates["m292"][data_type] = pd.DataFrame({str(data_type)+'_A': [1, 2, 12, 3, 5], str(data_type)+'_B': [4, 5, 6, 13, 1], 'Label': [0, 1, 2, 3, 4]})
        df_no_duplicates["m294"][data_type] = pd.DataFrame({str(data_type)+'_A': [4, 5, 6, 3, 5], str(data_type)+'_B': [7, 8, 4, 9, 3], 'Label': [0, 1, 2, 3, 4]})

    # Test Time Series Features
    for subject in subject_list:
        for data_type in data_types_list:
            # With Duplicates
            path = os.path.join("Features", str(subject), str(data_type), "TS_Test_Features_with_Duplicates.csv")
            df_with_duplicates[subject][data_type].to_csv(path)
            # Without Duplicates
            path = os.path.join("Features", str(subject), str(data_type), "TS_Test_Features_no_Duplicates.csv")
            df_no_duplicates[subject][data_type].to_csv(path)


    # Brain Imaging Features
    for subject in subject_list:
        # With Duplicates
        path = os.path.join("Features", str(subject), "BI_Test_Features_with_Duplicates.csv")
        df_with_duplicates[subject][data_type].to_csv(path)

        # Without Duplicates
        path = os.path.join("Features", str(subject), "BI_Test_Features_no_Duplicates.csv")
        df_no_duplicates[subject][data_type].to_csv(path)

    pass


def create_test_folds(df):

    subject_list = ["m292", "m294"]

    train_indices = {}
    validation_indices = {}

    for subject_idx, subject in enumerate(subject_list):

        labels = df[df["Subject"]==subject_idx]["Label"]
        train_indices[subject] = {}
        validation_indices[subject] = {}

        for unique_label in list(set(labels)):
            train_indices[subject]["Label_"+str(unique_label)] = {}
            validation_indices[subject]["Label_"+str(unique_label)] = {}

            for fold in range(5):
                occurence_of_label = list(labels).count(unique_label)
                test_indices_length = math.ceil(occurence_of_label/2)
                train_indices[subject]["Label_"+str(unique_label)]["Fold_"+str(fold)] = range(test_indices_length)
                validation_indices[subject]["Label_"+str(unique_label)]["Fold_"+str(fold)] = range(test_indices_length, occurence_of_label)

    return train_indices, validation_indices


def test_feature_dfs_have_duplicates():

    subject_list = ["m292", "m294"]
    label_list = [0, 1, 2, 3, 4]

    # Create Test Features
    create_test_features()

    ### Test Time Series Features ###
    filename = "TS_Test_Features_with_Duplicates.csv"
    ts_feature_df_with_duplicates = ts_helpers.import_and_concatenate_datasets(subject_list, [filename], parent_directory = "")
    
    assert has_duplicate_rows(ts_feature_df_with_duplicates) == True

    filename = "TS_Test_Features_no_Duplicates.csv"
    ts_feature_df_no_duplicates = ts_helpers.import_and_concatenate_datasets(subject_list, [filename], parent_directory = "")

    assert has_duplicate_rows(ts_feature_df_no_duplicates) == False


    #### Test Brain Imaging Features ###

    filename = "BI_Test_Features_with_Duplicates.csv"
    bi_feature_df_with_duplicates, _ = bi_helpers.import_and_concatenate_datasets(subject_list, [filename], parent_directory = "")
    
    assert has_duplicate_rows(bi_feature_df_with_duplicates) == True


    filename = "BI_Test_Features_no_Duplicates.csv"
    bi_feature_df_no_duplicates, _ = bi_helpers.import_and_concatenate_datasets(subject_list, [filename], parent_directory = "")

    assert has_duplicate_rows(bi_feature_df_no_duplicates) == False

    ### Test Concatenated and Splitted Dataframes ###

    feature_df = helpers.merge_feature_dfs(ts_feature_df_no_duplicates, bi_feature_df_no_duplicates)

    assert has_duplicate_rows(feature_df) == False

    train_indices, validation_indices = create_test_folds(feature_df)

    train_features_dfs_all_folds, train_labels_all_folds = helpers.filter_dataframe_with_indices(feature_df, train_indices, label_list)
    validation_features_dfs_all_folds, validation_labels_all_folds = helpers.filter_dataframe_with_indices(feature_df, validation_indices, label_list)


    for fold in range(5):
        assert has_duplicate_rows(train_features_dfs_all_folds["Fold_"+str(fold)]) == False

    X_train, y_train, X_test, y_test = helpers.initialize_fold_dicts(train_features_dfs_all_folds, train_labels_all_folds, validation_features_dfs_all_folds, validation_labels_all_folds)

    print(X_train)
    for fold in range(5):
        assert has_duplicate_rows(X_train[fold]) == False
        assert has_duplicate_rows(X_test[fold]) == False
        
        

def test_has_index_or_labels_as_feature():
        
    subject_list = ["m292", "m294"]
    label_list = [0, 1, 2, 3, 4]

    filename = "TS_Test_Features_no_Duplicates.csv"
    ts_feature_df_no_duplicates = ts_helpers.import_and_concatenate_datasets(subject_list, [filename], parent_directory = "")

    filename = "BI_Test_Features_no_Duplicates.csv"
    bi_feature_df_no_duplicates, _ = bi_helpers.import_and_concatenate_datasets(subject_list, [filename], parent_directory = "")


    feature_df = helpers.merge_feature_dfs(ts_feature_df_no_duplicates, bi_feature_df_no_duplicates)


    train_indices, validation_indices = create_test_folds(feature_df)

    train_features_dfs_all_folds, train_labels_all_folds = helpers.filter_dataframe_with_indices(feature_df, train_indices, label_list)
    validation_features_dfs_all_folds, validation_labels_all_folds = helpers.filter_dataframe_with_indices(feature_df, validation_indices, label_list)

    X_train, y_train, X_test, y_test = helpers.initialize_fold_dicts(train_features_dfs_all_folds, train_labels_all_folds, validation_features_dfs_all_folds, validation_labels_all_folds)

    # Is there a column left which contains 0, 1, 2, 3 and 4?
    for col in X_train[0].columns:
        column_contains_all_values =set([0, 1, 2, 3, 4]).issubset(X_train[0][col])

    assert column_contains_all_values == False