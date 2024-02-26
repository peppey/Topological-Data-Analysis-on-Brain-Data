import import_ipynb
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay


##### For Preprocessing #####

def keep_first_duplicate_columns(df):
    # Get list of duplicate column names
    duplicate_columns = df.columns[df.columns.duplicated(keep='first')].tolist()
    
    # Drop duplicate columns keeping only the first occurrence
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    return df


def merge_feature_dfs(time_series_feature_df, brain_imaging_feature_df):
    """
    Merge time series and brain imaging feature DataFrames.
    
    Args:
    - time_series_feature_df (pandas.DataFrame): DataFrame containing time series features.
    - brain_imaging_feature_df (pandas.DataFrame): DataFrame containing brain imaging features.
    
    Returns:
    - pandas.DataFrame: Merged DataFrame containing both time series and brain imaging features.
    """
    feature_df = pd.merge(time_series_feature_df, brain_imaging_feature_df, left_index=True, right_index=True, suffixes=('', '_BI'))
    feature_df.reset_index(inplace=True)
    feature_df.drop(columns=["index"], inplace=True)

    return feature_df




##### For Feature Selection #####

def remove_columns_with_str(feature_df, list_of_strings_in_column_name):
    """
    Remove columns from a DataFrame that contain specified strings in their names.

    Parameters:
    - feature_df (DataFrame): The DataFrame from which columns will be removed.
    - list_of_strings_in_column_name (list): List of strings to search for in column names.

    Returns:
    - DataFrame: DataFrame with specified columns removed.
    """
    
    for col_name_string in list_of_strings_in_column_name:
        feature_df.drop(feature_df.columns[feature_df.columns.str.contains(col_name_string,case = False)], axis = 1, inplace = True)

    return feature_df



##### For Train/Test Split #####

def load_folds(subject_list, parent_directory="Time_Series"):
    
    train_indices = {}
    validation_indices = {}
    test_indices = {}


    for subject in subject_list:
        train_path = os.path.join(parent_directory, "Train_Test_Splitting", str(subject), "Train_Indices_All_Labels_All_Folds.npy")
        validation_path = os.path.join(parent_directory, "Train_Test_Splitting", str(subject), "Validation_Indices_All_Labels_All_Folds.npy")
        test_path = os.path.join(parent_directory, "Train_Test_Splitting", str(subject), "Final_Test_Set_Indices_All_Labels.npy")

        train_indices[subject] = np.load(train_path, allow_pickle=True).item()
        validation_indices[subject] = np.load(validation_path, allow_pickle=True).item()
        test_indices[subject] = np.load(test_path, allow_pickle=True).item()

    return train_indices, validation_indices, test_indices


def return_fold_df(filtered_fold_df, indices_dict_all_subjects, label_list, fold_idx):
    fold_df = pd.DataFrame()
        
    for subject_idx, subject in enumerate(indices_dict_all_subjects.keys()):
        filtered_subject_df = filtered_fold_df[filtered_fold_df["Subject"] == subject_idx]

        for label in label_list:                
            filtered_label_df = filtered_subject_df.loc[filtered_subject_df["Label"] == label]
            set_indices_in_filtered_df = indices_dict_all_subjects[subject]["Label_" + str(label)]["Fold_" + str(fold_idx)]

            feature_df_with_set_indices = filtered_label_df.iloc[set_indices_in_filtered_df]
            fold_df = pd.concat([fold_df, feature_df_with_set_indices], ignore_index=True)
    
    return fold_df


def filter_dataframe_with_indices(feature_df, indices_dict_all_subjects, label_list, n_folds=5):
    
    feature_df_all_folds = {}
    all_labels = {}

    for fold_idx in range(n_folds):

        fold_df = return_fold_df(feature_df, indices_dict_all_subjects, label_list, fold_idx)

        all_labels["Fold_"+str(fold_idx)] = fold_df["Label"]

        fold_df.drop(columns=['Label'], inplace=True)

        feature_df_all_folds["Fold_"+str(fold_idx)] = fold_df
        
    return feature_df_all_folds, all_labels



def filter_fold_dependant_dataframe_with_indices(fold_dependant_feature_df, indices_dict_all_subjects, label_list, n_folds = 5):
    
    feature_df_all_folds = {}

    for fold_idx in range(n_folds):

        filtered_fold_df = fold_dependant_feature_df[fold_dependant_feature_df["Fold"] == fold_idx]

        feature_df_all_folds["Fold_"+str(fold_idx)] = return_fold_df(filtered_fold_df, indices_dict_all_subjects, label_list, fold_idx)

    return feature_df_all_folds



def combine_all_features(features_dfs_all_folds, fold_dependant_features_dfs_all_folds, n_folds = 5):

    features_df_all_folds_all_features = {}

    for fold in range(n_folds):
        features_df_all_folds_all_features["Fold_"+str(fold)] = pd.concat([features_dfs_all_folds["Fold_"+str(fold)], fold_dependant_features_dfs_all_folds["Fold_"+str(fold)]], axis=1)
        
    return features_df_all_folds_all_features


def initialize_fold_dicts(train_features_dfs_all_folds, train_labels_all_folds, validation_features_dfs_all_folds, validation_labels_all_folds, n_folds = 5):
    """
    Initialize dictionaries with folds as keys and assign features and labels accordingly.
    
    Args:
    - train_features_dfs_all_folds (dict): Dictionary containing training features for all folds.
    - train_labels_all_folds (dict): Dictionary containing training labels for all folds.
    - validation_features_dfs_all_folds (dict): Dictionary containing validation features for all folds.
    - validation_labels_all_folds (dict): Dictionary containing validation labels for all folds.
    - n_folds (int): Number of folds.
    
    Returns:
    - dict: Dictionary containing training features for each fold.
    - dict: Dictionary containing training labels for each fold.
    - dict: Dictionary containing validation features for each fold.
    - dict: Dictionary containing validation labels for each fold.
    """
    X_train = {}
    y_train = {}
    X_test = {}
    y_test = {}

    for fold in range(n_folds):
        # Set random seed for reproducibility
        np.random.seed(42)

        # Shuffle indices
        indices_train = np.random.permutation(len(train_features_dfs_all_folds["Fold_" + str(fold)]))
        indices_test = np.random.permutation(len(validation_features_dfs_all_folds["Fold_" + str(fold)]))

        # Shuffle rows of X_train[fold] and y_train[fold]
        X_train_fold = train_features_dfs_all_folds["Fold_" + str(fold)].iloc[indices_train]
        y_train_fold = [train_labels_all_folds["Fold_" + str(fold)][index] for index in indices_train]

        # Shuffle rows of X_test[fold] and y_test[fold]
        X_test_fold = validation_features_dfs_all_folds["Fold_" + str(fold)].iloc[indices_test]
        y_test_fold = [validation_labels_all_folds["Fold_" + str(fold)][index] for index in indices_test]

        X_train[fold] = X_train_fold
        y_train[fold] = y_train_fold
        X_test[fold] = X_test_fold
        y_test[fold] = y_test_fold

    return X_train, y_train, X_test, y_test




###### For Training #####
 
### Random Forest ###

def train_rf_cross_validation(X_train, y_train, X_test, y_test, n_estimators=900, n_folds = 5, random_state=5):
    """
    Train RandomForestClassifier using cross-validation, print accuracy for each fold, and calculate average accuracy.
    
    Args:
    - X_train (dict): Dictionary containing training features for each fold.
    - y_train (dict): Dictionary containing training labels for each fold.
    - X_test (dict): Dictionary containing validation features for each fold.
    - y_test (dict): Dictionary containing validation labels for each fold.
    - n_estimators (int): Number of trees in the forest (default=900).
    - random_state (int): Random seed (default=5).
    
    Returns:
    - float: Average accuracy across all folds.
    """
    
    rf = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators)
    all_accuracies = []

    for fold in range(5):
        rf.fit(X_train[fold], y_train[fold])
        y_pred = rf.predict(X_test[fold])
        accuracy = accuracy_score(y_pred, y_test[fold])
        all_accuracies.append(accuracy)
        print("Accuracy for fold", fold + 1, ":", accuracy)

    average_accuracy = np.mean(all_accuracies)
    print("Average Accuracy:", average_accuracy)
    pass


### XGBoost ###

def train_xgb_cross_validation(X_train, y_train, X_test, y_test, seed=41):
    """
    Train XGBoost Classifier using cross-validation and calculate mean accuracy.
    
    Args:
    - X_train (dict): Dictionary containing training features for each fold.
    - y_train (dict): Dictionary containing training labels for each fold.
    - X_test (dict): Dictionary containing validation features for each fold.
    - y_test (dict): Dictionary containing validation labels for each fold.
    - seed (int): Random seed (default=41).
    
    Returns:
    - float: Mean accuracy across all folds.
    """
    model = xgb.XGBClassifier(seed=seed)
    all_accuracies = []

    for fold in range(len(X_train)):
        # Remove duplicate columns
        X_train[fold] = X_train[fold].loc[:, ~X_train[fold].columns.duplicated()]

        model.fit(X_train[fold], y_train[fold])

        X_test[fold] = X_test[fold].loc[:, ~X_test[fold].columns.duplicated()]

        y_pred = model.predict(X_test[fold])
        accuracy = accuracy_score(y_pred, y_test[fold])
        all_accuracies.append(accuracy)
        print("Accuracy for fold", fold, ":", accuracy)

    mean_accuracy = np.mean(all_accuracies)
    print("Mean Accuracy:", mean_accuracy)
    return mean_accuracy


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

def tune_xgb_parameters(X_train, y_train, X_test, y_test, param_grid, n_folds, seed=3):
    """
    Perform hyperparameter tuning for XGBoost Classifier.
    
    Args:
    - X_train (dict): Dictionary containing training features for each fold.
    - y_train (dict): Dictionary containing training labels for each fold.
    - X_test (dict): Dictionary containing validation features for each fold.
    - y_test (dict): Dictionary containing validation labels for each fold.
    - param_grid (dict): Parameter grid for hyperparameter tuning.
    - n_folds (int): Number of folds for cross-validation.
    - seed (int): Random seed (default=3).
    
    Returns:
    - float: Average accuracy over all folds.
    - list: List of best models for each fold.
    """
    model = xgb.XGBClassifier(seed=seed)
    grid_search = GridSearchCV(model, param_grid, cv=n_folds, scoring='accuracy')

    best_models = []
    for fold in range(n_folds):
        X_train_fold = X_train[fold].loc[:, ~X_train[fold].columns.duplicated()]
        X_test_fold = X_test[fold].loc[:, ~X_test[fold].columns.duplicated()]

        grid_search.fit(X_train_fold, y_train[fold])

        best_model = grid_search.best_estimator_
        best_models.append(best_model)

        y_pred = best_model.predict(X_test_fold)
        accuracy = accuracy_score(y_pred, y_test[fold])
        print(f"Fold {fold + 1} - Best Parameters: {grid_search.best_params_}, Accuracy: {accuracy}")

    average_accuracy = np.mean([accuracy_score(y_test[i], best_models[i].predict(X_test[i].loc[:, ~X_test[i].columns.duplicated()])) for i in range(n_folds)])
    print("Average Accuracy:", average_accuracy)
    return average_accuracy, best_models

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
}


### Neural Network ###

# TODO look into old versions

##### Feature Engineering #####

# TODO look into old versions