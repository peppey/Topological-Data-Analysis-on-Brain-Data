import import_ipynb
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import mlflow


import Utils.Time_Series_Classification_Helpers as ts_helpers
import Utils.Brain_Imaging_Classification_Helpers as bi_helpers



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

        fold_df = fold_df.drop(columns=fold_df.filter(like='Label').columns)

        feature_df_all_folds["Fold_"+str(fold_idx)] = fold_df
        
    return feature_df_all_folds, all_labels



def filter_fold_dependant_dataframe_with_indices(fold_dependant_feature_df, indices_dict_all_subjects, label_list, n_folds = 5):
    
    feature_df_all_folds = {}
    all_labels = {}


    for fold_idx in range(n_folds):

        filtered_fold_df = fold_dependant_feature_df[fold_dependant_feature_df["Fold"] == fold_idx]

        fold_df = return_fold_df(filtered_fold_df, indices_dict_all_subjects, label_list, fold_idx)

        all_labels["Fold_"+str(fold_idx)] = fold_df["Label"]


        fold_df = fold_df.drop(columns=fold_df.filter(like='Label').columns)

        feature_df_all_folds["Fold_"+str(fold_idx)] = fold_df

    return feature_df_all_folds, all_labels



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


def create_final_input_data_dicts(feature_df, final_train_indices, test_indices, label_list):

    np.random.seed(42)


    X_train = pd.DataFrame()
    y_train = []
    X_test = pd.DataFrame()
    y_test = []

            
    for subject_idx, subject in enumerate(test_indices.keys()):
        
        filtered_subject_df = feature_df[feature_df["Subject"] == subject_idx]
        
        for label in label_list:

            filtered_label_df = filtered_subject_df.loc[filtered_subject_df["Label"] == label]

            filtered_label_df.reset_index(drop=True, inplace=True)

            train_indices_in_filtered_df = final_train_indices[subject]["Label_"+str(label)]
            test_indices_in_filtered_df = test_indices[subject]["Label_"+str(label)]

            feature_df_with_train_indices = filtered_label_df.iloc[train_indices_in_filtered_df]
            feature_df_with_test_indices = filtered_label_df.iloc[test_indices_in_filtered_df]

            
            y_train.extend(list(feature_df_with_train_indices["Label"]))
            y_test.extend(list(feature_df_with_test_indices["Label"]))

            feature_df_with_train_indices = feature_df_with_train_indices.drop(columns=feature_df_with_train_indices.filter(like='Label').columns)
            feature_df_with_test_indices = feature_df_with_test_indices.drop(columns=feature_df_with_test_indices.filter(like='Label').columns)

            X_train = pd.concat([X_train, feature_df_with_train_indices], ignore_index=True)
            X_test = pd.concat([X_test, feature_df_with_test_indices], ignore_index=True)


    # Shuffle X_train and y_train together
    combined_train = list(zip(X_train.values, y_train))
    np.random.shuffle(combined_train)
    X_train_shuffled, y_train_shuffled = zip(*combined_train)
    X_train_shuffled = pd.DataFrame(X_train_shuffled, columns=X_train.columns)

    # Shuffle X_test and y_test together
    combined_test = list(zip(X_test.values, y_test))
    np.random.shuffle(combined_test)
    X_test_shuffled, y_test_shuffled = zip(*combined_test)
    X_test_shuffled = pd.DataFrame(X_test_shuffled, columns=X_test.columns)

    return X_train_shuffled, list(y_train_shuffled), X_test_shuffled, list(y_test_shuffled)


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
    return average_accuracy


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


def create_and_merge_feature_dataframes(subject_list, time_series_filenames, brain_imagining_filenames):
    """
    Create and merge time series and brain imaging feature dataframes.

    Parameters:
    subject_list (list): List of subjects for data import.
    time_series_filenames (tuple): Tuple containing lists of filenames for time series features.
    brain_imagining_filenames (tuple): Tuple containing lists of filenames for brain imaging features.
    ts_helpers (module): Module containing helper functions for time series data processing.
    create_brain_imaging_feature_df (function): Function for creating brain imaging feature dataframe.
    helpers (module): Module containing helper functions for merging dataframes.

    Returns:
    tuple: A tuple containing three merged feature dataframes:
           - feature_df
           - fold_dependant_feature_df
           - fold_dependant_final_test_feature_df
    """
    # Create time series feature dataframes
    all_time_series_dataframes = ts_helpers.create_time_series_feature_dfs(subject_list, time_series_filenames)
    
    # Create brain imaging feature dataframes
    all_brain_imaging_dataframes = bi_helpers.create_brain_imaging_feature_df(subject_list, brain_imagining_filenames)
    
    # Merge the time series and brain imaging dataframes
    feature_df = merge_feature_dfs(all_time_series_dataframes[0], all_brain_imaging_dataframes[0])
    fold_dependant_feature_df = merge_feature_dfs(all_time_series_dataframes[1], all_brain_imaging_dataframes[1])
    fold_dependant_final_test_feature_df = merge_feature_dfs(all_time_series_dataframes[2], all_brain_imaging_dataframes[2])

    feature_df.fillna(0, inplace=True)
    
    return feature_df, fold_dependant_feature_df, fold_dependant_final_test_feature_df


def remove_features(feature_dfs, strings_to_remove):
    """
    Remove columns from dataframes whose column names contain specified substrings.

    Parameters:
    feature_dfs (list of pd.DataFrame): List of feature dataframes to process.
    strings_to_remove (list of str): List of substrings to identify columns for removal.

    Returns:
    list of pd.DataFrame: List of feature dataframes with specified columns removed.
    """

    if len(strings_to_remove) == 0:
        print("No features removed.")
        return feature_dfs

    print("There were "+str(len(feature_dfs[0].columns)+len(feature_dfs[1].columns))+" features before filtering.")
    
    filtered_dfs =  [remove_columns_with_str(df, strings_to_remove) for df in feature_dfs]

    print("There are "+str(len(filtered_dfs[0].columns)+len(filtered_dfs[1].columns))+" features after filtering.")

    return filtered_dfs


def only_use_one_feature_for_classification(feature, feature_dfs):
    """
    Keep only one feature. Useful for single-feature accuracy.

    Parameters:
    feature (str): The feature to keep for classification.
    feature_dfs (list of pd.DataFrame): List of dataframes (e.g., feature_df, fold_dependant_feature_df, etc.).

    Returns:
    list of pd.DataFrame: List of dataframes with only the relevant columns retained.
    """
    columns_to_keep_pattern = f"{feature}|Subject|Label|Fold"
    
    # Filter columns for each dataframe in the list
    filtered_dfs = [
        df[df.columns[df.columns.str.contains(columns_to_keep_pattern)]] for df in feature_dfs
    ]
    
    return filtered_dfs


def remove_duplicate_columns_from_data(X_train, X_test):
    """
    Remove duplicate columns from training and test datasets for each fold.

    Parameters:
    X_train (list of pd.DataFrame): List of training datasets, where each element is a DataFrame for a fold.
    X_test (list of pd.DataFrame): List of test datasets, where each element is a DataFrame for a fold.

    Returns:
    tuple: (X_train, X_test) with duplicate columns removed from each DataFrame.
    """
    for fold in range(len(X_train)):
        # For columns with duplicate content, only keep the first one
        X_train[fold] = keep_first_duplicate_columns(X_train[fold])
        X_test[fold] = keep_first_duplicate_columns(X_test[fold])
    
    return X_train, X_test


def create_training_and_validation_sets(feature_df, fold_dependant_feature_df, train_indices, validation_indices, label_list):
    """
    Create training and validation sets, combining ATOL vectorization features with all other features.

    Parameters:
    feature_df (pd.DataFrame): DataFrame containing the main feature set.
    fold_dependant_feature_df (pd.DataFrame): DataFrame containing fold-dependent features.
    train_indices (list or np.array): Indices for training set.
    validation_indices (list or np.array): Indices for validation set.
    label_list (list of str): List of label column names.

    Returns:
    tuple: (X_train, y_train, X_test, y_test) where each is a dictionary of features and labels for each fold.
    """
    # Filter dataframes with training and validation indices
    train_features_dfs_all_folds, train_labels_all_folds = filter_dataframe_with_indices(feature_df, train_indices, label_list)
    validation_features_dfs_all_folds, validation_labels_all_folds = filter_dataframe_with_indices(feature_df, validation_indices, label_list)
    
    # Filter fold-dependent dataframes with training and validation indices
    ATOL_train_features_dfs_all_folds, _ = filter_fold_dependant_dataframe_with_indices(fold_dependant_feature_df, train_indices, label_list)
    ATOL_validation_features_dfs_all_folds, _ = filter_fold_dependant_dataframe_with_indices(fold_dependant_feature_df, validation_indices, label_list)
    
    # Combine ATOL vectorization features with other features
    train_features_dfs_all_folds = combine_all_features(train_features_dfs_all_folds, ATOL_train_features_dfs_all_folds)
    validation_features_dfs_all_folds = combine_all_features(validation_features_dfs_all_folds, ATOL_validation_features_dfs_all_folds)
    
    # Initialize fold dictionaries for training and validation sets
    X_train, y_train, X_test, y_test = initialize_fold_dicts(
        train_features_dfs_all_folds, train_labels_all_folds,
        validation_features_dfs_all_folds, validation_labels_all_folds
    )

    X_train, X_test = remove_duplicate_columns_from_data(X_train, X_test)
    
    return X_train, y_train, X_test, y_test


def concatenate_data(X_train, X_test, y_train, y_test):
    """
    Concatenate train and test data across multiple folds for GridSearchCV compatibility.

    Parameters:
    X_train (list of pd.DataFrame): List of training datasets for each fold.
    X_test (list of pd.DataFrame): List of test datasets for each fold.
    y_train (list of lists or pd.Series): List of training labels for each fold.
    y_test (list of lists or pd.Series): List of test labels for each fold.

    Returns:
    concatenated_X (pd.DataFrame): Concatenated feature dataset.
    concatenated_y (list): Concatenated labels.
    """
    # Concatenate X_train and X_test
    train_dfs = [X_train[i] for i in range(len(X_train))]
    concatenated_X_train = pd.concat(train_dfs, ignore_index=True)
    
    test_dfs = [X_test[i] for i in range(len(X_test))]
    concatenated_X_test = pd.concat(test_dfs, ignore_index=True)
    
    concatenated_X = pd.concat([concatenated_X_train, concatenated_X_test], ignore_index=True)

    # Concatenate y_train and y_test
    concatenated_y_train = y_train[0] + y_train[1] +  y_train[2] + y_train[3] + y_train[4]
    concatenated_y_test = y_test[0] + y_test[1] +  y_test[2] + y_test[3] + y_test[4]
    concatenated_y = concatenated_y_train + concatenated_y_test


    return concatenated_X, concatenated_y


def define_fold_start_and_end_indices(X_train, X_test):
    """
    Define the start and end indices for each fold's training and test sets.

    Parameters:
    X_train (list of pd.DataFrame): List of training datasets for each fold.
    X_test (list of pd.DataFrame): List of test datasets for each fold.

    Returns:
    train_test_splits (list of tuples): A list of tuples where each tuple contains 
                                        the training indices and test indices for each fold.
    """
    train_test_splits = []
    current_train_start = 0
    current_test_start = sum(len(X_train[i]) for i in range(len(X_train)))  # Total train data length

    # Loop through each fold to define indices
    for i in range(len(X_train)):
        train_len = len(X_train[i])
        test_len = len(X_test[i])

        # Define training and testing indices for the current fold
        indices_for_training = np.arange(current_train_start, current_train_start + train_len)
        test_indices = np.arange(current_test_start, current_test_start + test_len)

        train_test_splits.append((indices_for_training, test_indices))

        # Update start positions for the next fold
        current_train_start += train_len
        current_test_start += test_len

    return train_test_splits


class CustomCV:
    def __init__(self, train_test_splits):
        self.train_test_splits = train_test_splits
    
    def split(self, X, y=None, groups=None):
        for train_idx, test_idx in self.train_test_splits:
            yield train_idx, test_idx
    
    def get_n_splits(self, X, y, groups=None):
        return len(self.train_test_splits)



def perform_grid_search(model, param_grid, custom_cv, concatenated_X, concatenated_y, scoring='accuracy', verbose=3):
    """
    Perform grid search with cross-validation and return the best parameters and score.

    Parameters:
    model (estimator): The machine learning model (e.g., RandomForestClassifier).
    param_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
    custom_cv (cross-validation generator): The cross-validation splitting strategy.
    concatenated_X (pd.DataFrame or np.array): Concatenated training and test feature data.
    concatenated_y (pd.Series or np.array): Concatenated target data.
    scoring (str): Scoring metric to evaluate the model (default is 'accuracy').
    verbose (int): Controls the verbosity of the output (default is 3).

    Returns:
    best_params (dict): Best parameter values found during the grid search.
    best_score (float): Best score corresponding to the best parameter set.
    """
    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=custom_cv, scoring=scoring, verbose=verbose)
    
    # Fit the grid search on the concatenated dataset
    grid_search.fit(concatenated_X, concatenated_y)

    # Extract the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Output best parameters and score
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)
    
    return best_params, best_score


def save_results_to_mlflow(params, X_train, average_accuracy, all_accuracies):
    """
    Save model parameters, accuracy metrics, and artifacts to MLflow.

    Parameters:
    params (dict): Hyperparameters or other configuration parameters.
    X_train (pd.DataFrame): Training dataset (features) from which column names will be extracted.
    average_accuracy (float): The average accuracy of the model across folds.
    all_accuracies (list of floats): List containing accuracy values for each fold.
    """
    # Extract features and update the parameters dictionary
    mlflow_params = params.copy()
    features = X_train[0].columns
    mlflow_params["features"] = features

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(mlflow_params)

        # Log accuracy metrics
        mlflow.log_metric("accuracy", average_accuracy)
        mlflow.log_metric("minimal accuracy", np.min(all_accuracies))
        mlflow.log_metric("maximal accuracy", np.max(all_accuracies))

        # Log artifacts (files)
        mlflow.log_artifact("Features/All_Train_Features.npy")
        mlflow.log_artifact("Features/All_Train_Labels.npy")
        mlflow.log_artifact("Features/All_Test_Features.npy")
        mlflow.log_artifact("Features/All_Test_Labels.npy")

        # Set tags for additional run information
        mlflow.set_tag("Training Info", "Random Forest - Both Modalities")
        mlflow.set_tag('mlflow.runName', 'Random Forest - Both Modalities')



def manual_tuning(model, X_train, X_test, y_train, y_test, params_for_manual_tuning):
    """
    Perform manual tuning of a RandomForestClassifier with specified parameters and evaluate on multiple folds.

    Parameters:
    X_train (list of pd.DataFrame): List of training datasets for each fold.
    X_test (list of pd.DataFrame): List of test datasets for each fold.
    y_train (list of pd.Series or lists): List of training labels for each fold.
    y_test (list of pd.Series or lists): List of test labels for each fold.
    params (dict): Dictionary of hyperparameters to pass into RandomForestClassifier.

    Returns:
    average_accuracy (float): The average accuracy across all folds.
    all_accuracies (list of floats): Accuracies for each individual fold.
    """
    # Initialize the RandomForestClassifier with the provided parameters
    all_accuracies = []

    # Loop through each fold to train and evaluate the model
    for fold in range(len(X_train)):
        # Fit the model on the training data
        model.fit(X_train[fold], y_train[fold])

        # Predict on the test data
        y_pred = model.predict(X_test[fold])

        # Calculate the accuracy for the current fold
        accuracy = accuracy_score(y_test[fold], y_pred)
        all_accuracies.append(accuracy)

        # Print the accuracy for the current fold
        print(f"Accuracy for fold {fold + 1}: {accuracy}")

    # Calculate the average accuracy across all folds
    average_accuracy = np.mean(all_accuracies)

    # Print the average accuracy
    print(f"Average Accuracy: {average_accuracy}")

    save_results_to_mlflow(params_for_manual_tuning, X_train, average_accuracy, all_accuracies)

    pass

def get_indices_of_final_training_set(train_indices, validation_indices, subject_list, label_list):
    """
    The new training data consists of the previous training plus the previous validation data
    """

    final_train_indices = {}

    for subject in subject_list:

        # Initialize
        final_train_indices[subject] = {}
        
        train_indices_for_subject = train_indices[subject]
        validation_indices_for_subject = validation_indices[subject]

        for label in label_list:
            # It does not matter which fold we choose, so simply choose fold 0
            train_indices_to_combine = train_indices_for_subject["Label_"+str(label)]["Fold_0"]
            validation_indices_to_combine = validation_indices_for_subject["Label_"+str(label)]["Fold_0"]
            final_train_indices[subject]["Label_"+str(label)] = np.concatenate((train_indices_to_combine, validation_indices_to_combine))


    return final_train_indices



def final_evaluation(model, params, X_train_final, y_train_final, X_test_final, y_test_final, n_seeds=10):
    """
    Evaluate the performance of a RandomForestClassifier across multiple random seeds.

    Parameters:
    X_train_final (pd.DataFrame): Training feature set.
    y_train_final (pd.Series): Training labels.
    X_test_final (pd.DataFrame): Test feature set.
    y_test_final (pd.Series): Test labels.
    n_seeds (int): Number of different random seeds to test. Default is 10.

    Returns:
    tuple: (mean_accuracy, std_accuracy) where mean_accuracy is the mean accuracy across all seeds,
           and std_accuracy is the standard deviation of accuracies across all seeds.
    """
    final_accuracies = []

    importances = []

    
    for seed in range(n_seeds):

        # Update the seed/random_state entry
        seed_key = next(iter(params))
        params[seed_key] = seed

        model.set_params(**params)

        # Fit the model and predict
        model.fit(X_train_final, y_train_final)
        y_pred = model.predict(X_test_final)
        accuracy = accuracy_score(y_test_final, y_pred)
        
        # Print the accuracy for this seed
        print(f"Accuracy for seed {seed}: {accuracy}")
        
        final_accuracies.append(accuracy)

        importances.append(model.feature_importances_)

    
    # Calculate mean and standard deviation of accuracies
    mean_accuracy = np.mean(final_accuracies)
    std_accuracy = np.std(final_accuracies)
    
    print(f"Mean accuracy: {mean_accuracy}, with standard deviation: {std_accuracy}.")

    return mean_accuracy, importances



def compute_decision_tree_feature_importance(importances, X_train_final, feature):
    """
    Compute the mean and standard deviation of feature importances for a given feature across multiple importance arrays.

    Parameters:
    importances (list of np.ndarray): List of arrays where each array contains feature importances from a decision tree model.
    X_train_final (pd.DataFrame): DataFrame containing the training features.
    feature (str): The specific feature name for which the importances are computed.

    Returns:
    tuple: (mean_importance, std_importance) where mean_importance is the mean of the computed feature importances,
           and std_importance is the standard deviation of the computed feature importances.
    """
    importances_of_single_feature = []

    # Iterate through each importance array
    for imp in importances:
        # Create a boolean mask for the feature columns
        mask = X_train_final.columns.str.contains(feature)
        # Use numpy to get the indices where the mask is True
        indices = np.where(mask)[0]
        # Sum the importances of the specified feature
        importances_of_single_feature.append(sum(imp[indices]))

    # Compute mean and standard deviation
    mean_importance = np.mean(importances_of_single_feature)
    std_importance = np.std(importances_of_single_feature)

    print(f"Mean feature importance: {mean_importance} for feature {feature}, with standard deviation: {std_importance}.")
    
    pass
