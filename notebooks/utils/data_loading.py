"""Wrappers to simplify data loading."""

import pandas as pd

# Set default path
DEFAULT_PATH = '../data/raw/'


def load_users_data(path=DEFAULT_PATH, preprocessed=False):
    """Load users data into train and test users.

    Parameters
    ----------
    path: str
        Path of the folder containing the data.

    Returns
    -------
    train_users, test_users: DataFrame, DataFrame
        Loaded DataFrames.
    """
    if not preprocessed:
        train_users = pd.read_csv(path + 'train_users.csv')
        test_users = pd.read_csv(path + 'test_users.csv')
    else:
        path = '../datasets/processed/'
        train_users = pd.read_csv(path + 'preprocessed_train_users.csv')
        test_users = pd.read_csv(path + 'preprocessed_test_users.csv')
    return train_users, test_users


def load_sessions_data(path=DEFAULT_PATH):
    """Load the users sessions data.

    Parameters
    ----------
    path: str
        Path of the folder containing the data.

    Returns
    -------
    sessions: DataFrame
        Loaded DataFrame.
    """
    return pd.read_csv(path + 'sessions.csv')
