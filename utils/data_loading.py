import pandas as pd

DEFAULT_PATH = '../data/raw/'


def load_users_data(path=DEFAULT_PATH):
    """Loads users data into train and test users."""
    train_users = pd.read_csv(path + 'train_users.csv')
    test_users = pd.read_csv(path + 'test_users.csv')
    return train_users, test_users


def load_sessions_data(path=DEFAULT_PATH):
    """Loads the users sessions data."""
    return pd.read_csv(path + 'sessions.csv')
