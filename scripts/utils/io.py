"""Wrappers to simplify data loading."""

import pandas as pd
import numpy as np
import datetime

# Set default path
DEFAULT_PATH = '../data/'


def load_users_data(path=DEFAULT_PATH, preprocessed=False, prefix=''):
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
        path += 'raw/'
        train_users = pd.read_csv(path + 'train_users.csv')
        test_users = pd.read_csv(path + 'test_users.csv')
    else:
        path += 'processed/'
        train_users = pd.read_csv(path + prefix + 'train_users.csv')
        test_users = pd.read_csv(path + prefix + 'test_users.csv')
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
    return pd.read_csv(path + 'raw/' + 'sessions.csv')


def generate_submission(y_pred, test_users_ids, label_encoder,
                        name='submission'):
    """Create a valid submission file given the predictions.

    The file is created with the current date appended on the name. The output
    file follows the structure:

        id, country
        001, NDF
        001, FR
        001, GB
        002, US
        002, NDF
        002, ES

    Parameters
    ----------
    y_pred: array, shape = [n_samples, n_classes]
        Probabilities of each class as predicted by any classifier.
    test_users_ids: array, shape = [n_samples]
        ID's of each sample of y_pred.
    label_encoder: LabelEncoder object
        Fitted LabelEncoder object.
    name: str
        Name of the output file.
    """
    ids = []
    cts = []
    for i in range(len(test_users_ids)):
        idx = test_users_ids[i]
        ids += [idx] * 5
        sorted_countries = np.argsort(y_pred[i])[::-1]
        cts += label_encoder.inverse_transform(sorted_countries)[:5].tolist()

    id_stacks = np.column_stack((ids, cts))
    submission = pd.DataFrame(id_stacks, columns=['id', 'country'])

    date = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
    name = name + str(date) + '.csv'

    return submission.to_csv('../data/submissions/' + name, index=False)
