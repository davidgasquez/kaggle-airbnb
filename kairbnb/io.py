"""Wrappers to simplify data input and output."""

import pandas as pd
import numpy as np
import datetime


def load_users(version='', nrows=None, na_values='-unknown-'):
    """Load user data."""
    if version:
        path = '../cache/'
    else:
        path = '../data/'

    train_users = path + 'train_users.csv' + version
    test_users = path + 'test_users.csv' + version

    train_users = pd.read_csv(train_users, nrows=nrows, na_values=na_values)
    test_users = pd.read_csv(test_users, nrows=nrows, na_values=na_values)

    return train_users, test_users


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
    name = name + '_' + str(date) + '.csv'

    return submission.to_csv('../' + name, index=False)
