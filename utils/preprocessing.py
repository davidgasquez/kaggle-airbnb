"""Utils used in the preprocessing stage"""

import numpy as np
import pandas as pd


def one_hot_encoding(data, categorical_features):
    """Encode a list of categorical features using a one-hot scheme.

    Parameters
    ----------
    data :
    categorical_features :

    Returns
    -------
    data : DataFrame
        Returns a pandas DataFrame with new binary columns.
    """
    for feature in categorical_features:
        data_dummy = pd.get_dummies(data[feature], prefix=feature)
        data = data.drop([feature], axis=1)
        data = pd.concat((data, data_dummy), axis=1)

    return data


def input_missing_values(data, column):
    """Input missing values into the DataFrame using SVM.

    Parameters
    ----------
    data :
    column :

    Returns
    -------
    data : DataFrame
    Returns a pandas DataFrame with new binary columns.
    """
    null_values = data[column].isnull()

    from sklearn.svm import SVR
    clf = SVR()

    clf.fit(
        data[~null_values].drop(column, axis=1),
        data[~null_values][column]
    )

    predicted_values = clf.predict(data[null_values].drop(column, axis=1))

    data.loc[null_values, column] = np.around(predicted_values)

    return data
