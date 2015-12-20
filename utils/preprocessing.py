"""Utils used in the preprocessing stage"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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


def input_missing_values(data, column, model, scale=True):
    """Input missing values into the DataFrame using SVM.

    Parameters
    ----------
    data :
    column :

    Returns
    -------
    data : DataFrame
        Returns a pandas DataFrame with the feature column filled with
        predictions.
    """
    df = data
    null_values = df[column].isnull()

    if scale:
        X = df.columns.tolist()
        X.remove(column)

        df[X] = StandardScaler().fit_transform(df[X])

    model.fit(
        df[~null_values].drop(column, axis=1),
        df[~null_values][column]
    )

    predicted_values = model.predict(df[null_values].drop(column, axis=1))

    data.loc[null_values, column] = np.around(predicted_values)

    return data
