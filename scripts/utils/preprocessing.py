"""Utils used in the preprocessing stage."""

import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier


def one_hot_encoding(data, categorical_features):
    """Encode a list of categorical features using a one-hot scheme.

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing the categorical features.
    categorical_features : list
        Names of the categorial features to encode

    Returns
    -------
    data : DataFrame
        Returns a pandas DataFrame with one-hot scheme binary columns.
    """
    for feature in categorical_features:
        data_dummy = pd.get_dummies(data[feature], prefix=feature)
        data.drop([feature], axis=1, inplace=True)
        data = pd.concat((data, data_dummy), axis=1)

    return data


def get_weekday(date):
    """Compute the weekday of the given date."""
    # TODO: Use operator getter
    return date.weekday()


class XGBFeatureSelection(XGBClassifier):
    """A custom XGBClassifier with feature importances computation.

    This class implements XGBClassifier and also computes feature importances
    based on the fscores. Implementing feature_importances_ property allow us
    to use `SelectFromModel` with XGBClassifier.
    """

    def __init__(self, n_features, *args, **kwargs):
        """Init method adding n_features."""
        super(XGBFeatureSelection, self).__init__(*args, **kwargs)
        self._n_features = n_features

    @property
    def n_features(self):
        """Number of classes to predict."""
        return self._n_features

    @n_features.setter
    def n_features(self, value):
        self._n_features = value

    @property
    def feature_importances_(self):
        """Return the feature importances.

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        booster = self.booster()
        fscores = booster.get_fscore()

        importances = np.zeros(self.n_features)

        for k, v in fscores.iteritems():
            importances[int(k[1:])] = v

        return importances
