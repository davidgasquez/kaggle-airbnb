from __future__ import division

import numpy as np

from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import _fit_binary
from sklearn.externals.joblib import Parallel, delayed
from unbalanced_dataset import SMOTE


def _fit_ovo_binary(estimator, X, y, i, j, sampling=None):
    """Fit a single binary estimator (one-vs-one)."""
    cond = np.logical_or(y == i, y == j)
    y = y[cond]
    y_binary = np.empty(y.shape, np.int)
    y_binary[y == i] = 0
    y_binary[y == j] = 1
    ind = np.arange(X.shape[0])

    X_values = X[ind[cond]]
    y_values = y_binary

    if sampling == 'SMOTE':
        ones = np.count_nonzero(y_values == 1)
        zeros = np.count_nonzero(y_values == 0)
        ratio = abs(ones - zeros) / min(ones, zeros)
        smote = SMOTE(ratio=ratio, verbose=True)
        X_values, y_values = smote.fit_transform(X_values, y_values)

    return _fit_binary(estimator, X_values, y_values, classes=[i, j])


class CustomOneVsOneClassifier(OneVsOneClassifier):

    def __init__(self, estimator, n_jobs=1, sampling=None):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.sampling = sampling

    def predict_proba(self, X):
        return super(CustomOneVsOneClassifier, self).decision_function(X)

    def fit(self, X, y):
        """Fit underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.
        y : array-like, shape = [n_samples]
            Multi-class targets.
        Returns
        -------
        self
        """
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_ovo_binary)(
                self.estimator, X, y,
                self.classes_[i], self.classes_[j], sampling=self.sampling)
            for i in range(n_classes) for j in range(i + 1, n_classes))

        return self
