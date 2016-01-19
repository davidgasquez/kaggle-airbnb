"""Multiclass and multilabel classification strategies.

This module implements an one-vs-one multiclass learning algorithm that uses
SMOTE algorithm to over sample the minority class in each fit.
"""
from __future__ import division

import numpy as np

from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import _fit_binary
from sklearn.externals.joblib import Parallel, delayed
from unbalanced_dataset import SMOTE
from unbalanced_dataset import SMOTEENN


def _fit_ovo_binary(estimator, X, y, i, j, sampling=None, verbose=False):
    """Fit a single binary estimator (one-vs-one)."""
    cond = np.logical_or(y == i, y == j)
    y = y[cond]
    y_binary = np.empty(y.shape, np.int)
    y_binary[y == i] = 0
    y_binary[y == j] = 1
    ind = np.arange(X.shape[0])

    X_values = X[ind[cond]]
    y_values = y_binary

    if sampling:
        ones = np.count_nonzero(y_values == 1)
        zeros = np.count_nonzero(y_values == 0)

        if sampling == 'SMOTE':
            ratio = abs(ones - zeros) / min(ones, zeros)
            smote = SMOTE(ratio=ratio, verbose=verbose)

        if sampling == 'SMOTEENN':
            ratio = (abs(ones - zeros) / min(ones, zeros)) * 0.3
            smote = SMOTEENN(ratio=ratio, verbose=verbose)

        X_values, y_values = smote.fit_transform(X_values, y_values)

    return _fit_binary(estimator, X_values, y_values, classes=[i, j])


class CustomOneVsOneClassifier(OneVsOneClassifier):
    """One-vs-one multiclass strategy.

    This strategy consists in fitting one classifier per class pair.
    At prediction time, the class which received the most votes is selected.

    Requires to fit `n_classes * (n_classes - 1) / 2` classifiers.

    Attributes
    ----------
    estimators_ : list of `n_classes * (n_classes - 1) / 2` estimators
        Estimators used for predictions.
    classes_ : numpy array of shape [n_classes]
        Array containing labels.
    """

    def __init__(self, estimator, n_jobs=1, sampling=None, verbose=False):
        """Init method.

        Parameters
        ----------
        estimator : estimator object
            An estimator object implementing fit and one of decision_function
            or predict_proba.
        n_jobs : int, optional, default: 1
            The number of jobs to use. If -1 all CPUs are used.
            If 1 is given, no parallel computing code is used at all, which is
            useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs)
            are used. Thus for n_jobs = -2, all CPUs but one are used.
        sampling : str, optional default:None
            Samplig method to use when fitting each estimator.
            Can be 'SMOTE' or SMOTEENN'.
        """
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.sampling = sampling
        self.verbose = verbose

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is the same as
        the result of decision_function.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples.
        """
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
                self.estimator, X, y, self.classes_[i], self.classes_[j],
                sampling=self.sampling, verbose=self.verbose
            ) for i in range(n_classes) for j in range(i + 1, n_classes))

        return self
