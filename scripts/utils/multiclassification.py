"""Multiclass and multilabel classification strategies.

This file implements an one-vs-one multiclass learning algorithm that uses
over sampling algorithms and variances of the decision function.
"""
from __future__ import division

import numpy as np

from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import _fit_binary, check_is_fitted
from sklearn.multiclass import _ovr_decision_function, _predict_binary
from sklearn.externals.joblib import Parallel, delayed
from unbalanced_dataset import SMOTE, SMOTEENN, OverSampler
from unbalanced_dataset import UnderSampler, TomekLinks
from sklearn.neighbors import NearestNeighbors


def _score_matrix(probabilities, n_classes):
    """Create a probability matrix representing the probability of each class.

    Parameters
    ----------
    probabilities : array, shape = [n_classifiers]
        Vector containing the predicted probabilities for the positive class
        for each classifier.
    n_classes : int
        Number of classes.

    Returns
    -------
    probability_matrix : array of shape = [n_classes, n_classes]
        The class probabilities of the input sample as an antisymetric matrix.

    Example
    -------
    >>> probabilities = [0.9, 0.8, 0.2, 0.5, 0.3, 0.1]
    >>> _score_matrix(probabilities, 4)
    array([[ 0. ,  0.9,  0.8,  0.2],
           [ 0.1,  0. ,  0.5,  0.3],
           [ 0.2,  0.5,  0. ,  0.1],
           [ 0.8,  0.7,  0.9,  0. ]])
    """
    # Make empty matrix
    matrix = np.zeros((n_classes, n_classes))

    # Fill upper triangle with the vector
    matrix[np.triu_indices(n_classes, 1)] = probabilities

    # Fill lower triangle with the difference to one of the upper one
    for i in range(n_classes):
        for j in range(i, n_classes):
            matrix[j][i] = 1 - matrix[i][j]

    # Set diagonal to zero
    np.fill_diagonal(matrix, 0)

    return matrix


def _sample_values(X, y, method=None, ratio=1, verbose=False):
    """Perform any kind of sampling(over and under).

    Parameters
    ----------
    X : array, shape = [n_samples, n_features]
        Data.
    y : array, shape = [n_samples]
        Target.
    method : str, optional default: None
        Over or under smapling method.
    ratio: float
        Unbalanced class ratio.

    Returns
    -------
    X, y : tuple
        Sampled X and y.
    """
    # TODO: Add kwargs
    if method == 'SMOTE':
        sampler = SMOTE(ratio=ratio, verbose=verbose)

    elif method == 'SMOTEENN':
        ratio = ratio * 0.3
        sampler = SMOTEENN(ratio=ratio, verbose=verbose)

    elif method == 'random_over_sample':
        sampler = OverSampler(ratio=ratio, verbose=verbose)

    elif method == 'random_under_sample':
        sampler = UnderSampler(verbose=verbose)

    elif method == 'TomekLinks':
        sampler = TomekLinks(verbose=verbose)

    return sampler.fit_transform(X, y)


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

        # Class inbalancing ratio
        ratio = abs(ones - zeros) / min(ones, zeros)

        # Sample X and y
        X_values, y_values = _sample_values(
            X_values, y_values, method=sampling, ratio=ratio)

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

    def __init__(self, estimator, n_jobs=1, sampling=None,
                 strategy='vote', verbose=False):
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
        sampling : str, optional default: None
            Samplig method to use when fitting each estimator.
        """
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.sampling = sampling
        self.verbose = verbose
        self.strategy = strategy

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
        return self.decision_function(X)

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
        valid_strategies = ('vote', 'weighted_vote',
                            'dynamic_vote', 'relative_competence')
        if self.strategy not in valid_strategies:
            raise ValueError('Strategy %s is not valid.' % (self.strategy))

        valid_sampling_methods = ('SMOTE', 'SMOTEENN', 'random_over_sample',
                                  'random_under_sample', 'TomekLinks', None)
        if self.sampling not in valid_sampling_methods:
            raise ValueError('Sampling %s is not valid.' % (self.sampling))

        y = np.asarray(y)

        self.X = X
        self.y = y
        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_ovo_binary)(
                self.estimator, X, y, self.classes_[i], self.classes_[j],
                sampling=self.sampling, verbose=self.verbose
            ) for i in range(n_classes) for j in range(i + 1, n_classes))

        return self

    def decision_function(self, X):
        """Decision function for the CustomOneVsOneClassifier.

        By default, the decision values for the samples are computed by adding
        the normalized sum of pair-wise classification confidence levels to the
        votes in order to disambiguate between the decision values when the
        votes for all the classes are equal leading to a tie.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        Y : array-like, shape = [n_samples, n_classes]
        """
        check_is_fitted(self, 'estimators_')

        predictions = np.vstack([est.predict(X) for est in self.estimators_]).T
        confidences = np.vstack([_predict_binary(est, X)
                                 for est in self.estimators_]).T

        n_clases = len(self.classes_)

        if self.strategy in ('weighted_vote', 'dynamic_vote',
                             'relative_competence'):
            # Compute matrix with classes probabilities
            scores = [_score_matrix(c, n_clases) for c in confidences]

            if self.strategy == 'dynamic_vote':
                scores = self._dynamic_ovo(scores, X, n_clases)

            elif self.strategy == 'relative_competence':
                raise NotImplementedError('Strategy %s not implemented.'
                                          % (self.strategy))

            # Sum of each probability column representing each class
            votes = np.vstack([np.sum(m, axis=0) for m in scores])

            return votes

        elif self.strategy == 'vote':
            return _ovr_decision_function(predictions, confidences,
                                          n_clases)

    def _dynamic_ovo(self, scores, x, n_classes):
        """Dynamic One vs One classifier selection strategy.

        Dynamic classifier selection strategy for One vs One scheme tries to
        avoid the non-competent classifiers when their output is probably not
        of interest considering the neighborhood of each instance to decide
        whether a classifier may be competent or not.

        References
        ----------
        Mikel Galar, Alberto Fernandez, Edurne Barrenechea, Humberto Bustince,
        and Francisco Herrera. Dynamic classifier selection for One-vs-One
        strategy: Avoiding non-competent classifiers. 2013.
        """
        # Select all the neighborhood
        k = n_classes * 6

        # Fit the training data
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        neigh.fit(self.X)

        # Compute the indices of the k neighbors for x
        n = neigh.kneighbors(x, return_distance=False)

        # Get the unique classes of each neighbors
        c = map(self._get_neighbors_classes, n)

        # Select the column classes in the score matrices
        # that appears into the neighborhood.
        for i, score in enumerate(scores):

            # If there is only one class, return the same score
            # matrix as this sample will be properly classified
            if len(c[i]) == 1:
                continue

            mask = np.ones(n_classes, dtype=bool)
            mask[c[i]] = False

            # Apply mask to score matrix rows and columns
            score[:, mask] = score[:, mask] * 0.1
            score[mask, :] = score[mask, :] * 0.1

        return scores

    def _get_neighbors_classes(self, n):
        """Extract unique classes for the heighborhood.

        Parameters
        ----------
        n : array-like, shape = [n_samples, n_neighbors]
            Indices of the instance neighbors
        """
        n_classes = len(self.classes_)

        # Set limits to explore the neighborhood
        lower_bound = n_classes * 3
        upper_bound = n_classes * 6

        # Go throught the neighborhood while there is only one class or the
        # upper limit is not reached
        for x in range(lower_bound, upper_bound):
            neighbors_classes = np.unique(self.y[n[:x]])

            # Exit the loop if we have found more classes in the neighborhood
            if len(neighbors_classes) > 1:
                return neighbors_classes
        else:
            return neighbors_classes
