"""Metrics to compute the model performance."""

import numpy as np


def dcg_score(y_true, y_pred, k=5):
    """Discounted Cumulative Gain (DCG) at rank K.

    Parameters
    ----------
    y_true : 1d array-like, shape = [n_samples]
        Ground truth (correct) labels.
    y_pred : 2d array-like, shape = [n_samples, k]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float

    Examples
    --------
    >>> y_true = ['FR', 'GR']
    >>> y_pred = [['FR', 'ES', 'PT'], ['US', 'GR']]
    >>> dcg_score(y_true, y_pred)
    """
    score = 0
    for y_true_value, y_pred_array in zip(y_true, y_pred):
        for i in xrange(min(len(y_pred_array), k)):
            numerator = 2**(y_true_value == y_pred_array[i]) - 1
            denominator = np.log2(i + 1 + 1)
            score += numerator / denominator
    return score


def ndcg_score(y_true, y_pred, k=5):
    """Normalized Discounted Cumulative Gain (NDCG) at rank K.

    Normalized Discounted Cumulative Gain (NDCG) measures the performance of a
    recommendation system based on the graded relevance of the recommended
    entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal
    ranking of the entities.

    Parameters
    ----------
    y_true : 1d array-like, shape = [n_samples]
        Ground truth (correct) labels.
    y_pred : 2d array-like, shape = [n_samples, k]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float

    Examples
    --------
    >>> y_true = ['FR', 'GR']
    >>> y_pred = [['FR', 'ES', 'PT'], ['US', 'GR']]
    >>> ndcg_score(y_true, y_pred))
    """
    actual = dcg_score(y_true, y_pred, k)
    return actual / len(y_true)
