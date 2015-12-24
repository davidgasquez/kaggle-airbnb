import pandas as pd
import numpy as np

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k=5, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def score_predictions(preds, truth):
    """
    preds: pd.DataFrame
      one row for each observation, one column for each prediction.
      Columns are sorted from left to right descending in order of likelihood.
    truth: pd.Series
      one row for each obeservation.
    """
    assert(len(preds)==len(truth))

    r = pd.DataFrame()
    for col in preds.columns:
        r[col] = (preds[col] == truth)

    score = pd.Series(r.apply(ndcg_at_k, axis=1, reduce=True))
    return np.mean(score)

preds = pd.DataFrame([['US','FR', 'ES'],['FR','US'],['FR','FR']])
truth = pd.Series(['US','US','FR'])

print(score_predictions(preds, truth))
