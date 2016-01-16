#!/usr/bin/env python

"""Gradient Boosting Method."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import DMatrix
import xgboost


def customized_eval(preds, dtrain):
    labels = dtrain.get_label()
    top = []
    for i in range(preds.shape[0]):
        top.append(np.argsort(preds[i])[::-1][:5])
    mat = np.reshape(np.repeat(labels, np.shape(top)[1]) == np.array(
        top).ravel(), np.array(top).shape).astype(int)
    score = np.mean(
        np.sum(mat / np.log2(np.arange(2, mat.shape[1] + 2)), axis=1))
    return 'ndcg5', score


def main():
    """Generate the submission file calling a XGBClassifier."""
    path = '../datasets/processed/'
    train_users = pd.read_csv(path + 'processed_train_users.csv')

    y_train = train_users['country_destination']
    train_users.drop('country_destination', axis=1, inplace=True)
    train_users.drop('id', axis=1, inplace=True)
    train_users = train_users.fillna(-1)

    x_train = train_users.values

    label_encoder = LabelEncoder()
    encoded_y_train = label_encoder.fit_transform(y_train)

    dtrain = DMatrix(x_train, label=encoded_y_train)

    xgb_params = {
        'bst:max_depth': 7,
        'bst:eta': 0.18,
        'silent': 1,
        'objective': 'multi:softprob',
        'num_class': 12
    }

    cv = xgboost.cv(
        xgb_params,
        dtrain,
        num_boost_round=54,
        nfold=5,
        seed=0,
        feval=customized_eval,
        maximize=True,
        show_progress=True
    )

    print np.mean(cv['test-ndcg5-mean'])

if __name__ == '__main__':
    main()
