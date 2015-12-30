#!/usr/bin/env python

import pandas as pd
import xgboost
import sys

sys.path.append('..')
from sklearn.preprocessing import LabelEncoder
from utils.metrics import ndcg_scorer
from sklearn.grid_search import GridSearchCV


def main():
    path = '../datasets/processed/'
    train_users = pd.read_csv(path + 'train_users.csv')
    test_users = pd.read_csv(path + 'test_users.csv')

    y_train = train_users['country_destination']
    train_users.drop('country_destination', axis=1, inplace=True)
    train_users.drop('id', axis=1, inplace=True)

    x_train = train_users.values

    test_users.drop('id', axis=1, inplace=True)

    label_encoder = LabelEncoder()
    encoded_y_train = label_encoder.fit_transform(y_train)

    xgb_model = xgboost.XGBClassifier(
        objective="multi:softprob",
        nthread=-1,
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        colsample_bylevel=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        seed=42
    )

    clf = GridSearchCV(
        xgb_model,
        {
            'max_depth': [8, 10, 12],
            'n_estimators': [45, 48, 50],
            'learning_rate': [0.18, 0.2, 0.22],
            'subsample': [0.5, 0.6, 0.7],
            'colsample_bytree': [0.5, 0.5, 0.7],
        },
        cv=5,
        verbose=10,
        n_jobs=3,
        scoring=ndcg_scorer
    )

    clf.fit(x_train, encoded_y_train)

    print
    print(clf.grid_scores_)
    print
    print(clf.best_params_)
    print
    print(clf.best_score_)
    print


if __name__ == '__main__':
    main()
