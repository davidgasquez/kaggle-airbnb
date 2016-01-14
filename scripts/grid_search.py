#!/usr/bin/env python

import pandas as pd
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV

import sys
sys.path.append('..')
from utils.metrics import ndcg_scorer


def main():
    path = '../datasets/processed/'
    train_users = pd.read_csv(path + 'processed_train_users.csv')
    test_users = pd.read_csv(path + 'processed_test_users.csv')

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
        colsample_bytree=1,
        subsample=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        seed=42
    )

    clf = GridSearchCV(
        xgb_model,
        {
            'max_depth': [4, 5, 6],
            'n_estimators': [45, 50, 55],
            'learning_rate': [0.15, 0.2, 0.22],
        },
        cv=10,
        verbose=10,
        n_jobs=1,
        scoring='log_loss'
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
