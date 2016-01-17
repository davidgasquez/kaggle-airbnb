#!/usr/bin/env python

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier

import sys
sys.path.append('..')
from utils.metrics import ndcg_scorer
from utils.multiclassification import CustomOneVsOneClassifier


def main():
    path = '../datasets/processed/'
    train_users = pd.read_csv(path + 'processed_train_users.csv')

    y_train = train_users['country_destination']
    train_users.drop('country_destination', axis=1, inplace=True)
    train_users.drop('id', axis=1, inplace=True)
    train_users = train_users.fillna(-1)

    x_train = train_users.values

    label_encoder = LabelEncoder()
    encoded_y_train = label_encoder.fit_transform(y_train)

    xgb = XGBClassifier(
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        missing=None,
        silent=True,
        nthread=-1,
        seed=42
    )

    clf = CustomOneVsOneClassifier(xgb, sampling='SMOTE')

    gs_clf = GridSearchCV(
        clf,
        {
            'estimator__max_depth': [2, 4, 6],
            'estimator__n_estimators': [4, 6, 8, 10],
            'estimator__learning_rate': [0.1],
        },
        cv=5,
        verbose=10,
        n_jobs=1,
        scoring=ndcg_scorer
    )

    gs_clf.fit(x_train, encoded_y_train)

    print
    print(gs_clf.grid_scores_)
    print
    print(gs_clf.best_params_)
    print
    print(gs_clf.best_score_)
    print


if __name__ == '__main__':
    main()
