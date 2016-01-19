#!/usr/bin/env python

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV

import sys
sys.path.append('..')
from utils.metrics import ndcg_scorer


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

    rf = RandomForestClassifier(
        criterion='gini',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False,
        class_weight=None,
        n_jobs=-1,
        random_state=42
    )

    grid = GridSearchCV(
        rf,
        {
            'max_depth': [None, 2, 3, 4, 5],
            'max_features': [None, 'log2', 'auto', 10],
            'n_estimators': [20, 30, 40, 50],
        },
        cv=5,
        verbose=10,
        n_jobs=1,
        scoring=ndcg_scorer
    )

    grid.fit(x_train, encoded_y_train)

    print
    print(grid.grid_scores_)
    print
    print(grid.best_params_)
    print
    print(grid.best_score_)
    print


if __name__ == '__main__':
    main()
