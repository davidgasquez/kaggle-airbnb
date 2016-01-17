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
    test_users = pd.read_csv(path + 'processed_test_users.csv')

    y_train = train_users['country_destination']
    train_users.drop('country_destination', axis=1, inplace=True)
    train_users.drop('id', axis=1, inplace=True)

    x_train = train_users.values

    test_users.drop('id', axis=1, inplace=True)

    label_encoder = LabelEncoder()
    encoded_y_train = label_encoder.fit_transform(y_train)

    clf = RandomForestClassifier(
        criterion='gini',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=True,
        class_weight=None,
        n_jobs=-1,
        random_state=42
    )

    grid_search = GridSearchCV(
        clf,
        {
            'max_depth': [None, 8, 10],
            'n_estimators': [30, 40, 50],
            'max_features': ['auto', 'log2', None],
        },
        cv=5,
        verbose=10,
        n_jobs=1,
        scoring=ndcg_scorer
    )

    grid_search.fit(x_train, encoded_y_train)

    print
    print(grid_search.grid_scores_)
    print
    print(grid_search.best_params_)
    print
    print(grid_search.best_score_)
    print


if __name__ == '__main__':
    main()
