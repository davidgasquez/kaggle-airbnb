#!/usr/bin/env python

import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append('..')
from utils.metrics import ndcg_scorer


def generate_submission(y_pred, test_users_ids, label_encoder):
    """Create a valid submission file given the predictions."""
    ids = []
    cts = []
    for i in range(len(test_users_ids)):
        idx = test_users_ids[i]
        ids += [idx] * 5
        sorted_countries = np.argsort(y_pred[i])[::-1]
        cts += label_encoder.inverse_transform(sorted_countries)[:5].tolist()

    sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
    return sub


def main():
    path = '../datasets/processed/'
    train_users = pd.read_csv(path + 'processed_train_users.csv')
    test_users = pd.read_csv(path + 'processed_test_users.csv')

    y_train = train_users['country_destination']
    train_users.drop('country_destination', axis=1, inplace=True)
    train_users.drop('id', axis=1, inplace=True)
    train_users = train_users.fillna(-1)

    x_train = train_users.values

    test_users_ids = test_users['id']
    test_users.drop('id', axis=1, inplace=True)
    test_users = test_users.fillna(-1)

    x_test = test_users.values

    label_encoder = LabelEncoder()
    encoded_y_train = label_encoder.fit_transform(y_train)

    clf = RandomForestClassifier(
        max_depth=3,
        max_features=None,
        n_estimators=20,
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

    clf.fit(x_train, encoded_y_train)

    y_pred = clf.predict_proba(x_test)

    submission = generate_submission(y_pred, test_users_ids, label_encoder)

    date = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
    name = __file__.split('.')[0] + '_' + str(date) + '.csv'
    submission.to_csv('../datasets/submissions/' + name, index=False)

    ndcg = cross_val_score(clf, x_train, encoded_y_train,
                           verbose=10, cv=10, scoring=ndcg_scorer)

    print 'Score:', ndcg.mean()


if __name__ == '__main__':
    main()
