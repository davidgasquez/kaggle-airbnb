#!/usr/bin/env python

import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier


def generate_submission(y_pred, test_users_ids, label_encoder):
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

    x_train = train_users.values

    test_users_ids = test_users['id']
    test_users.drop('id', axis=1, inplace=True)

    x_test = test_users.values

    label_encoder = LabelEncoder()
    encoded_y_train = label_encoder.fit_transform(y_train)

    xgb = XGBClassifier(
        max_depth=6,
        learning_rate=0.2,
        n_estimators=45,
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
        seed=42
    )

    xgb.fit(x_train, encoded_y_train)

    y_pred = xgb.predict_proba(x_test)

    submission = generate_submission(y_pred, test_users_ids, label_encoder)

    date = datetime.datetime.now().strftime("%m-%d-%H:%M")
    name = __file__.split('.')[0] + '_' + str(date) + '.csv'
    submission.to_csv('../datasets/submissions/' + name, index=False)


if __name__ == '__main__':
    main()
