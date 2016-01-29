#!/usr/bin/env python

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from utils.multiclassification import CustomOneVsOneClassifier
from utils.io import generate_submission


def main():
    path = '../data/processed/'
    prefix = 'processed_'
    suffix = '1'
    train_users = pd.read_csv(path + prefix + 'train_users.csv' + suffix)
    test_users = pd.read_csv(path + prefix + 'test_users.csv' + suffix)

    y_train = train_users['country_destination']
    train_users.drop(['country_destination', 'id'], axis=1, inplace=True)
    train_users = train_users.fillna(-1)
    x_train = train_users.values
    label_encoder = LabelEncoder()
    encoded_y_train = label_encoder.fit_transform(y_train)

    test_users_ids = test_users['id']
    test_users.drop('id', axis=1, inplace=True)
    test_users = test_users.fillna(-1)
    x_test = test_users.values

    xgb = XGBClassifier(
        max_depth=6,
        learning_rate=0.2,
        n_estimators=23,
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        missing=None,
        silent=True,
        nthread=-1,
        seed=42
    )
    clf = CustomOneVsOneClassifier(xgb, strategy='vote')
    clf.fit(x_train, encoded_y_train)
    y_pred = clf.predict_proba(x_test)

    generate_submission(y_pred, test_users_ids, label_encoder, name='OvO-6-23')


if __name__ == '__main__':
    main()
