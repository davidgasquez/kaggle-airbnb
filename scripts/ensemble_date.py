#!/usr/bin/env python

from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
import pandas as pd

from kairbnb.io import generate_submission

train_users = pd.read_csv('../cache/train_users.csv1')
test_users = pd.read_csv('../cache/test_users.csv1')

x_train_users = train_users
print(x_train_users.shape)

y_train = x_train_users['country_destination']
x_train_users = x_train_users.drop(['country_destination', 'id'], axis=1)
x_train_users = x_train_users.fillna(-1)

x_train = x_train_users.values
label_encoder = LabelEncoder()
encoded_y_train = label_encoder.fit_transform(y_train)

id_test_users = test_users['id']
test_users.drop('id', axis=1, inplace=True)
test_users = test_users.fillna(-1)
x_test = test_users.values

clf = XGBClassifier(
    max_depth=7,
    learning_rate=0.15,
    n_estimators=80,
    objective="rank:pairwise",
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

clf.fit(x_train, encoded_y_train)
y_pred = clf.predict_proba(x_test)

months = train_users['month_account_created'] > 0
years = train_users['year_account_created'] > 2013

x_train_users = train_users.loc[months & years]
print(x_train_users.shape)

y_train = x_train_users['country_destination']
x_train_users = x_train_users.drop(['country_destination', 'id'], axis=1)
x_train_users = x_train_users.fillna(-1)

x_train = x_train_users.values
label_encoder = LabelEncoder()
encoded_y_train = label_encoder.fit_transform(y_train)

clf = XGBClassifier(
    max_depth=7,
    learning_rate=0.15,
    n_estimators=80,
    objective="rank:pairwise",
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

clf.fit(x_train, encoded_y_train)
y_pred_2 = clf.predict_proba(x_test)

months = train_users['month_account_created'] > 2
years = train_users['year_account_created'] > 2013

x_train_users = train_users.loc[months & years]
print(x_train_users.shape)

y_train = x_train_users['country_destination']
x_train_users = x_train_users.drop(['country_destination', 'id'], axis=1)
x_train_users = x_train_users.fillna(-1)

x_train = x_train_users.values
label_encoder = LabelEncoder()
encoded_y_train = label_encoder.fit_transform(y_train)

clf = XGBClassifier(
    max_depth=7,
    learning_rate=0.15,
    n_estimators=80,
    objective="rank:pairwise",
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

clf.fit(x_train, encoded_y_train)
y_pred_3 = clf.predict_proba(x_test)

generate_submission(y_pred, id_test_users, label_encoder, name='ensemble_date')
