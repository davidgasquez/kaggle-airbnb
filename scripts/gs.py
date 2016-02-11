#!/usr/bin/env python

from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier
from kairbnb.metrics import ndcg_scorer
import pandas as pd

from kairbnb.io import generate_submission

x_train_users = pd.read_csv('t.csv')

print(x_train_users.shape)

y_train = x_train_users['country_destination']
x_train_users = x_train_users.drop(['country_destination', 'id'], axis=1)
x_train_users = x_train_users.fillna(-1)

x_train = x_train_users.values
label_encoder = LabelEncoder()
encoded_y_train = label_encoder.fit_transform(y_train)

xgb_model = XGBClassifier(
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
        'max_depth': [4, 6],
        'n_estimators': [40, 50],
        'learning_rate': [0.1, 0.2],
    },
    cv=10,
    verbose=10,
    n_jobs=1,
    scoring=ndcg_scorer
)

clf.fit(x_train, encoded_y_train)

print(clf.best_params_)
print(clf.best_score_)
