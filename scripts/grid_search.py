import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

sys.path.append('..')
from utils.data_loading import load_users_data
from utils.preprocessing import one_hot_encoding

print "Loading data...",
train_users, test_users = load_users_data()
users = pd.read_csv('../datasets/processed/users_with_session.csv')
print "\tDONE"

print "Preprocessing...",

# Get train labels and ids
labels = train_users['country_destination'].values
train_users = train_users.drop(['country_destination'], axis=1)
id_test = test_users['id']
piv_train = train_users.shape[0]

drop_list = [
    'id',
    'country_destination',
    'Unnamed: 0',
    'date_account_created',
    'date_first_active',
    'timestamp_first_active'
]

# Drop columns
users = users.drop(drop_list, axis=1)

# Fill NaNs
users = users.fillna(-1)

# Encode categorical features
categorical_features = [
    'gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
    'affiliate_provider', 'first_affiliate_tracked', 'signup_app',
    'first_device_type', 'first_browser', 'most_used_device'
]
users = one_hot_encoding(users, categorical_features)

# Splitting train and test
values = users.values
values = StandardScaler().fit_transform(values)
X = values[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)
X_test = values[piv_train:]

print "\tDONE"

import xgboost
from sklearn.grid_search import GridSearchCV

xgb_model = xgboost.XGBClassifier(
    objective="multi:softprob",
    nthread=-1,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.6,
    colsample_bytree=0.6,
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
        'max_depth':[1,2,3,4,6],
        'n_estimators': [20,25,30,35,40],
        'learning_rate':[0.2, 0.4, 0.1],
    },
    cv=2,
    verbose=2,
    n_jobs=-1,
    scoring='log_loss'
    )

clf.fit(X,y)

print
print(clf.grid_scores_)
print
print(clf.best_score_)
print(clf.best_params_)
