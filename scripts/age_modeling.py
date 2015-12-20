import sys
import numpy as np
import pandas as pd
sys.path.append('..')
from utils.data_loading import load_users_data
from utils.preprocessing import one_hot_encoding
from sklearn.preprocessing import StandardScaler
from sklearn import grid_search

# Load users
print("Loading data...")
train_users, test_users = load_users_data()

# Join users and select important features
train_users = train_users.drop(['country_destination'], axis=1)
id_test = test_users['id']
users = pd.concat((train_users, test_users), axis=0, ignore_index=True)
users = users.drop(['id', 'date_first_booking'], axis=1)

# Set NaNs
users['gender'].replace('-unknown-', np.nan, inplace=True)
users['language'].replace('-unknown-', np.nan, inplace=True)

print("Preprocessing...")
# Preprocess dates
users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['year_account_created'] = pd.DatetimeIndex(users['date_account_created']).year
users['month_account_created'] = pd.DatetimeIndex(users['date_account_created']).month
users['day_account_created'] = pd.DatetimeIndex(users['date_account_created']).day
users = users.drop(['date_account_created'], axis=1)

users['timestamp_first_active'] = pd.to_datetime(users['timestamp_first_active'], format='%Y%m%d%H%M%S')
users['year_first_active'] = pd.DatetimeIndex(users['timestamp_first_active']).year
users['month_first_active'] = pd.DatetimeIndex(users['timestamp_first_active']).month
users['day_first_active'] = pd.DatetimeIndex(users['timestamp_first_active']).day
users = users.drop(['timestamp_first_active'], axis=1)

# Encode categorical features
categorical_features = [
    'gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
    'affiliate_provider', 'first_affiliate_tracked', 'signup_app',
    'first_device_type', 'first_browser'
]
users = one_hot_encoding(users, categorical_features)

# Fix weird values in age
age_values = users.age.values
users['age'] = np.where(np.logical_or(age_values < 14, age_values > 100), np.nan, age_values)

print("Scaling...")

# Get mask with users with NaN age
null_values = users['age'].isnull()

# Scale
X = users.columns.tolist()
X.remove('age')
users[X] = StandardScaler().fit_transform(users[X])

# Backup user DataFrame
processed_users = users

# Set training data
users_with_age = users[~null_values]

print("Evaluating Models:")

################################################################################
print("- RF:")

from sklearn.ensemble import RandomForestRegressor
svr = RandomForestRegressor(n_jobs=-1)

# Grid Search
param_grid = [
    {
        'n_estimators': [50,100,150,200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30, 40]
    },
]

clf = grid_search.GridSearchCV(svr, param_grid, n_jobs=-1,
                               scoring='r2', cv=5, verbose=0)

# Run Grid Search
clf.fit(
    users_with_age.drop('age', axis=1),
    users_with_age['age']
)

print '\t', clf.best_score_, 'with', clf.best_params_

################################################################################
print("- ET:")

from sklearn.ensemble import ExtraTreesRegressor
svr = ExtraTreesRegressor(n_jobs=-1)

# Grid Search
param_grid = [
    {'n_estimators': [50,100,150,200]},
]

clf = grid_search.GridSearchCV(svr, param_grid, n_jobs=-1,
                               scoring='r2', cv=5, verbose=0)

# Run Grid Search
clf.fit(
    users_with_age.drop('age', axis=1),
    users_with_age['age']
)

print '\t', clf.best_score_, 'with', clf.best_params_

################################################################################
print("- SVR:")
from sklearn.svm import SVR
svr = SVR()

# Grid Search
param_grid = {
    'C': [1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.001, 0.005, 0.01]
}

clf = grid_search.GridSearchCV(svr, param_grid, n_jobs=-1,
                               scoring='r2', cv=5, verbose=0)

# Run Grid Search
clf.fit(
    users_with_age.drop('age', axis=1),
    users_with_age['age']
)

print '\t', clf.best_score_, 'with', clf.best_params_

################################################################################
print("- KNN:")
from sklearn.neighbors import KNeighborsClassifier
svr = KNeighborsClassifier()

# Grid Search
param_grid = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute']
}

clf = grid_search.GridSearchCV(svr, param_grid, n_jobs=-1,
                               scoring='r2', cv=5, verbose=0)

# Run Grid Search
clf.fit(
    users_with_age.drop('age', axis=1),
    users_with_age['age']
)

print '\t', clf.best_score_, 'with', clf.best_params_
