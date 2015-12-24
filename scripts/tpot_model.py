import sys
import numpy as np
import pandas as pd
from tpot import TPOT

sys.path.append('..')
from utils.preprocessing import one_hot_encoding
from utils.data_loading import load_users_data
from sklearn.preprocessing import LabelEncoder

print 'START'
print 'Loading data...'

train_users, test_users = load_users_data()
labels = train_users['country_destination'].values
train_users = train_users.drop(['country_destination'], axis=1)

print '\tDONE'
print 'Preprocessing...',

id_test = test_users['id']
piv_train = train_users.shape[0]
users = pd.concat((train_users, test_users), axis=0, ignore_index=True)
users = users.drop(['id', 'date_first_booking'], axis=1)

# Fill NaN values
users = users.fillna(-1)
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
age_values = users.age.values
users['age'] = np.where(np.logical_or(age_values < 14, age_values > 100), -1, age_values)

categorical_features = [
    'gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
    'affiliate_provider', 'first_affiliate_tracked', 'signup_app',
    'first_device_type', 'first_browser'
]

users = one_hot_encoding(users, categorical_features)

# Splitting train and test
values = users.values
X = values[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)
X_test = values[piv_train:]
print '\tDONE'

print 'TPOT...',
tpot = TPOT(generations=2, verbosity=2)

tpot.fit(X, y)
print '\tDONE'

print 'Scoring...',
print tpot.score(X, y, X, y)
print 'END'
