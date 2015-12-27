import sys
import pandas as pd
sys.path.append('..')
from utils.preprocessing import one_hot_encoding

path = '../datasets/processed/'
train_users = pd.read_csv(path + 'semi_processed_train_users.csv')
test_users = pd.read_csv(path + 'semi_processed_test_users.csv')

# Join users
users = pd.concat((train_users, test_users), axis=0, ignore_index=True)
users = users.set_index('id')

drop_list = [
    'date_account_created',
    'date_first_active',
    'timestamp_first_active'
]

# Drop columns
users = users.drop(drop_list, axis=1)

# Encode categorical features
categorical_features = [
    'gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
    'affiliate_provider', 'first_affiliate_tracked', 'signup_app',
    'first_device_type', 'first_browser', 'most_used_device'
]

users = one_hot_encoding(users, categorical_features)
train_users = train_users.set_index('id')
test_users = test_users.set_index('id')

users.index.name = 'id'

processed_train_users = users.loc[train_users.index]
processed_test_users = users.loc[test_users.index]
processed_test_users.drop('country_destination', inplace=True, axis=1)

processed_train_users.to_csv('train_users.csv')
processed_test_users.to_csv('test_users.csv')
