import pandas as pd

from utils.preprocessing import one_hot_encoding

# Define data path and suffix
processed_data_path = '../data/processed/'
suffix = '4'

# Load raw data
train_users = pd.read_csv(processed_data_path + 'train_users.csv' + suffix)
test_users = pd.read_csv(processed_data_path + 'test_users.csv' + suffix)

# Join users
users = pd.concat((train_users, test_users), axis=0, ignore_index=True)

# Set ID as index
users = users.set_index('id')
train_users = train_users.set_index('id')
test_users = test_users.set_index('id')

# Drop columns
drop_list = [
    'date_account_created',
    'date_first_active',
    'timestamp_first_active'
]

users.drop(drop_list, axis=1, inplace=True)

# TODO: add singup flow to categorical features

# Encode categorical features
categorical_features = [
    'gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
    'affiliate_provider', 'first_affiliate_tracked', 'signup_app',
    'first_device_type', 'first_browser', 'most_used_device'
]

users = one_hot_encoding(users, categorical_features)

# Split into train and test users
train_users = users.loc[train_users.index]
test_users = users.loc[test_users.index]
test_users.drop('country_destination', inplace=True, axis=1)

# IDEA: Average distance to N neighbors of each class

# Save to csv
prefix = 'processed_'
train_users.to_csv(processed_data_path + prefix + 'train_users.csv' + suffix)
test_users.to_csv(processed_data_path + prefix + 'test_users.csv' + suffix)
