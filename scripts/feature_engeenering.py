import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from sklearn.preprocessing import LabelEncoder
from utils.preprocessing import distance_to_holidays
from utils.preprocessing import process_user_actions
from utils.preprocessing import process_user_secs_elapsed
from utils.io import load_users

# TODO: NROWS + VERSION should be parameters to the script

# Load raw data
train_users, test_users = load_users()
sessions = pd.read_csv('../data/raw/sessions.csv')

# Join users
users = pd.concat((train_users, test_users), axis=0, ignore_index=True)
users = users.set_index('id')

# Drop date_first_booking column (empty since competition's restart)
users = users.drop('date_first_booking', axis=1)

# Remove weird age values
users.loc[users['age'] > 100, 'age'] = np.nan
users.loc[users['age'] < 13, 'age'] = np.nan

# Change type to date
users['date_account_created'] = pd.to_datetime(users['date_account_created'],
                                               errors='ignore')
users['date_first_active'] = pd.to_datetime(users['timestamp_first_active'],
                                            format='%Y%m%d%H%M%S')

# Convert to DatetimeIndex
date_account_created = pd.DatetimeIndex(users['date_account_created'])
date_first_active = pd.DatetimeIndex(users['date_first_active'])

# Get weekday based on the date
users['weekday_account_created'] = date_account_created.weekday
users['weekday_first_active'] = date_first_active.weekday

# Split dates into day, week, month, year
users['year_account_created'] = date_account_created.year
users['month_account_created'] = date_account_created.month
users['day_account_created'] = date_account_created.day
users['week_account_created'] = date_account_created.week
users['year_first_active'] = date_first_active.year
users['month_first_active'] = date_first_active.month
users['day_first_active'] = date_first_active.day
users['week_first_active'] = date_first_active.week

# Get the count of general session information
result = sessions.groupby('user_id').count()
result.rename(columns=lambda x: x + '_count', inplace=True)
users = pd.concat([users, result], axis=1)

# Add number of NaNs per row
users['nan_sum'] = users.isnull().sum(axis=1)

le = LabelEncoder()
sessions['user_id'] = le.fit_transform(sessions['user_id'].astype(str))

sessions_ids = sessions['user_id'].unique()

# Make pool to process in parallel
p = multiprocessing.Pool(multiprocessing.cpu_count())

# Count of each user action in sessions
result = p.map(partial(process_user_actions, sessions), sessions_ids)
result = pd.DataFrame(result).set_index('id')
result.index = le.inverse_transform(result.index)
users = pd.concat([users, result], axis=1)

# Elapsed seconds statistics
result = p.map(partial(process_user_secs_elapsed, sessions), sessions_ids)
result = pd.DataFrame(result).set_index('id')
result.index = le.inverse_transform(result.index.values.astype(int))
users = pd.concat([users, result], axis=1)

# IDEA: Add interaction features

# IDEA: Classify and group by dispositive

# Add distance to holidays
users['date_account_created'].apply(distance_to_holidays)

# Set ID as index
train_users = train_users.set_index('id')
test_users = test_users.set_index('id')

# Split into train and test users
users.index.name = 'id'
processed_train_users = users.loc[train_users.index]
processed_test_users = users.loc[test_users.index]
processed_test_users.drop(['country_destination'], inplace=True, axis=1)

# Save to csv
version = '4'
processed_train_users.to_csv('../data/processed/train_users.csv' + version)
processed_test_users.to_csv('../data/processed/test_users.csv' + version)
