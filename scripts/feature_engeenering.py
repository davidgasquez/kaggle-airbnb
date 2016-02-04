import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from sklearn.preprocessing import LabelEncoder

from kairbnb.preprocessing import distance_to_holidays
from kairbnb.preprocessing import process_user_actions
from kairbnb.preprocessing import process_user_secs_elapsed
from kairbnb.io import load_users

NROWS = 1000
VERSION = '1'

if __name__ == '__main__':
    # Load raw data
    train_users, test_users = load_users(nrows=NROWS)
    sessions = pd.read_csv('../data/sessions.csv', nrows=NROWS)

    # Join users
    users = pd.concat((train_users, test_users), axis=0, ignore_index=True)
    users = users.set_index('id')

    # Drop date_first_booking column (empty since competition's restart)
    users = users.drop('date_first_booking', axis=1)

    # Remove weird age values
    users.loc[users['age'] > 100, 'age'] = np.nan
    users.loc[users['age'] < 13, 'age'] = np.nan

    # Change type to date
    users['date_account_created'] = pd.to_datetime(
        users['date_account_created'], errors='ignore')
    users['date_first_active'] = pd.to_datetime(
        users['timestamp_first_active'], format='%Y%m%d%H%M%S')

    # Convert to DatetimeIndex
    date_account_created = pd.DatetimeIndex(users['date_account_created'])
    date_first_active = pd.DatetimeIndex(users['date_first_active'])

    # Split dates into day, week, month, year
    users['day_account_created'] = date_account_created.day
    users['weekday_account_created'] = date_account_created.weekday
    users['week_account_created'] = date_account_created.week
    users['month_account_created'] = date_account_created.month
    users['year_account_created'] = date_account_created.year
    users['day_first_active'] = date_first_active.day
    users['weekday_first_active'] = date_first_active.weekday
    users['week_first_active'] = date_first_active.week
    users['month_first_active'] = date_first_active.month
    users['year_first_active'] = date_first_active.year

    # IDEA: Add distance to holidays
    # users['date_account_created'].apply(distance_to_holidays)

    # IDEA: Classify and group by dispositive

    # Get the count of general session information
    result = sessions.groupby('user_id').count()
    result.rename(columns=lambda x: x + '_count', inplace=True)
    users = pd.concat([users, result], axis=1)

    # Add number of NaNs per row
    users['nan_sum'] = users.isnull().sum(axis=1)

    # To improve performance we translate each different user_id string into a
    # integer. This yields almost 50% of performance gain when multiprocessing
    # because Python pickes strings and integers differently
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

    # Set ID as index
    train_users = train_users.set_index('id')
    test_users = test_users.set_index('id')

    # Split into train and test users
    users.index.name = 'id'
    processed_train_users = users.loc[train_users.index]
    processed_test_users = users.loc[test_users.index]
    processed_test_users.drop(['country_destination'], inplace=True, axis=1)

    # Save to csv
    processed_train_users.to_csv('../cache/train_users.csv' + VERSION)
    processed_test_users.to_csv('../cache/test_users.csv' + VERSION)
