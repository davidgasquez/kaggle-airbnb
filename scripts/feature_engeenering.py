import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder
from utils.preprocessing import distance_to_holidays


def process_user_actions(user):
    """Count the elapsed seconds per action.

    Parameters
    ----------
    user : str
        User ID.
    user_session : Pandas DataFrame
        Session of the user.

    Returns
    -------
    user_session_data : Series
        Returns a pandas Series with the elapsed second per each action.
    """
    # Get the user session
    user_session = sessions.loc[sessions['user_id'] == user]
    user_session_data = pd.Series()

    # Length of the session
    user_session_data['session_lenght'] = len(user_session)
    user_session_data['id'] = user

    # Take the count of each value per column
    for column in ['action', 'action_type', 'action_detail', 'device_type']:
        column_data = user_session[column].value_counts()
        column_data.index = column_data.index + '_count'
        user_session_data = user_session_data.append(column_data)

    # Get the most used device
    user_session_data['most_used_device'] = user_session['device_type'].mode()

    # Grouby ID and add values
    return user_session_data.groupby(user_session_data.index).sum()


def process_user_secs_elapsed(user):
    """Compute some statistical values of the elapsed seconds of a given user.

    Parameters
    ----------
    user : str
        User ID.
    user_secs : array
        Seconds elapsed by each action.

    Returns
    -------
    user_processed_secs : Series
        Returns a pandas Series with the statistical values.
    """
    # Locate user in sessions file
    user_secs = sessions.loc[sessions['user_id'] == user, 'secs_elapsed']
    user_processed_secs = pd.Series()
    user_processed_secs['id'] = user

    user_processed_secs['secs_elapsed_sum'] = user_secs.sum()
    user_processed_secs['secs_elapsed_mean'] = user_secs.mean()
    user_processed_secs['secs_elapsed_min'] = user_secs.min()
    user_processed_secs['secs_elapsed_max'] = user_secs.max()
    user_processed_secs['secs_elapsed_quantile_1'] = user_secs.quantile(0.1)
    user_processed_secs['secs_elapsed_quantile_2'] = user_secs.quantile(0.25)
    user_processed_secs['secs_elapsed_quantile_3'] = user_secs.quantile(0.75)
    user_processed_secs['secs_elapsed_quantile_3'] = user_secs.quantile(0.9)
    user_processed_secs['secs_elapsed_median'] = user_secs.median()
    user_processed_secs['secs_elapsed_std'] = user_secs.std()
    user_processed_secs['secs_elapsed_var'] = user_secs.var()
    user_processed_secs['secs_elapsed_skew'] = user_secs.skew()

    # Number of elapsed seconds greater than 1 day
    user_processed_secs['day_pauses'] = user_secs[user_secs > 86400].count()

    # Clicks with less than one hour of differences
    user_processed_secs['short_sessions'] = user_secs[user_secs < 3600].count()

    # Long breaks
    user_processed_secs['long_pauses'] = user_secs[user_secs > 300000].count()

    return user_processed_secs

# Define data path
raw_data_path = '../data/raw/'
processed_data_path = '../data/processed/'
rows = None

# Load raw data
train_users = pd.read_csv(raw_data_path + 'train_users.csv', nrows=rows)
test_users = pd.read_csv(raw_data_path + 'test_users.csv', nrows=rows)
sessions = pd.read_csv(raw_data_path + 'sessions.csv', nrows=rows)

# Join users
users = pd.concat((train_users, test_users), axis=0, ignore_index=True)
users = users.set_index('id')

# Drop date_first_booking column (empty since competition's restart)
users = users.drop('date_first_booking', axis=1)

# Replace NaNs
users['gender'].replace('-unknown-', np.nan, inplace=True)
users['language'].replace('-unknown-', np.nan, inplace=True)
sessions.replace('-unknown-', np.nan, inplace=True)

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

# Process user actions in parallel
p = Pool(multiprocessing.cpu_count())
result = p.map(process_user_actions, sessions['user_id'].unique())
result = pd.DataFrame(result).set_index('id')
result.index = le.inverse_transform(result.index)
users = pd.concat([users, result], axis=1)

# Process seconds elapsed statistics in parallel
p = Pool(multiprocessing.cpu_count())
result = p.map(process_user_secs_elapsed, sessions['user_id'].unique())
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
suffix = '3'
processed_train_users.to_csv(processed_data_path + 'train_users.csv' + suffix)
processed_test_users.to_csv(processed_data_path + 'test_users.csv' + suffix)
