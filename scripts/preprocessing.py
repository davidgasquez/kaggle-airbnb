import sys
import numpy as np
import pandas as pd
sys.path.append('..')
from utils.preprocessing import one_hot_encoding

path = '../datasets/raw/'
train_users = pd.read_csv(path + 'train_users.csv')
test_users = pd.read_csv(path + 'test_users.csv')
sessions = pd.read_csv(path + 'sessions.csv')

# Join users
users = pd.concat((train_users, test_users), axis=0, ignore_index=True)

# Drop useless column
users = users.drop('date_first_booking', axis=1)

# Replace NaNs
users['gender'].replace('-unknown-', np.nan, inplace=True)
users['language'].replace('-unknown-', np.nan, inplace=True)
sessions.replace('-unknown-', np.nan, inplace=True)

# Remove weird age values
users.loc[users['age'] > 100, 'age'] = np.nan
users.loc[users['age'] < 14, 'age'] = np.nan

# List categorical features
categorical_features = [
    'affiliate_channel',
    'affiliate_provider',
    'country_destination',
    'first_affiliate_tracked',
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method'
]

# Change categorical features
for categorical_feature in categorical_features:
    users[categorical_feature] = users[categorical_feature].astype('category')

# Change type to date
users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_active'] = pd.to_datetime(users['timestamp_first_active'],
                                            format='%Y%m%d%H%M%S')

# Compute date_account_created weekday
weekdays = []
for date in users.date_account_created:
    weekdays.append(date.weekday())
users['weekday_account_created'] = pd.Series(weekdays)

# Compute weekday_first_active weekday
weekdays = []
for date in users.date_account_created:
    weekdays.append(date.weekday())
users['weekday_first_active'] = pd.Series(weekdays)

# Split dates into day,month,year
year_account_created = pd.DatetimeIndex(users['date_account_created']).year
users['year_account_created'] = year_account_created
month_account_created = pd.DatetimeIndex(users['date_account_created']).month
users['month_account_created'] = month_account_created
day_account_created = pd.DatetimeIndex(users['date_account_created']).day
users['day_account_created'] = day_account_created
year_first_active = pd.DatetimeIndex(users['date_first_active']).year
users['year_first_active'] = year_first_active
month_first_active = pd.DatetimeIndex(users['date_first_active']).month
users['month_first_active'] = month_first_active
day_first_active = pd.DatetimeIndex(users['date_first_active']).day
users['day_first_active'] = day_first_active

# The constant N it's used to limit the values we get from the session data.
N = 100

for user in sessions['user_id'].unique():
    # Get the user session
    user_session = sessions.loc[sessions['user_id'] == user]

    # Length of the session
    users.loc[users['id'] == user, 'session_length'] = int(len(user_session))

    # Save the number of times the user repeated his N top action_types
    action_type = user_session['action_type'].value_counts()
    for i in range(min(N, len(action_type.index))):
        new_column = action_type.index[i] + '_count'
        users.loc[users['id'] == user, new_column] = action_type.values[i]

    # Count numer of times the user repeated his top N actions
    action = user_session['action'].value_counts()
    for i in range(min(N, len(action.index))):
        new_column = action.index[i] + '_count'
        users.loc[users['id'] == user, new_column] = action.values[i]

    # The same with action detail
    action_detail = user_session['action_detail'].value_counts()
    for i in range(min(N, len(action_detail.index))):
        new_column = action_detail.index[i] + '_count'
        users.loc[users['id'] == user, new_column] = action_detail.values[i]

    # Get the most used device
    if user_session['device_type'].value_counts().sum() is not 0:
        most_used_device = user_session['device_type'].value_counts().index[0]
        users.loc[users['id'] == user, 'most_used_device'] = most_used_device

users = users.set_index('id')

# Elapsed seconds sum
secs_elapsed_sum = sessions.groupby('user_id')['secs_elapsed'].sum()
secs_elapsed_sum.name = 'secs_elapsed_sum'
users = pd.concat([users, secs_elapsed_sum], axis=1)

# Elapsed seconds mean
secs_elapsed_average = sessions.groupby('user_id')['secs_elapsed'].mean()
secs_elapsed_average.name = 'secs_elapsed_average'
users = pd.concat([users, secs_elapsed_average], axis=1)

# Elapsed seconds min
min_secs_elapsed = sessions.groupby('user_id')['secs_elapsed'].min()
min_secs_elapsed.name = 'min_secs_elapsed'
users = pd.concat([users, min_secs_elapsed], axis=1)

# Elapsed seconds max
max_secs_elapsed = sessions.groupby('user_id')['secs_elapsed'].max()
max_secs_elapsed.name = 'max_secs_elapsed'
users = pd.concat([users, max_secs_elapsed], axis=1)

# Elapsed seconds first_quantile
first_quantile = sessions.groupby('user_id')['secs_elapsed'].quantile(0.25)
first_quantile.name = 'secs_elapsed_first_quantile'
users = pd.concat([users, first_quantile], axis=1)

# Elapsed seconds second_quantile
second_quantile = sessions.groupby('user_id')['secs_elapsed'].quantile(0.5)
second_quantile.name = 'secs_elapsed_second_quantile'
users = pd.concat([users, second_quantile], axis=1)

# Elapsed seconds third_quantile
third_quantile = sessions.groupby('user_id')['secs_elapsed'].quantile(0.75)
third_quantile.name = 'secs_elapsed_third_quantile'
users = pd.concat([users, third_quantile], axis=1)

# Elapsed seconds fourth_quantile
fourth_quantile = sessions.groupby('user_id')['secs_elapsed'].quantile(0.9)
fourth_quantile.name = 'secs_elapsed_fourth_quantile'
users = pd.concat([users, fourth_quantile], axis=1)

# Elapsed seconds median
median = sessions.groupby('user_id')['secs_elapsed'].median()
median.name = 'secs_elapsed_median'
users = pd.concat([users, median], axis=1)

# Elapsed seconds std
std = sessions.groupby('user_id')['secs_elapsed'].std()
std.name = 'secs_elapsed_std'
users = pd.concat([users, std], axis=1)

# Elapsed seconds var
var = sessions.groupby('user_id')['secs_elapsed'].var()
var.name = 'secs_elapsed_var'
users = pd.concat([users, var], axis=1)

# Elapsed seconds skew
skew = sessions.groupby('user_id')['secs_elapsed'].skew()
skew.name = 'secs_elapsed_skew'
users = pd.concat([users, skew], axis=1)

# Number of elapsed seconds greater than 1 day
query = sessions['secs_elapsed'] > 86400
day_pauses = sessions.loc[query].groupby('user_id').count()['secs_elapsed']
day_pauses.name = 'day_pauses'
users = pd.concat([users, day_pauses], axis=1)

# Number of elapsed seconds lesser than 1 hour
query = sessions['secs_elapsed'] < 3600
short_sessions = sessions.loc[query].groupby('user_id').count()['secs_elapsed']
short_sessions.name = 'short_sessions'
users = pd.concat([users, short_sessions], axis=1)

# Users not returning in a big time
query = sessions['secs_elapsed'] > 300000
long_sessions = sessions.loc[query].groupby('user_id').count()['secs_elapsed']
long_sessions.name = 'long_sessions'
users = pd.concat([users, long_sessions], axis=1)

# First value
first_secs_elapsed = sessions.groupby('user_id')['secs_elapsed'].first()
first_secs_elapsed.name = 'first_secs_elapsed'
users = pd.concat([users, first_secs_elapsed], axis=1)

# Last value
last_secs_elapsed = sessions.groupby('user_id')['secs_elapsed'].last()
last_secs_elapsed.name = 'last_secs_elapsed'
users = pd.concat([users, last_secs_elapsed], axis=1)

train_users = train_users.set_index('id')
test_users = test_users.set_index('id')

processed_train_users = users.loc[train_users.index]
processed_test_users = users.loc[test_users.index]
processed_test_users.drop('country_destination', inplace=True, axis=1)

path = '../datasets/processed/'

processed_train_users.to_csv(path + 'semi_processed_train_users.csv')
processed_test_users.to_csv(path + 'semi_processed_test_users.csv')

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

users.index.name = 'id'
processed_train_users = users.loc[train_users.index]
processed_test_users = users.loc[test_users.index]
processed_test_users.drop('country_destination', inplace=True, axis=1)

processed_train_users.to_csv(path + 'processed_train_users.csv')
processed_test_users.to_csv(path + 'processed_test_users.csv')
