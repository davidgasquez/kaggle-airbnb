import sys
import numpy as np
import pandas as pd

sys.path.append('..')
from utils.data_loading import load_users_data, load_sessions_data

print 'START'

print 'Loading data...',
train_users, test_users = load_users_data()

# Number of train user for latter splitting
piv_train = train_users.shape[0]

# Sessions data
sessions = load_sessions_data()
print '\tDONE'

print 'Basic preprocessing...',

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
users['year_account_created'] = pd.DatetimeIndex(users['date_account_created']).year
users['month_account_created'] = pd.DatetimeIndex(users['date_account_created']).month
users['day_account_created'] = pd.DatetimeIndex(users['date_account_created']).day
users['year_first_active'] = pd.DatetimeIndex(users['timestamp_first_active']).year
users['month_first_active'] = pd.DatetimeIndex(users['timestamp_first_active']).month
users['day_first_active'] = pd.DatetimeIndex(users['timestamp_first_active']).day

print '\tDONE'

# The constant N it's used to limit the values we get from the session data.
N = 8

# Counter to compute the progress
processed_users = 0

# Number of users with session
unique_session_users = len(sessions['user_id'].unique())

print 'Processing Sessions...\t0%\r',

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

    # Print the processing progress
    processed_users = processed_users + 1

    if processed_users % 1000 == 0:
        percentage = float(processed_users) / float(unique_session_users)
        print 'Processing Sessions...\t{0}%\r'.format(percentage * 100),

print 'Processing Sessions...\tDONE\r'

print 'Processing secs_elapsed...',

# Remove columns with a lot of NaNs
to_remove = users.isnull().sum().loc[users.isnull().sum() > 275542].index
users.drop(to_remove, axis=1, inplace=True)

# Elapsed seconds sum
elapsed_secs_sum = sessions.groupby('user_id')['secs_elapsed'].sum()
elapsed_secs_sum.name = 'elapsed_secs_sum'
users = pd.concat([users, elapsed_secs_sum], axis=1)

# Elapsed seconds mean
elapsed_secs_average = sessions.groupby('user_id')['secs_elapsed'].mean()
elapsed_secs_average.name = 'elapsed_secs_average'
users = pd.concat([users, elapsed_secs_average], axis=1)

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
first_quantile.name = 'first_quantile'
users = pd.concat([users, first_quantile], axis=1)

# Elapsed seconds second_quantile
second_quantile = sessions.groupby('user_id')['secs_elapsed'].quantile(0.5)
second_quantile.name = 'second_quantile'
users = pd.concat([users, second_quantile], axis=1)

# Elapsed seconds third_quantile
third_quantile = sessions.groupby('user_id')['secs_elapsed'].quantile(0.75)
third_quantile.name = 'third_quantile'
users = pd.concat([users, third_quantile], axis=1)

# Elapsed seconds fourth_quantile
fourth_quantile = sessions.groupby('user_id')['secs_elapsed'].quantile(0.9)
fourth_quantile.name = 'fourth_quantile'
users = pd.concat([users, fourth_quantile], axis=1)

# Elapsed seconds median
median = sessions.groupby('user_id')['secs_elapsed'].median()
median.name = 'elapsed_secs_median'
users = pd.concat([users, median], axis=1)

# Elapsed seconds std
std = sessions.groupby('user_id')['secs_elapsed'].std()
std.name = 'elapsed_secs_std'
users = pd.concat([users, std], axis=1)

# Elapsed seconds var
var = sessions.groupby('user_id')['secs_elapsed'].var()
var.name = 'elapsed_secs_var'
users = pd.concat([users, var], axis=1)

# Elapsed seconds skew
skew = sessions.groupby('user_id')['secs_elapsed'].skew()
skew.name = 'elapsed_secs_skew'
users = pd.concat([users, skew], axis=1)

# Number of elapsed seconds greater than 1 day
day_pauses = sessions.loc[sessions['secs_elapsed'] > 86400].groupby('user_id').count()['secs_elapsed']
day_pauses.name = 'day_pauses'
users = pd.concat([users, day_pauses], axis=1)

# Number of elapsed seconds lesser than 1 hour
short_sessions = sessions.loc[sessions['secs_elapsed'] < 3600].groupby('user_id').count()['secs_elapsed']
short_sessions.name = 'short_sessions'
users = pd.concat([users, short_sessions], axis=1)

# Users not returning in a big time
long_sessions = sessions.loc[sessions['secs_elapsed'] > 300000].groupby('user_id').count()['secs_elapsed']
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

users.to_csv('users.cvs')
print 'END'
