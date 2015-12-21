
# coding: utf-8

# In[ ]:

import sys
import numpy as np
import pandas as pd

sys.path.append('..')


# In[ ]:

from utils.data_loading import load_users_data, load_sessions_data

train_users, test_users = load_users_data()
sessions = load_sessions_data()
sessions.replace('-unknown-', np.nan, inplace=True)


# In[ ]:

users = pd.concat((train_users, test_users), axis=0, ignore_index=True)
users = users.drop('date_first_booking', axis=1)


# In[ ]:

users['gender'].replace('-unknown-', np.nan, inplace=True)
users['language'].replace('-unknown-', np.nan, inplace=True)


# In[ ]:

users.loc[users['age'] > 100, 'age'] = np.nan
users.loc[users['age'] < 14, 'age'] = np.nan


# In[ ]:

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

for categorical_feature in categorical_features:
    users[categorical_feature] = users[categorical_feature].astype('category')


# In[ ]:

users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_active'] = pd.to_datetime(users['timestamp_first_active'], format='%Y%m%d%H%M%S')


# In[ ]:

weekdays = []
for date in users.date_account_created:
    weekdays.append(date.weekday())
users['weekday_account_created'] = pd.Series(weekdays)

weekdays = []
for date in users.date_account_created:
    weekdays.append(date.weekday())
users['weekday_first_active'] = pd.Series(weekdays)

users['year_account_created'] = pd.DatetimeIndex(users['date_account_created']).year
users['month_account_created'] = pd.DatetimeIndex(users['date_account_created']).month
users['day_account_created'] = pd.DatetimeIndex(users['date_account_created']).day

users['year_first_active'] = pd.DatetimeIndex(users['timestamp_first_active']).year
users['month_first_active'] = pd.DatetimeIndex(users['timestamp_first_active']).month
users['day_first_active'] = pd.DatetimeIndex(users['timestamp_first_active']).day


# In[ ]:

sessions_lengths = sessions['user_id'].value_counts()


# In[ ]:

# for user, session_length in sessions_lengths.iteritems():
#     users.loc[users['id'] == user, 'session_length'] = int(session_length)


# In[ ]:

n = 5
c = 0
for user in sessions['user_id'].unique():
    user_session = sessions.loc[sessions['user_id'] == user]

    users.loc[users['id'] == user, 'session_length'] = int(len(user_session))

    # Count numer of time repeating each actions
    action_type = user_session['action_type'].value_counts()
    for i in range(min(n, len(action_type.index))):
        users.loc[users['id'] == user, action_type.index[i] + '_count'] = action_type.values[i]
    
    # Count numer of time repeating each actions
    action = user_session['action'].value_counts()
    for i in range(min(n, len(action.index))):
        users.loc[users['id'] == user, action.index[i] + '_count'] = action.values[i]

    # Count numer of time repeating each actions
    action_detail = user_session['action_detail'].value_counts()
    for i in range(min(n, len(action_detail.index))):
        users.loc[users['id'] == user, action_detail.index[i] + '_count'] = action_detail.values[i]

    if user_session['device_type'].value_counts().sum() is not 0:
        users.loc[users['id'] == user, 'most_used_device'] = user_session['device_type'].value_counts().index[0]

    if c % 1000 == 0:                                                                                            
        print c
    c = c + 1                                                                                                       


# In[ ]:

users.to_csv('preprocessed.csv')

