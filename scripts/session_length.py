
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

n = 5
c = 0
for user in sessions['user_id'].unique():
    user_session = sessions.loc[sessions['user_id'] == user]

    users.loc[users['id'] == user, 'session_length'] = int(len(user_session))


    if c % 1000 == 0:
        print c
    c = c + 1

users['session_length'].to_csv('session_length.csv')

