import sys
import numpy as np
import pandas as pd

sys.path.append('..')
from utils.data_loading import load_users_data, load_sessions_data

print 'START'

print 'Loading data...',
train_users, test_users = load_users_data()
sessions = load_sessions_data()
sessions.replace('-unknown-', np.nan, inplace=True)
users = pd.concat((train_users, test_users), axis=0, ignore_index=True)
users = users.drop('date_first_booking', axis=1)

print '\tDONE'

n = 5
c = 0
total = len(sessions['user_id'].unique())

print 'Processing...\t0%\r',
sys.stdout.flush()
for user in sessions['user_id'].unique():
    user_session = sessions.loc[sessions['user_id'] == user]
    sys.stdout.flush()
    users.loc[users['id'] == user, 'session_length'] = int(len(user_session))

    c = c + 1
    if c % 1350 == 0:
        percentage = float(c) / float(total)
        print 'Processing...\t{0}%\r'.format(percentage * 100),
        sys.stdout.flush()

print 'Processing...\tDONE\r'

users['session_length'].to_csv('session_length.csv')
print 'END'
