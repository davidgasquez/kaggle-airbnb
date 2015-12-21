import pandas as pd
import numpy as np

train_users = pd.read_csv('/home/david/projects/airbnb-kaggle/datasets/raw/train_users.csv')
test_users = pd.read_csv('/home/david/projects/airbnb-kaggle/datasets/raw/test_users.csv')
sessions = pd.read_csv('/home/david/projects/airbnb-kaggle/datasets/raw/sessions.csv')

print sessions.columns

sessions.replace('-unknown-', np.nan)
for user in sessions['user_id'].unique():
    user_session = sessions.loc[sessions['user_id'] == user]
    # Session length
    print len(user_session)

    # Action repetitions
    user_session['action_type'].value_counts()

    print user_session['action_type']
