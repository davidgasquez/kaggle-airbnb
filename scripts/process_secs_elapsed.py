import sys
import numpy as np
import pandas as pd

users = pd.read_csv('../datasets/processed/preprocessed.csv')

users.drop('Unnamed: 0', axis=1, inplace=True)

to_remove = users.isnull().sum().loc[users.isnull().sum() > 275542].index

users.drop(to_remove, axis=1, inplace=True)

sys.path.append('..')
from utils.data_loading import load_sessions_data
sessions = load_sessions_data()

elapsed_secs_sum = sessions.groupby('user_id')['secs_elapsed'].sum()
elapsed_secs_sum.name = 'elapsed_secs_sum'
elapsed_secs_average = sessions.groupby('user_id')['secs_elapsed'].mean()
elapsed_secs_average.name = 'elapsed_secs_average'

users = pd.concat([users, elapsed_secs_sum], axis=1)
users = pd.concat([users, elapsed_secs_average], axis=1)

min_secs_elapsed = sessions.groupby('user_id')['secs_elapsed'].min()
min_secs_elapsed.name = 'min_secs_elapsed'
max_secs_elapsed = sessions.groupby('user_id')['secs_elapsed'].max()
max_secs_elapsed.name = 'max_secs_elapsed'
users = pd.concat([users, min_secs_elapsed], axis=1)
users = pd.concat([users, max_secs_elapsed], axis=1)

first_quantile = sessions.groupby('user_id')['secs_elapsed'].quantile(0.25)
first_quantile.name = 'first_quantile'
second_quantile = sessions.groupby('user_id')['secs_elapsed'].quantile(0.5)
second_quantile.name = 'second_quantile'
third_quantile = sessions.groupby('user_id')['secs_elapsed'].quantile(0.75)
third_quantile.name = 'third_quantile'
fourth_quantile = sessions.groupby('user_id')['secs_elapsed'].quantile(0.9)
fourth_quantile.name = 'fourth_quantile'
users = pd.concat([users, first_quantile], axis=1)
users = pd.concat([users, second_quantile], axis=1)
users = pd.concat([users, third_quantile], axis=1)
users = pd.concat([users, fourth_quantile], axis=1)

median = sessions.groupby('user_id')['secs_elapsed'].median()
median.name = 'elapsed_secs_median'
users = pd.concat([users, median], axis=1)
std = sessions.groupby('user_id')['secs_elapsed'].std()
std.name = 'elapsed_secs_std'
users = pd.concat([users, std], axis=1)
var = sessions.groupby('user_id')['secs_elapsed'].var()
var.name = 'elapsed_secs_var'
users = pd.concat([users, var], axis=1)
skew = sessions.groupby('user_id')['secs_elapsed'].skew()
skew.name = 'elapsed_secs_skew'
users = pd.concat([users, skew], axis=1)

day_pauses = sessions.loc[sessions['secs_elapsed'] > 86400].groupby('user_id').count()['secs_elapsed']
day_pauses.name = 'day_pauses'
users = pd.concat([users, day_pauses], axis=1)
short_sessions = sessions.loc[sessions['secs_elapsed'] < 3600].groupby('user_id').count()['secs_elapsed']
short_sessions.name = 'short_sessions'
users = pd.concat([users, short_sessions], axis=1)
long_sessions = sessions.loc[sessions['secs_elapsed'] > 300000].groupby('user_id').count()['secs_elapsed']
long_sessions.name = 'long_sessions'
users = pd.concat([users, long_sessions], axis=1)

first_secs_elapsed = sessions.groupby('user_id')['secs_elapsed'].first()
first_secs_elapsed.name = 'first_secs_elapsed'
users = pd.concat([users, first_secs_elapsed], axis=1)
last_secs_elapsed = sessions.groupby('user_id')['secs_elapsed'].last()
last_secs_elapsed.name = 'last_secs_elapsed'
users = pd.concat([users, last_secs_elapsed], axis=1)

users.to_csv('users_with_session.cvs')
