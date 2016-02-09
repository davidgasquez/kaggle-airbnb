from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

train_users = pd.read_csv('../cache/train_users.csv1', nrows=400)
test_users = pd.read_csv('../cache/test_users.csv1', nrows=400)

train_users.drop(['country_destination', 'id'], axis=1, inplace=True)
test_users.drop('id', axis=1, inplace=True)

users = pd.concat((train_users, test_users), axis=0, ignore_index=True)
processed_train_users = users.loc[users['age'].notnull()]

y_train = processed_train_users['age']
processed_train_users = processed_train_users.drop('age', axis=1)
processed_train_users = processed_train_users.fillna(-1)
x_train = processed_train_users.values

rf = RandomForestRegressor()

rf.fit(x_train, y_train)

predictions = rf.predict(users.loc[users['age'].isnull()].drop('age', axis=1).fillna(-1).values).astype(int)

users.loc[users['age'].isnull(), 'age'] = predictions
