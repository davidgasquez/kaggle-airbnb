from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd

train_users = pd.read_csv('../cache/train_users.csv1')
test_users = pd.read_csv('../cache/test_users.csv1')

train = train_users.drop(['country_destination', 'id'], axis=1)
test = test_users.drop('id', axis=1)

users = pd.concat((train, test), axis=0, ignore_index=True)

x_train = users.loc[users['age'].notnull()]

y_train = x_train['age']
x_train = x_train.drop('age', axis=1)
x_train = x_train.fillna(-1)

rf = ExtraTreesRegressor(n_estimators=300, n_jobs=-1)
rf.fit(x_train, y_train)

train = train_users.drop(['country_destination', 'id'], axis=1)
x_train = train.loc[train['age'].isnull()]
x_train = x_train.drop('age', axis=1)
x_train = x_train.fillna(-1)
train_users.loc[train_users['age'].isnull(), 'age'] = rf.predict(x_train).astype(int)

test = test_users.drop('id', axis=1)
x_train = test.loc[test['age'].isnull()]
x_train = x_train.drop('age', axis=1)
x_train = x_train.fillna(-1)
test_users.loc[test_users['age'].isnull(), 'age'] = rf.predict(x_train).astype(int)

train_users.to_csv('train_users_with_age_et.csv')
test_users.to_csv('test_users_with_age_et.csv')
