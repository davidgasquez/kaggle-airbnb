import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

# Set magic seed
np.random.seed(42)

# Load Data
df_train = pd.read_csv('data/raw/train_users.csv')
df_test = pd.read_csv('data/raw/test_users.csv')
labels = df_train['country_destination'].values

# Remove target feature from train DataFrame
df_train = df_train.drop(['country_destination'], axis=1)

# Get the tests user ids
id_test = df_test['id']

# Get number of train instances
piv_train = df_train.shape[0]

# Creating a DataFrame with train and test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

# Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)

# Filling nan values
df_all.loc[df_all.age > 500, 'age'] = 2015 - df_all.age
df_all = df_all.fillna(-1)

# Parse account date creation
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)

# Parse first active date
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

# Remove weird age values
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

# One-hot-encoding features
ohe_feats = [
    'gender', 'signup_method', 'signup_flow', 'language',
    'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
    'signup_app', 'first_device_type', 'first_browser'
]

for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

# Splitting train and test
values = df_all.values
X = values[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)
X_test = values[piv_train:]

# Classifier
xgb = XGBClassifier(max_depth=6, learning_rate=0.25, n_estimators=43,
                    objective='multi:softprob', subsample=0.6,
                    colsample_bytree=0.6, seed=0)
xgb.fit(X, y)
y_pred = xgb.predict_proba(X_test)

# Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

# Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('data/submissions/sub.csv', index=False)
