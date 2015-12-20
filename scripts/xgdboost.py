import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

# Data loading
train_users = pd.read_csv('datasets/raw/train_users.csv')
test_users = pd.read_csv('datasets/raw/test_users.csv')
labels = train_users['country_destination'].values
train_users = train_users.drop(['country_destination'], axis=1)
id_test = test_users['id']
piv_train = train_users.shape[0]

# Join train and test users
users = pd.concat((train_users, test_users), axis=0, ignore_index=True)

# Drop ID and date_first_booking from the DataFrame
users = users.drop(['id', 'date_first_booking'], axis=1)

# Fill NaN values
users = users.fillna(-1)  # TODO: Try with NaN

###############################################################################
#                                Preprocessing                                #
###############################################################################

# Date Account Creation
dac = np.vstack(users.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
users['dac_year'] = dac[:,0]
users['dac_month'] = dac[:,1]
users['dac_day'] = dac[:,2]
users = users.drop(['date_account_created'], axis=1)

# Timestamp First Active
tfa = np.vstack(users.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
users['tfa_year'] = tfa[:,0]
users['tfa_month'] = tfa[:,1]
users['tfa_day'] = tfa[:,2]
users = users.drop(['timestamp_first_active'], axis=1)

# Age
av = users.age.values
users['age'] = np.where(np.logical_or(av < 14, av > 100), -1, av)

# One-hot-Encoding categorical features
categorical_features = [
    'gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
    'affiliate_provider', 'first_affiliate_tracked', 'signup_app',
    'first_device_type', 'first_browser'
]

for category in categorical_features:
    users_dummy = pd.get_dummies(users[category], prefix=category)
    users = users.drop([category], axis=1)
    users = pd.concat((users, users_dummy), axis=1)

# Splitting train and test
values = users.values
X = values[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)
X_test = values[piv_train:]

# Classifier
xgb = XGBClassifier(
    max_depth=10,
    learning_rate=0.22,
    n_estimators=100,
    objective="multi:softprob",
    nthread=-1,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.6,
    colsample_bytree=0.6,
    colsample_bylevel=1,
    reg_alpha=0.1,
    reg_lambda=0.9,
    scale_pos_weight=1,
    base_score=0.5,
    seed=42
)

xgb.fit(X, y)
y_pred = xgb.predict_proba(X_test)

# Taking the 5 classes with highest probabilities
ids = []
cts = []
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

# Generate Submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('datasets/submissions/xgboost.csv',index=False)
