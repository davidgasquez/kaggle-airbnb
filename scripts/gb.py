import sys
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier

sys.path.append('..')
from utils.data_loading import load_users_data
from utils.preprocessing import one_hot_encoding

print "Loading data...",
train_users, test_users = load_users_data()
users = pd.read_csv('../datasets/processed/users_with_session.csv')
print "\tDONE"

print "Preprocessing...",

# Get train labels and ids
labels = train_users['country_destination'].values
train_users = train_users.drop(['country_destination'], axis=1)
id_test = test_users['id']
piv_train = train_users.shape[0]

drop_list = [
    'id',
    'country_destination',
    'Unnamed: 0',
    'date_account_created',
    'date_first_active',
    'timestamp_first_active'
]

# Drop columns
users = users.drop(drop_list, axis=1)

# Fill NaNs
users = users.fillna(-1)

# Encode categorical features
categorical_features = [
    'gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
    'affiliate_provider', 'first_affiliate_tracked', 'signup_app',
    'first_device_type', 'first_browser', 'most_used_device'
]
users = one_hot_encoding(users, categorical_features)

# Splitting train and test
values = users.values
values = StandardScaler().fit_transform(values)
X = values[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)
X_test = values[piv_train:]

print "\tDONE"

print "Fitting...",
# Classifier
xgb = XGBClassifier(
    max_depth=6,
    learning_rate=0.2,
    n_estimators=45,
    objective="multi:softprob",
    nthread=-1,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.6,
    colsample_bytree=0.6,
    colsample_bylevel=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    seed=42
)

xgb.fit(X, y)
print "\tDONE"

print "Predicting...",
y_pred = xgb.predict_proba(X_test)
print "\tDONE"

print "Generating submission...",
# Taking the 5 classes with highest probabilities
ids = []
cts = []
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

# Generate Submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
date = datetime.datetime.now().strftime("%m-%d_%H:%M")
sub.to_csv('../datasets/submissions/xgboost' + str(date) + '.csv',index=False)
print "\tDONE"
print "END"
