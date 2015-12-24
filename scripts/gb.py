import sys
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
sys.path.append('..')

print("START")
from utils.data_loading import load_users_data
train_users, test_users = load_users_data()

labels = train_users['country_destination'].values
train_users = train_users.drop(['country_destination'], axis=1)
users = pd.read_csv('../datasets/processed/preprocessed.csv')

print("Data Loaded")
id_test = test_users['id']
piv_train = train_users.shape[0]

users = users.drop(['id', 'country_destination'], axis=1)
users = users.drop(['date_account_created', 'date_first_active'], axis=1)
users = users.drop(['timestamp_first_active'], axis=1)

users = users.fillna(-1)

from utils.preprocessing import one_hot_encoding

categorical_features = [
    'gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
    'affiliate_provider', 'first_affiliate_tracked', 'signup_app',
    'first_device_type', 'first_browser', 'most_used_device'
]

users = one_hot_encoding(users, categorical_features)

print("Preprocessed")


# Splitting train and test
values = users.values
values = StandardScaler().fit_transform(values)
X = values[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)
X_test = values[piv_train:]

from xgboost.sklearn import XGBClassifier
# Classifier
xgb = XGBClassifier(
    max_depth=8,
    learning_rate=0.2,
    n_estimators=50,
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
print("Fitted")

y_pred = xgb.predict_proba(X_test)
print("Predicted")
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
print("END")
