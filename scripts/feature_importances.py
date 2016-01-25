import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

path = '../data/processed/'
train_users = pd.read_csv(path + 'processed_train_users.csv')
y_train = train_users['country_destination']
train_users.drop('country_destination', axis=1, inplace=True)
train_users.drop('id', axis=1, inplace=True)
x_train = train_users.fillna(-1)
label_encoder = LabelEncoder()
encoded_y_train = label_encoder.fit_transform(y_train)

clf = XGBClassifier(
    max_depth=7,
    learning_rate=0.18,
    n_estimators=80,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=1,
    colsample_bytree=1,
    colsample_bylevel=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    missing=None,
    silent=True,
    nthread=-1,
    seed=42
)

clf.fit(x_train, encoded_y_train)

processed = clf.booster().get_fscore().keys()
print 'Scores processed:', clf.booster().get_fscore()

path = '../data/processed/'
train_users = pd.read_csv(path + 'ohe_count_processed_train_users.csv')
y_train = train_users['country_destination']
train_users.drop('country_destination', axis=1, inplace=True)
train_users.drop('id', axis=1, inplace=True)
x_train = train_users.fillna(-1)
label_encoder = LabelEncoder()
encoded_y_train = label_encoder.fit_transform(y_train)

clf.fit(x_train, encoded_y_train)

print
count = clf.booster().get_fscore().keys()
print 'Scores count:', clf.booster().get_fscore()
print
print 'diffs', set(processed) - set(count)
print
print 'diffs2', set(count) - set(processed)
