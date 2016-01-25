import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectFromModel

from utils.metrics import ndcg_scorer

path = '../data/processed/'
train_users = pd.read_csv(path + 'ohe_count_processed_train_users.csv')
y_train = train_users['country_destination']
train_users.drop('country_destination', axis=1, inplace=True)
train_users.drop('id', axis=1, inplace=True)
train_users = train_users.fillna(-1)
x_train = train_users.values
label_encoder = LabelEncoder()
encoded_y_train = label_encoder.fit_transform(y_train)

clf = XGBClassifier(n_estimators=1, nthread=-1, seed=42)
kf = KFold(len(x_train), n_folds=5, random_state=42)

score = cross_val_score(clf, x_train, encoded_y_train,
                        cv=kf, scoring=ndcg_scorer)
print 'Score:', score.mean()


class CustomXGB(XGBClassifier):

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).
        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        booster = self.booster()
        scores = booster.get_fscore()
        all_scores = pd.Series(np.zeros(x_train.shape[1]))
        scores = pd.Series(scores)
        scores.index = scores.index.map(lambda x: x[1:]).astype(int)
        final_scores = all_scores + scores
        importances = final_scores.fillna(0).values
        return importances


custom = CustomXGB(n_estimators=1, seed=42, nthread=-1)
model = SelectFromModel(custom)
X_new = model.fit_transform(x_train, encoded_y_train)

score = cross_val_score(clf, X_new, encoded_y_train,
                        cv=kf, scoring=ndcg_scorer)
print 'Score:', score.mean()
