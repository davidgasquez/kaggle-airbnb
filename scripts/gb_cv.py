import argparse

from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

from kairbnb.metrics import ndcg_scorer
from kairbnb.io import load_users

VERSION = '3'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--max_depth', default=7, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.18, type=float)
    parser.add_argument('-n', '--n_estimators', default=80, type=int)
    parser.add_argument('-ct', '--colsample_bytree', default=1, type=float)
    parser.add_argument('-cl', '--colsample_bylevel', default=1, type=float)
    parser.add_argument('-sub', '--subsample', default=1, type=float)
    parser.add_argument('-md', '--max_delta', default=0, type=float)
    args = parser.parse_args()

    train_users, _ = load_users(version=VERSION)
    train_users.fillna(-1, inplace=True)
    y_train = train_users['country_destination']
    train_users.drop(['country_destination', 'id'], axis=1, inplace=True)
    x_train = train_users.values

    label_encoder = LabelEncoder()
    encoded_y_train = label_encoder.fit_transform(y_train)

    xgb = XGBClassifier(
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        objective="multi:softprob",
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        colsample_bylevel=args.colsample_bylevel,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        missing=None,
        silent=True,
        nthread=-1,
        seed=42
    )

    kf = KFold(len(x_train), n_folds=10, random_state=42)

    score = cross_val_score(xgb, x_train, encoded_y_train,
                            cv=kf, scoring=ndcg_scorer, verbose=10)

    print(xgb.get_params(), score.mean())
