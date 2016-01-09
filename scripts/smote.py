#!/usr/bin/env python

import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.append('..')
from utils.unbalanced_dataset import NearMiss, SMOTEENN


def main():
    path = '../datasets/processed/'
    train_users = pd.read_csv(path + 'processed_train_users.csv')

    y_train = train_users['country_destination']

    train_users.drop('country_destination', axis=1, inplace=True)
    train_users.drop('id', axis=1, inplace=True)
    train_users = train_users.fillna(0)

    x_train = train_users.values

    label_encoder = LabelEncoder()
    encoded_y_train = label_encoder.fit_transform(y_train)

    verbose = True
    NM1 = NearMiss(version=1, verbose=verbose)
    nm1x, nm1y = NM1.fit_transform(x_train, encoded_y_train)

    print "Undersample Completed"

    SENN = SMOTEENN(ratio=ratio, verbose=verbose)
    ennx, enny = SENN.fit_transform(x_train, encoded_y_train)

    print ennx


if __name__ == '__main__':
    main()
