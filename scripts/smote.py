#!/usr/bin/env python

import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.append('..')
from utils.unbalanced_dataset import (
    UnderSampler, NearMiss, CondensedNearestNeighbour, OneSidedSelection,
    NeighbourhoodCleaningRule, TomekLinks, ClusterCentroids, OverSampler,
    SMOTE, SMOTETomek, SMOTEENN, EasyEnsemble, BalanceCascade
)


def main():
    path = '../datasets/processed/'
    train_users = pd.read_csv(path + 'processed_train_users.csv')
    train_users = train_users.head(10000)
    y_train = train_users['country_destination']

    train_users.drop('country_destination', axis=1, inplace=True)
    train_users.drop('id', axis=1, inplace=True)
    train_users = train_users.fillna(0)

    x = train_users.values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_train)

    verbose = True
    # 'Random under-sampling'
    US = UnderSampler(verbose=verbose)
    usx, usy = US.fit_transform(x, y)
    # 'Tomek links'
    TL = TomekLinks(verbose=verbose)
    tlx, tly = TL.fit_transform(x, y)
    # 'Clustering centroids'
    CC = ClusterCentroids(verbose=verbose)
    ccx, ccy = CC.fit_transform(x, y)
    # 'NearMiss-1'
    NM1 = NearMiss(version=1, verbose=verbose)
    nm1x, nm1y = NM1.fit_transform(x, y)
    # 'NearMiss-2'
    NM2 = NearMiss(version=2, verbose=verbose)
    nm2x, nm2y = NM2.fit_transform(x, y)
    # 'NearMiss-3'
    NM3 = NearMiss(version=3, verbose=verbose)
    nm3x, nm3y = NM3.fit_transform(x, y)
    # 'Condensed Nearest Neighbour'
    CNN = CondensedNearestNeighbour(size_ngh=2, n_seeds_S=2, verbose=verbose)
    cnnx, cnny = CNN.fit_transform(x, y)
    # 'One-Sided Selection'
    OSS = OneSidedSelection(size_ngh=2, n_seeds_S=2, verbose=verbose)
    ossx, ossy = OSS.fit_transform(x, y)
    # 'Neighboorhood Cleaning Rule'
    NCR = NeighbourhoodCleaningRule(size_ngh=2, verbose=verbose)
    ncrx, ncry = NCR.fit_transform(x, y)

    ratio = 0.1
    # 'Random over-sampling'
    OS = OverSampler(ratio=ratio, verbose=verbose)
    osx, osy = OS.fit_transform(x, y)
    # 'SMOTE'
    smote = SMOTE(ratio=ratio, verbose=verbose, kind='regular')
    smox, smoy = smote.fit_transform(x, y)
    # 'SMOTE bordeline 1'
    bsmote1 = SMOTE(ratio=ratio, verbose=verbose, kind='borderline1')
    bs1x, bs1y = bsmote1.fit_transform(x, y)
    # 'SMOTE bordeline 2'
    bsmote2 = SMOTE(ratio=ratio, verbose=verbose, kind='borderline2')
    bs2x, bs2y = bsmote2.fit_transform(x, y)
    # 'SMOTE SVM'
    svm_args = {'class_weight': 'auto'}
    svmsmote = SMOTE(ratio=ratio, verbose=verbose, kind='svm', **svm_args)
    svsx, svsy = svmsmote.fit_transform(x, y)
    # 'SMOTE Tomek links'
    STK = SMOTETomek(ratio=ratio, verbose=verbose)
    stkx, stky = STK.fit_transform(x, y)
    # 'SMOTE ENN'
    SENN = SMOTEENN(ratio=ratio, verbose=verbose)
    ennx, enny = SENN.fit_transform(x, y)

    # 'EasyEnsemble'
    EE = EasyEnsemble(verbose=verbose)
    eex, eey = EE.fit_transform(x, y)
    # 'BalanceCascade'
    BS = BalanceCascade(verbose=verbose)
    bsx, bsy = BS.fit_transform(x, y)


if __name__ == '__main__':
    main()
