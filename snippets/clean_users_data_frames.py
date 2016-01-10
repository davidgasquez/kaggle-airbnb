#!/usr/bin/env python

import pandas as pd


def main():
    path = '../datasets/processed/'
    train_users = pd.read_csv(path + 'processed_train_users.csv')
    test_users = pd.read_csv(path + 'processed_test_users.csv')

    percentage = 0.95

    train_mask = train_users.isnull().sum() > train_users.shape[0] * percentage
    train_to_remove = list(train_users.isnull().sum()[train_mask].index)

    test_mask = test_users.isnull().sum() > test_users.shape[0] * percentage
    test_to_remove = list(test_users.isnull().sum()[test_mask].index)

    to_remove = list(set(train_to_remove).intersection(test_to_remove))

    train_users.drop(to_remove, axis=1, inplace=True)
    test_users.drop(to_remove, axis=1, inplace=True)

    train_users.to_csv('clean_processed_train_users.csv')
    test_users.to_csv('clean_processed_test_users.csv')

if __name__ == '__main__':
    main()
