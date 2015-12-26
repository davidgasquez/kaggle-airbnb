#!/usr/bin/env python

import pandas as pd


def main():
    """Main function."""

    print "Loading data...",
    path = '../datasets/processed/'
    train_users = pd.read_csv(path + 'preprocessed_train_users.csv')
    test_users = pd.read_csv(path + 'preprocessed_test_users')

    target = train_users['country_destination']
    train_users.drop('country_destination', inplace=True)
    train_users.drop('id', inplace=True)

    x_train = train_users.values

    test_users_ids = test_users['id']
    test_users.drop('id', inplace=True)

    x_test = test_users.values
    print "\tDONE"

if __name__ == '__main__':
    main()
