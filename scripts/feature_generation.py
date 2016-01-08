#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA


def interaction_features(data, degree):
    poly = PolynomialFeatures(degree, interaction_only=True)

    interaction_data = poly.fit_transform(data)

    interaction = pd.DataFrame(interaction_data).drop(0, axis=1)

    base_columns = np.shape(data)[1] + 1
    interaction = interaction.ix[:, base_columns:]

    # Drop empty columns
    y = lambda x: np.all(x == 0)
    drop_columns = interaction.columns[interaction.apply(y)]
    df = interaction.drop(drop_columns, axis=1)

    return df


def main():
    path = '../datasets/processed/'
    train_users = pd.read_csv(path + 'processed_train_users.csv')
    test_users = pd.read_csv(path + 'processed_train_users.csv')

    train_users = train_users.fillna(0)
    test_users = test_users.fillna(0)

    # Remove country_destination from train_users
    train_users.drop('country_destination', axis=1, inplace=True)

    interaction_columns = [
        'age',
        'day_account_created',
        'day_first_active',
        'day_pauses',
        'elapsed_secs_average',
        'elapsed_secs_median',
        # 'elapsed_secs_skew',
        # 'elapsed_secs_std',
        # 'elapsed_secs_sum',
        # 'elapsed_secs_var',
        # 'first_quantile',
        # 'second_quantile',
        # 'third_quantile',
        # 'fourth_quantile',
        # 'first_secs_elapsed',
        # 'last_secs_elapsed',
        # 'long_sessions',
        # 'max_secs_elapsed',
        # 'min_secs_elapsed',
        # 'month_account_created',
        # 'month_first_active',
        'session_length',
        # 'short_sessions',
        'weekday_account_created',
        'weekday_first_active',
        # 'year_account_created',
        # 'year_first_active'
    ]

    pca = PCA(n_components=10)

    # Add features to train users
    train_df = interaction_features(train_users[interaction_columns], 3)
    pca_train = pd.DataFrame(pca.fit_transform(train_users.drop('id', axis=1)))

    train_users = pd.read_csv(path + 'train_users.csv')
    train_users = pd.concat([train_users, train_df, pca_train], axis=1)
    train_users.to_csv('full_train_users.csv')

    # Add features to test users
    test_df = interaction_features(test_users[interaction_columns], 3)
    pca_test = pd.DataFrame(pca.fit_transform(test_users.drop('id', axis=1)))

    test_users = pd.read_csv(path + 'test_users.csv')
    test_users = pd.concat([test_users, test_df, pca_test], axis=1)
    test_users.to_csv('full_test_users.csv')

if __name__ == '__main__':
    main()
