#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def polinomial_features(data, degree):
    # Generate 3 degrees
    poly = PolynomialFeatures(degree)

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
    train_users = pd.read_csv(path + 'train_users.csv')
    test_users = pd.read_csv(path + 'test_users.csv')

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
        'elapsed_secs_skew',
        'elapsed_secs_std',
        'elapsed_secs_sum',
        'elapsed_secs_var',
        'first_quantile',
        'first_secs_elapsed',
        'fourth_quantile',
        'last_secs_elapsed',
        'long_sessions',
        'max_secs_elapsed',
        'min_secs_elapsed',
        'month_account_created',
        'month_first_active',
        'second_quantile',
        'session_length',
        'short_sessions',
        'third_quantile',
        'weekday_account_created',
        'weekday_first_active',
        'year_account_created',
        'year_first_active',
        'gender_FEMALE',
        'gender_MALE',
        'gender_OTHER'
    ]

    train_df = polinomial_features(train_users[interaction_columns], 3)

    train_users = pd.read_csv(path + 'train_users.csv')
    train_users = pd.concat([train_users, train_df], axis=1)
    train_users.to_csv('train_users_interaction.csv')

    test_df = polinomial_features(test_users[interaction_columns], 3)

    test_users = pd.read_csv(path + 'test_users.csv')
    test_users = pd.concat([test_users, test_df], axis=1)
    test_users.to_csv('test_users_interaction.csv')

if __name__ == '__main__':
    main()
