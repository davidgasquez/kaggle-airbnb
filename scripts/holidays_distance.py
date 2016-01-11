#!/usr/bin/env python

import pandas as pd
from datetime import date
import holidays


def sanitize_holiday_name(name):
    new_name = [c for c in name if c.isalpha() or c.isdigit() or c == ' ']
    new_name = "".join(new_name).lower().replace(" ", "_")
    return new_name


def process_holidays(df):
    # Create a date object
    user_date = date(
        df['year_account_created'],
        df['month_account_created'],
        df['day_account_created']
    )

    # Get US holidays for this year
    holidays_dates = holidays.US(years=df['year_account_created'])

    for holiday_date, name in holidays_dates.iteritems():
        # if 'observed' in name:
        #     pass

        # Compute difference in days
        days = (holiday_date - user_date).days

        # Clean holiday name
        name = sanitize_holiday_name(name)

        # Add the computed days to holiday into our DataFrame
        df['days_to_' + name] = days

    return df


def main():
    path = '../datasets/processed/'
    train_users = pd.read_csv(path + 'processed_train_users.csv')
    test_users = pd.read_csv(path + 'processed_train_users.csv')

    train_users = train_users.apply(process_holidays, axis=1)
    test_users = test_users.apply(process_holidays, axis=1)

    train_users.to_csv('train_users_with_holidays.csv')
    test_users.to_csv('test_users_with_holidays.csv')

if __name__ == '__main__':
    main()
