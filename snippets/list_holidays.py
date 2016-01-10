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
    user_date = date(df['year'], df['month'], df['day'])

    # Get US holidays for this year
    holidays_dates = holidays.US(years=df['year'])

    for holiday_date, name in holidays_dates.iteritems():
        # Compute difference in days
        days = (holiday_date - user_date).days

        # Past dates will happen more or less in the next year
        if days < 0:
            days = days + 364

        # Clean holiday name
        name = sanitize_holiday_name(name)

        # Add the computed days to holiday into our DataFrame
        df['days_to_' + name] = days

    return df


def main():
    df = pd.DataFrame()
    df['year'] = pd.Series(range(2010, 2015))
    df['day'] = pd.Series(range(11, 27, 3))
    df['month'] = pd.Series(range(2, 12, 2))

    print df.apply(process_holidays, axis=1)


if __name__ == '__main__':
    main()
