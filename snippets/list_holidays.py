import pandas as pd
from datetime import date
import holidays

def process_holidays(df):
    year = df['year']
    user_date = date(year, df['month'], df['day'])
    hd = holidays.US(years=year)
    for holiday_date, name in hd.iteritems():
        days = (holiday_date - user_date).days

        # Get the real data for the new year
        if days < 0:
            days = days + 364

        name = "".join([c for c in name if c.isalpha() or c.isdigit() or c==' ']).rstrip().lower().replace(" ", "_")

        df['days_to_' + name] = days

    return df

df = pd.DataFrame()
df['year'] = pd.Series(range(2010, 2015))
df['day'] = pd.Series(range(11, 27, 3))
df['month'] = pd.Series(range(2, 12, 2))

print df.apply(process_holidays, axis=1)
