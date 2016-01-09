import pandas as pd


def country_probabilities(df):
    user_age = df['age']
    user_gender = df['gender']

    gender_mask = bkts['gender'] == user_gender
    age_mask = (bkts['from_age'] <= user_age) & (user_age <= bkts['to_age'])
    similar_population = bkts.loc[age_mask & gender_mask]

    for index, row in similar_population.iterrows():
        country = row['country_destination']
        df['similar_population_in_' + country] = row['population_in_thousands'] # / country_population[country]

    return df


path = '../datasets/processed/'
train_users = pd.read_csv(path + 'processed_train_users.csv')
test_users = pd.read_csv(path + 'processed_test_users.csv')

path = '../datasets/raw/'
bkts = pd.read_csv(path + 'age_gender_bkts.csv')

users = pd.concat((train_users, test_users), axis=0, ignore_index=True)

bkts.loc[bkts['age_bucket'] == '100+', 'age_bucket'] = '100-200'

bkts['from_age'] = bkts.age_bucket.str.split('-').apply(pd.Series, 2)[0].astype(int)
bkts['to_age'] = bkts.age_bucket.str.split('-').apply(pd.Series, 2)[1].astype(int)

bkts.drop('age_bucket', axis=1, inplace=True)

bkts.loc[bkts['gender'] == 'male', 'gender'] = 'MALE'
bkts.loc[bkts['gender'] == 'female', 'gender'] = 'FEMALE'

country_population = bkts.groupby('country_destination')['population_in_thousands'].sum()

train_users = train_users.apply(country_probabilities, axis=1)
test_users = test_users.apply(country_probabilities, axis=1)

train_users.to_csv('train_users_extra.csv')
test_users.to_csv('test_users_extra.csv')
