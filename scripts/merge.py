import pandas as pd

df1 = pd.DataFrame(
        [
            ['id1', 10, 'C1', 'D1'],
            ['id2', 40, 'C2', 'D2'],
            ['id3', 50, 'C3', 'D3'],
            ['id4', 20, 'C4', 'D4']
        ], columns=['id', 'age', 'letter', 'another']
)

df1 = df1.set_index('id')


df2 = pd.DataFrame(
        [
            ['id1', 2, 38],
            ['id1', 4, 28],
            ['id2', 3, 18],
            ['id2', 5, 48],
            ['id3', 1, 10]
        ], columns=['id', 'day', 'secs_elapsed']
)

grouped = df2.groupby('id')['day'].apply(lambda x: x.tolist())

result = pd.concat([df1, grouped], axis=1)


def to_series(x):
    return pd.Series(x)

degrouped = result['day'].apply(to_series)

degrouped.columns = ['day_' + str(col) for col in degrouped.columns]

result = pd.concat([df1, degrouped], axis=1)

print df1
print df2
print
print result
