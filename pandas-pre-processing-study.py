from pandas import DataFrame
from pandas import Series
import pandas as pd
import numpy as np
import json

df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df2 = DataFrame({'key': ['a', 'b', 'd'], 'data1': range(3)})

print(pd.merge(df1, df2))
print(pd.merge(df1, df2, on='key'))

df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df4 = DataFrame({'rkey': ['a', 'b', 'd'], 'data1': range(3)})

print(pd.merge(df3, df4))
print(pd.merge(df3, df4, left_on='lkey', right_on='rkey'))

print(pd.merge(df1, df2, how='outer'))

left1 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])

print(pd.merge(left1, right1, left_on='key', right_index=True))

arr = np.arange(12).reshape((3, 4))
print(np.concatenate([arr, arr], axis=1))

s1 = Series([0, 1], index=['a', 'b'])
s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = Series([5, 6], index=['f', 'g'])

print(pd.concat([s1, s2, s3]))

print(pd.concat([s1, s2, s3], axis=1, sort=True))

s4 = pd.concat([s1 * 5, s3])

print(pd.concat([s1, s4], axis=1, sort=True))

print(pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']]))

np.random.seed(12345)

data = DataFrame(np.random.randn(1000, 4))

print(data)
print(data.describe())

col = data[3]

print(col[np.abs(col) > 3])

print(data[(np.abs(data) > 3)])
print(data[(np.abs(data) > 3).any(1)])

data[np.abs(data) > 3] = np.sign(data) * 3

print(data.describe())

df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})

print(pd.get_dummies(df['key']))

dummies = pd.get_dummies(df['key'], prefix='key')

print(dummies)

df_with_dummy = df[['data1']].join(dummies)

print(df_with_dummy)

# mnames = ['movie_id', 'title', 'genres']
#
# movies = pd.read_table('data/ch07/movies.dat', sep='::')


db = json.load(open('data/ch07/foods-2011-10-03.json'))

print(len(db))

print(db[0].keys())
print(db[0]['nutrients'][0])

nutrients = DataFrame(db[0]['nutrients'])

print(nutrients[:7])

info_keys = ['description', 'group', 'id', 'manufacturer']

info = DataFrame(db, columns=info_keys)
print(info[:5])

print(info)

print(pd.value_counts(info.group)[:10])

nutrients = []

for rec in db:
    fnuts = DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)

nutrients = pd.concat(nutrients, ignore_index=True)

print(nutrients)

print(nutrients.duplicated().sum())

nutrients = nutrients.drop_duplicates()
col_mapping = {'description': 'food', 'group' : 'fgroup'}

info = info.rename(columns=col_mapping, copy=False)

print(info)

col_mapping = {'description': 'nutrient', 'group' : 'nutgroup'}

nutrients = nutrients.rename(columns=col_mapping, copy=False)
ndata = pd.merge(nutrients, info, on='id', how='outer')

print(ndata.ix[30000])