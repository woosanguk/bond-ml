from pandas import DataFrame
import numpy as np
import pandas as pd

df = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                'key2': ['one', 'two', 'one', 'two', 'one'],
                'data1': np.random.randn(5),
                'data2': np.random.randn(5)
                })

print(df)

grouped = df['data1'].groupby(df['key1'])
print(grouped.mean())

means = df['data1'].groupby([df['key1'], df['key2']]).mean()

print(means)

print(means.unstack())

states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])

print(df['data1'].groupby([states, years]).mean())

print(df.groupby('key1').mean())
print(df.groupby(['key1', 'key2']).mean())
from pandas import DataFrame
import numpy as np

df = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                'key2': ['one', 'two', 'one', 'two', 'one'],
                'data1': np.random.randn(5),
                'data2': np.random.randn(5)
                })

print(df)

grouped = df['data1'].groupby(df['key1'])
print(grouped.mean())

means = df['data1'].groupby([df['key1'], df['key2']]).mean()

print(means)

print(means.unstack())

states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])

print(df['data1'].groupby([states, years]).mean())

print(df.groupby('key1').mean())
print(df.groupby(['key1', 'key2']).mean())

for name, group in df.groupby('key1'):
    print(name)

pieces = dict(list(df.groupby('key1')))

print(pieces['b'])

tips = pd.read_csv('data/ch08/tips.csv')
print(tips)

tips['tip_pct'] = tips['tip'] / tips['total_bill']

print(tips.tail())

grouped = tips.groupby(['sex', 'smoker'])

grouped_pct = grouped['tip_pct']

print(grouped_pct.agg('mean'))