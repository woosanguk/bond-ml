from pandas import Series, DataFrame
import numpy as np
import pandas as pd
from datetime import time
import random
import string

ts1 = Series(np.random.randn(3), index=pd.date_range('2012-6-13', periods=3, freq='W-WED'))
# print(ts1)
ts1 = ts1.resample('B').ffill()
# print(ts1)

dates = pd.DatetimeIndex(['2012-6-12', '2012-6-17', '2012-6-18', '2012-6-21', '2012-6-22', '2012-6-29'])
ts2 = Series(np.random.randn(6), index=dates)
# print(ts2)
# print(ts1.reindex(ts2.index).ffill())
# print(ts2 + ts1.reindex(ts2.index, method='ffill'))

gdp = Series([1.78, 1.95, 2.08, 2.01, 2.15, 2.31, 2.46], index=pd.period_range('1984Q2', periods=7, freq='Q-SEP'))
inf1 = Series([0.025, 0.045, 0.037, 0.04], index=pd.period_range('1982', periods=4, freq='A-DEC'))
inf1_q = inf1.asfreq('Q-SEP', how='end')

# print(gdp)
# print(inf1)
# print(inf1_q)
#
# print(inf1_q.reindex(gdp.index, method='ffill'))

rng = pd.date_range('2012-06-01 09:30', '2012-06-01 15:59', freq='T')
rng = rng.append([rng + pd.offsets.BDay(i) for i in range(1, 4)])
# print(rng)

ts = Series(np.arange(len(rng), dtype=float), index=rng)

# print(ts)
#
# print(ts[time(10, 0)])
# print(ts.at_time(time(10, 0)))
# print(ts.between_time(time(10, 0), time(10, 1)))

indexer = np.sort(np.random.permutation(len(ts))[700:])

irr_ts = ts.copy()

irr_ts[indexer] = np.nan
# print(irr_ts['2012-06-01 09:50':'2012-06-01 10:00'])

selection = pd.date_range('2012-06-01 10:00', periods=4, freq='B')

# print(irr_ts.asof(selection))

data1 = DataFrame(np.ones((6, 3), dtype=float), columns=['a', 'b', 'c'], index=pd.date_range('6/12/2012', periods=6))
data2 = DataFrame(np.ones((6, 3), dtype=float) * 2, columns=['a', 'b', 'c'],
                  index=pd.date_range('6/13/2012', periods=6))
spliced = pd.concat([data1.ix[:'2012-06-14'], data2.ix['2012-06-15':]])
# print(data1)
# print(data2)
# print(spliced)

data2 = DataFrame(np.ones((6, 4), dtype=float) * 3, columns=['a', 'b', 'c', 'd'],
                  index=pd.date_range('6/13/2012', periods=6))
spliced = pd.concat([data1.ix[:'2012-06-14'], data2.ix['2012-06-15':]], sort=False)

# print(spliced)

# spliced_filled = spliced.combind_first(data2)

# print(spliced_filled)
spliced.update(data2, overwrite=False)
# print(spliced)

cp_spliced = spliced.copy()
cp_spliced[['a', 'c']] = data1[['a', 'c']]

# print(cp_spliced)

N = 1000
M = 500


def rands(n):
    choices = string.ascii_uppercase
    return ''.join([random.choice(choices) for _ in range(n)])


def zscore(group):
    return (group - group.mean()) / group.std()


tickers = np.array([rands(5) for _ in range(N)])
df = DataFrame({'Momentum': np.random.randn(M) / 200 + 0.03,
                'Value': np.random.randn(M) / 200 + 0.08,
                'ShortInterest': np.random.randn(M) / 200 - 0.02},
               index=tickers[:M])

ind_names = np.array(['FINANCIAL', 'THCH'])
sampler = np.random.randint(0, len(ind_names), N)
industries = Series(ind_names[sampler], index=tickers, name='industry')

by_industry = df.groupby(industries)
df_stand = by_industry.apply(zscore)

ind_rank = by_industry.rank(ascending=False)
# print(tickers)
# print(df)
# print(industries)
print(by_industry.mean())
print(by_industry.describe())
print(ind_rank)
# print(df_stand.groupby(industries).agg(['mean', 'std']))
fac1, fac2, fac3 = np.random.rand(3, 1000)

ticker_subset = tickers.take(np.random.permutation(N))