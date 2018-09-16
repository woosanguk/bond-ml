from pandas import Series
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7), datetime(2011, 1, 8), datetime(2011, 1, 10),
         datetime(2011, 1, 12)]

ts = Series(np.random.randn(6), index=dates)

longer_ts = Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))

# print(longer_ts.tail())

# print(longer_ts['2001'])
# print(longer_ts['2001-05'])
# print(longer_ts['2001-04':'2001-05'])
# print(longer_ts.truncate(after='2001-12'))

dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000', '1/3/2000'])
dup_ts = Series(np.arange(5), index=dates)
grouped = dup_ts.groupby(level=0)
# print(dup_ts)
# print(dup_ts.index.is_unique)
# print(grouped.mean())
# print(grouped.count())

# print(ts.tail())
# a = ts.resample('D')
# print(a)

rng = pd.date_range('1/1/2000', periods=100, freq='D')
ts = Series(np.random.randn(len(rng)), index=rng)

# print(rng)
# print(ts)
#
# print(ts.resample('M').mean())
# print(ts.resample('M', kind='period').mean())
#
# print(ts.resample('T').mean())

rng = pd.date_range('1/1/2000', periods=12, freq='T')
ts = Series(np.arange(12), index=rng)
# print(rng)
# print(ts)
#
# print(ts.resample('5min').sum())
# print(ts.resample('5min', closed='right').sum())
# print(ts.resample('5min', closed='right', label='right').sum())
# print(ts.resample('5min', loffset='-1s').sum())
# print(ts.resample('5min').ohlc())

close_px_all = pd.read_csv('data/ch09/stock_px.csv', parse_dates=True, index_col=0)
close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
close_px = close_px.resample('B').ffill()

# close_px['AAPL'].plot()
# close_px.ix['2009'].plot()
# close_px['AAPL'].ix['01-2011':'03-2011'].plot()
appl_q = close_px['AAPL'].resample('Q-DEC').ffill()
# appl_q.ix['2009':].plot()



# fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(12, 7))
#
# aapl_px = close_px.AAPL['2005':'2009']
#
# pd.ewma # 지수 가중 함수
#
# ma60 = pd.rolling_
#
# plt.show()

spx_px = close_px_all['SPX']

spx_rets = spx_px / spx_px.shift(1) - 1
returns = spx_rets.pct_change()
# print(returns)
corr = returns.rolling(window=125, min_periods=100)
corr.plot()

plt.show()
