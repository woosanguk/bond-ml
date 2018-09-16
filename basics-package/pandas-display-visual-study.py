import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn

# fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# plt.plot(randn(50).cumsum(), 'k--')
# ax2 = fig.add_subplot(2, 2, 2)
# plt.plot(np.arange(10))
# ax3 = fig.add_subplot(2, 2, 3)
# _ = ax3.hist(randn(100), bins=20, color='k', alpha=0.3)
# ax4 = fig.add_subplot(2, 2, 4)
# ax4.scatter(np.arange(30), np.arange(30) + 3 * randn(30))

# plt.show()

# fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
#
# for i in range(2):
#     for j in range(2):
#         axes[i, j].hist(randn(500), bins=50, color='k', alpha=0.5)
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(randn(1000).cumsum())
# ticks = ax.set_xticks([0, 250, 500, 750, 1000])
# labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30, fontsize='small')
# ax.set_title('My first matplotlib plot')
# ax.set_xlabel('Stages')
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(randn(1000).cumsum(), 'k', label='one')
# ax.plot(randn(1000).cumsum(), 'k--', label='two')
# ax.plot(randn(1000).cumsum(), 'k.', label='three')
# ax.legend(loc='best')
# plt.show()

import pandas as pd
from pandas import DataFrame


def to_cat_list(catstr):
    stripped = (x.strip() for x in catstr.split(','))
    return [x for x in stripped if x]


def get_all_categories(cat_series):
    cat_sets = (set(to_cat_list(x)) for x in cat_series)
    return sorted(set.union(*cat_sets))


def get_english(cat):
    code, names = cat.split('.')
    if '|' in names:
        names = names.split('|')[1]
    return code, names.strip()


def get_code(seq):
    return [x.split('.')[0] for x in seq if x]


data = pd.read_csv('data/ch08/Haiti.csv')

print(data.tail())
print(data.columns)
print(data[['INCIDENT TITLE', 'LATITUDE', 'LONGITUDE']][:10])
print(data['CATEGORY'][:6])

data = data[(data.LATITUDE > 18) & (data.LATITUDE < 20) &
            (data.LONGITUDE > -75) & (data.LONGITUDE < -70) &
            data.CATEGORY.notnull()]

print(data.tail())

print(get_english('1. Urgences | Emergency'))

all_cats = get_all_categories(data.CATEGORY)

print(all_cats)

english_mapping = dict(get_english(x) for x in all_cats)

print(english_mapping['2a'])
print(english_mapping['6c'])

all_codes = get_code(all_cats)
code_index = pd.Index(np.unique(all_codes))
dummy_frame = DataFrame(np.zeros((len(data), len(code_index))), index=data.index, columns=code_index)

print(dummy_frame.ix[:, :6])
