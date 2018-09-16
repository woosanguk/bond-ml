import pandas as pd
from pandas import Series
from lxml.html import parse
import csv
import json

print(pd.read_csv('data/ch06/ex1.csv'))
print(pd.read_csv('data/ch06/ex1.csv', sep=','))
print(pd.read_csv('data/ch06/ex2.csv'))
print(pd.read_csv('data/ch06/ex2.csv', header=None))
print(pd.read_csv('data/ch06/ex2.csv', names=['a', 'b', 'c', 'd', 'message']))
print(pd.read_csv('data/ch06/ex2.csv', names=['a', 'b', 'c', 'd', 'message'], index_col='message'))
print(pd.read_csv('data/ch06/csv_mindex.csv', index_col=['key1', 'key2']))
print(list(open('data/ch06/ex3.txt')))
print(pd.read_table('data/ch06/ex3.txt', sep='\s+'))
print(pd.read_csv('data/ch06/ex4.csv', skiprows=[0, 2, 3]))
print(pd.read_csv('data/ch06/ex5.csv'))
print(pd.isnull(pd.read_csv('data/ch06/ex5.csv')))
print(pd.read_csv('data/ch06/ex5.csv', na_values=['NULL']))
print(pd.read_csv('data/ch06/ex5.csv', na_values={'message': ['foo', 'NA'], 'something': ['two']}))
print(pd.read_csv('data/ch06/ex6.csv'))
print(pd.read_csv('data/ch06/ex6.csv', nrows=5))
chunker = pd.read_csv('data/ch06/ex6.csv', chunksize=1000)
tot = Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)
tot = tot.sort_values(ascending=False)
print(tot[:10])

f = open('data/ch06/ex7.csv')
reader = csv.reader(f)
for line in reader:
    print(line)

lines = list(csv.reader(open('data/ch06/ex7.csv')))
print(lines)
header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}
print(data_dict)

# print(pd.read_csv('data/ch06/ex7.csv'))
