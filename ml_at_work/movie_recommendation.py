from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from fastFM import mcmc
from fastFM import als
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def basic():
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip-code']

    users = pd.read_csv('data/ml-100k/u.user', sep='|', names=u_cols)

    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=r_cols)
    ratings['date'] = pd.to_datetime(ratings['unix_timestamp'], unit='s')

    m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
    movies = pd.read_csv('data/ml-100k/u.item', sep='|', names=m_cols, usecols=range(5), encoding="latin1")

    movie_rating = pd.merge(movies, ratings)
    lens = pd.merge(movie_rating, users)
    # print(lens.title.value_counts()[:25])

    movie_state = lens.groupby('title').agg({'rating': [np.size, np.mean]})
    # print(movie_state.sort_values(by=[('rating', 'mean')], ascending=False).head())

    atleast_100 = movie_state['rating']['size'] >= 100
    # print(movie_state[atleast_100].sort_values(by=[('rating', 'mean')], ascending=False).head())

    lens.groupby('user_id').size().sort_values(ascending=False).hist()

    # plt.style.use('ggplot')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    user_stats = lens.groupby('user_id').agg({'rating': [np.size, np.mean]})

    # print(user_stats['rating'].describe())

    # train = [
    #     {'user': '1', 'item': '5', 'age': 19},
    #     {'user': '2', 'item': '43', 'age': 33},
    #     {'user': '3', 'item': '20', 'age': 55},
    #     {'user': '4', 'item': '10', 'age': 20}
    # ]
    #
    # v = DictVectorizer()
    # X = v.fit_transform(train)
    #
    # y = np.array([5, 0, 1.0, 2.0, 4.0])
    #
    # fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
    # fm.fit(X, y)
    # print(fm.predict(v.transform({'user': '5', 'item': '10', 'age': 24})))
    # print(x)


def load_data(filename, path="data/ml-100k/"):
    data = []
    y = []
    users = set()
    items = set()
    with open(path + filename) as f:
        for line in f:
            (user, movieid, rating, ts) = line.split('\t')
            data.append({"user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

    return (data, np.array(y), users, items)


(dev_data, y_dev, dev_users, dev_items) = load_data("ua.base")
(test_data, y_test, test_users, test_items) = load_data("ua.test")

v = DictVectorizer()
X_dev = v.fit_transform(dev_data)
X_test = v.fit_transform(test_data)
np.std(y_test)
X_train, X_dev_test, y_train, y_dev_test = train_test_split(X_dev, y_dev, test_size=0.1, random_state=42)

n_iter = 300
step_size = 1
seed = 123
rank = 4
fm = mcmc.FMRegression(n_iter=0, rank=rank, random_state=seed)
fm.fit_predict(X_train, y_train, X_dev_test)

rmse_dev_test = []
rmse_test = []
hyper_param = np.zeros((n_iter -1, 3 + 2 * rank), dtype=np.float64)

for nr, i in enumerate(range(1, n_iter)):
    fm.random_state = i * seed
    y_pred = fm.fit_predict(X_train, y_train, X_dev_test, n_more_iter=step_size)
    rmse_test.append(np.sqrt(mean_squared_error(y_pred, y_dev_test)))
    hyper_param[nr, :] = fm.hyper_param_

values = np.arange(1, n_iter)
x = values * step_size
burn_in = 5
x = x[burn_in:]

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 8))

axes[0, 0].plot(x, rmse_test[burn_in:], label='dev test rmse', color="r")
axes[0, 0].legend()
axes[0, 1].plot(x, hyper_param[burn_in:,0], label='alpha', color="b")
axes[0, 1].legend()
axes[1, 0].plot(x, hyper_param[burn_in:,1], label='lambda_w', color="g")
axes[1, 0].legend()
axes[1, 1].plot(x, hyper_param[burn_in:,3], label='mu_w', color="g")
axes[1, 1].legend()

plt.show()

print(np.min(rmse_test))
