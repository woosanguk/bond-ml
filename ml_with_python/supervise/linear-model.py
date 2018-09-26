import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_wave(n_samples=60)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

print(lr.coef_)
print(lr.intercept_)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

print(lr.coef_)
print(lr.intercept_)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))

lr = Ridge().fit(X_train, y_train)

print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))
# print(mglearn.plots.plot_linear_regression_wave())
