from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

iris_dataset = load_iris()

# print(iris_dataset.keys())
# print(iris_dataset['data'])
# print(iris_dataset['data'].shape)
# print(iris_dataset['target'])

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# pd.plotting.scatter_matric(iris_dataset, c=y_train, figsize=(15, 15), market='o', hist_kwds={'bins': 20}, s=60,
#                            alpha=.8)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(X_new)

print('예측: {}'.format(prediction))
print('예측 타킷 : {}'.format(iris_dataset['target_names'][prediction]))


y_pred = knn.predict(X_test)

print('정확도: {:.2f}'.format(np.mean(y_pred == y_test)))
print('테스트 세트 정확도: {:.2f}'.format(knn.score(X_test, y_test)))

# print(iris_dataframe)

# plt.show()
# print(iris_dataset)
