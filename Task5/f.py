import pandas as pd
from sklearn import datasets

iris_data = datasets.load_iris()

df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

df.describe()