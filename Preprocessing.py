import numpy as np
import pandas as pd

dataset = pd.read_csv('Dataset/Crimes_2001_to_Present.csv')
# X = dataset.iloc[:, :-14].values
X = dataset.iloc[:, 1:22].values

print('Displaying DataSets for X \n', X)
# print('\n Displaying DataSets for y \n', y)

# dealing with the missing values
# Imputer class has been deprecated in newer versions so better use SimpleImpute
from sklearn.impute import SimpleImputer

# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# # axis 0 means mean of column, axi =1 means mean of row
# # strategy mean, median, most_frequent
# imputer = imputer.fit(X[:, 1:22])
# X[:, 1:22] = imputer.transform(X[:, 1:22])
# print('Filling the value using Mean method: \n')
# print('Displaying DataSets for X \n', X)
# # print('\n Displaying DataSets for y \n', y)

# imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# imputer = imputer.fit(X[:, 1:22])
# print('\n Splitting the Values In X \n')
# print('Displaying DataSets for X \n', X)
# X[:, 1:22] = imputer.transform(X[:, 1:22])
# print('Filling the value using Median method: \n')
# print('Displaying DataSets for X \n', X)
# # print('\n Displaying DataSets for y \n', y)

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(X[:, 1:22])
print(' Splitting the Values In X \n')
print(' Displaying DataSets for X \n', X)
X[:, 1:22] = imputer.transform(X[:, 1:22])
print('Filling the value using Most Frequent method: \n')
print(' Displaying DataSets for X \n', X)
# print('\n Displaying DataSets for y \n', y)

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
print('\n Displaying DataSets in X after Encoding the first column \n', X)

# Thus, we should also use OneHotEncoding by adding dummy columns as per number of distinct values in column country
# onehotencoder = OneHotEncoder()
# X = onehotencoder.fit_transform(X).toarray()
# labelEncoder_y = LabelEncoder()
# y = labelEncoder_y.fit_transform(y)
print('\n DataSets After Encoding and Sorting the column \n')
print('Displaying DataSets for X \n', X)
# print('\n Displaying DataSets for y \n', y)
