import numpy as np
import pandas as pd

dataset = pd.read_csv('Dataset/Sample.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, :21].values

print('Displaying DataSets for X \n', X)
print('\n Displaying DataSets for y \n', y)

# dealing with the missing values
# Imputer class has been deprecated in newer versions so better use SimpleImpute
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(X[:, 1:22])
print(' Splitting the Values In X \n')
print(' Displaying DataSets for X \n', X)
X[:, 1:22] = imputer.transform(X[:, 1:22])
print('Filling the value using Most Frequent method: \n')
print(' Displaying DataSets for X \n', X)
print('\n Displaying DataSets for y \n', y)

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder()
X[:, 22] = labelEncoder_X.fit_transform(X[:, 22])
print('\n Displaying DataSets in X after Encoding the first column \n', X)

# Thus, we should also use OneHotEncoding by adding dummy columns as per number of distinct values in column country
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)
print('\n DataSets After Encoding and Sorting the column \n')
print('Displaying DataSets for X \n', X)
print('\n Displaying DataSets for y \n', y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='liblinear',multi_class='ovr')
lr.fit(X_train, y_train)
print('Accuracy of Logistic Regression', lr.score(X_test, y_test))
