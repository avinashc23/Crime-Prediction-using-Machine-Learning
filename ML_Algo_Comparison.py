import numpy as np
import pandas as pd

dataset = pd.read_csv('Dataset\Crimes_2001_to_2022 _v1.csv', low_memory=False)
print('Columns in dataset: ', dataset.columns)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 20].values

print('\n Displaying DataSets for X \n', X)
print('\n Displaying DataSets for y \n', y)

print('Displaying dataset,', dataset.head(5))

from sklearn.utils import shuffle

dataset = shuffle(dataset)
print("After shuffling the dataset")

print('Displaying dataset,', dataset.head(5))

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(X[:, 1:22])

print(' Splitting the Values In X \n')
print(' Displaying DataSets for X \n', X)

X[:, 1:22] = imputer.transform(X[:, 1:22])
print('Filling the value using Most Frequent method: \n')
print(' Displaying DataSets for X \n', X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Thus, we should also use OneHotEncoding by adding dummy columns as per number of distinct values in column country
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)
print('\n DataSets After Encoding and Sorting the column \n')
print('Displaying DataSets for X \n', X)
print('\n Displaying DataSets for y \n', y)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder()
X[:, 22] = labelEncoder_X.fit_transform(X[:, 22])
print('\n Displaying DataSets in X after Encoding the first column \n', X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(X_train, y_train)
print('Accuracy of Logistic Regression', lr.score(X_test, y_test))

from sklearn.svm import SVC

svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
print('Accuracy of SVM', svm.score(X_test, y_test))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
print('Accuracy of KNN', knn.score(X_test, y_test))

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
print('Accuracy of Naive Bayes', nb.score(X_test, y_test))

from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
scores_logistic = []
for train_index, test_index in folds.split(X_train, y_train):
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    lr = LogisticRegression(solver='liblinear', multi_class='ovr')
    lr.fit(X_train_fold, y_train_fold)
    print('Accuracy of Logistic Regression by k fold: ', scores_logistic.append(lr.score(X_train_fold, y_train_fold)))
