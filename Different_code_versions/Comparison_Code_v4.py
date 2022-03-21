import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

dataset = pd.read_csv('Crimes_2001_to_Present (1).csv',low_memory=False)
 
dataset=dataset.drop(columns=['ID','Case Number','Description','Updated On','IUCR','Block','FBI Code'])
#

print('Columns in dataset: ', dataset.columns)
# dataset.shape

# dataset.dtypes

# droping the null value enteries
dataset.dropna(inplace=True)
dataset

scaler = preprocessing.MinMaxScaler()
dataset[['Beat']] = scaler.fit_transform(dataset[['Beat']])
dataset[['X Coordinate', 'Y Coordinate']] = scaler.fit_transform(dataset[['X Coordinate', 'Y Coordinate']])
dataset

import datetime
dataset.Date=pd.to_datetime(dataset.Date)

dataset.head(10)

dataset['month']=dataset.Date.dt.month
dataset['weekday'] = dataset.Date.dt.day_of_week

dataset=dataset.drop(columns='Date')
dataset.head(10)

labelEncoder = LabelEncoder()

locDes_enc = labelEncoder.fit_transform(dataset['Primary Type'])
dataset['Primary Type'] = locDes_enc

dataset.head()

labelEncoder2 = LabelEncoder()

locDes_enc = labelEncoder2.fit_transform(dataset['Location Description'])
dataset['Location Description'] = locDes_enc

dataset.head()

# using correlation for the feature selection
corelation = dataset.corr()
corelation

plt.figure(figsize=(10,7))
sns.heatmap(corelation,annot=True)

# month week day have low correlation they isn't effect our results so we drop them
# since beat have high correlation with district so we drop beat
# and X cordinate have high correlation with longitube and Y cordinate with latitude so we drop longitude and latitude
# we already select year wise data previously
selected_cols=['Primary Type','Location Description','Domestic','District','Ward', 'Community Area', 'X Coordinate','Y Coordinate']


# our target feature
y = dataset["Arrest"]
print(y,"\n")

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(dataset[selected_columns],dataset['PrimaryType'],test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(dataset[selected_cols],y,test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10) 
knn.fit(X_train,y_train)
print('Accuracy of KNN', knn.score(X_test, y_test))
# print(knn_5.score(X_test,y_test))
pred_train = knn.predict(X_train)
pred_i = knn.predict(X_test)
print('Test accuracy ', metrics.accuracy_score(y_train, pred_train))
print('Accuracy ', metrics.accuracy_score(y_test, pred_i))

error_rate = []

krange = range(10,50,10)
for i in krange:
 
 knn = KNeighborsClassifier(n_neighbors=i, metric='manhattan', weights = 'uniform',n_jobs= -1) 
 knn.fit(X_train,y_train)
 predicted_train = knn.predict(X_train)
 pred_i = knn.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test))
 print(' Neighbours  ',i)
 print('Test accuracy ', metrics.accuracy_score(y_train, predicted_train))
 print('Accuracy ', metrics.accuracy_score(y_test, pred_i))

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(X_train, y_train)
print('Accuracy of Logistic Regression', lr.score(X_test, y_test))
