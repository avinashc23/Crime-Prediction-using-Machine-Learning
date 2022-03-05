import numpy as np
import pandas as pd

dataset = pd.read_csv('Crimes_2001_to_2022 .csv',low_memory=False)
dataset=dataset.drop(columns=['Case_Number','Date','Updated_On','IUCR','Arrest','Domestic','FBI_Code','Block'])
X = dataset.iloc[:, :-2].values
# X.shape
y = dataset.iloc[:, -2].values
# dataset['Location']=

print('\n Displaying DataSets for X \n', X)
print('\n Displaying DataSets for y \n', y)
# dataset.shape
# print('Columns in dataset: ', dataset.columns)
# dataset_=dataset.replace('[^\d.]','',regex=True).astype(float)
# dataset_
# dataset.dtypes
dataset.head(10)

# dataset.Date=pd.to_datetime(dataset.Date)
# dataset.Updated_On=pd.to_datetime(dataset.Updated_On)

# # dataset.Location= dataset.Location.astype(float)
# dataset.head(5)

print('Displaying dataset,\n') 
dataset.head()

# filling the empty values in dataset
# imputer is a transformer for missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent',verbose=0)
imputer = imputer.fit(dataset.iloc[: , :])
print(' Displaying DataSets \n', dataset)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(X_train, y_train)
print('Accuracy of Logistic Regression', lr.score(X_test, y_test))

from sklearn.svm import SVC
svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
print('Accuracy of SVM',svm.score(X_test, y_test))

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)
print('Accuracy of Naive Bayes',nb.score(X_test,y_test))

dataset.iloc[:,:] = imputer.transform(dataset.iloc[: ,:])
# filled_dataset = dataset
# dataset.iloc[:,:22]=X[:, :22]
print('\nFilling the value using Most Frequent method: \n')
print(' Displaying DataSets \n', dataset)

dataset.shape


# newds=dataset
newds = pd.read_csv('Crimes_2001_to_2022 .csv',usecols=['PrimaryType','Description','Location_Description'], low_memory=False)
newds.head(20)

# find out how many unique labels are there in data columns 
for col in newds.columns:
    print(col , ':', len(newds[col].unique()),'labels')
    
# tells us how many row and colums we want for one hot encoding
pd.get_dummies(newds, drop_first=True).shape

# display the top 15 values count 
newds.PrimaryType.value_counts().sort_values(ascending=False).head(15)

# list of top 15 most frequent value
top_15 = [x for x in newds.PrimaryType.value_counts().sort_values(ascending=False).head(15).index]
top_15

# now we make the 15 binary variables
for label in top_15:
    newds[label] = np.where(newds['PrimaryType']==label , 1,0 )
    
newds[['PrimaryType']+top_15].head(15)

# get whole of dummies variable for all the categorical variables
# function to do this
def one_hot_enc_top_x(df,variable,top_x_labels):
    for label in top_x_labels:
        df[variable+'_'+label]=np.where(newds[variable]==label,1,0)

newds = pd.read_csv('Crimes_2001_to_2022 .csv',usecols=['PrimaryType','Description','Location_Description'], low_memory=False)
one_hot_enc_top_x(dataset,'PrimaryType',top_15)
dataset=dataset.drop(columns=['PrimaryType'])
dataset.head()

# for Description we use 20
newds.Description.value_counts().sort_values(ascending=False).head(15)

# for Description we use 15
top_15 = [x for x in newds.Description.value_counts().sort_values(ascending=False).head(15).index]
# now creating dummy variables
one_hot_enc_top_x(dataset,'Description',top_15)
dataset=dataset.drop(columns=['Description'])
dataset.head()

# for Location_Description we use 15
top_15 = [x for x in newds.Location_Description.value_counts().sort_values(ascending=False).head(15).index]
# now creating dummy variables
one_hot_enc_top_x(dataset,'Location_Description',top_15)
dataset=dataset.drop(columns=['Location_Description'])
dataset.head(15)

dataset.head(10)
# dataset=dataset.drop(columns=['Case_Number','Date','Updated_On','IUCR','Arrest','Domestic','FBI_Code','Block'])

X = dataset.iloc[:, :-2].values
# X.shape
# y = dataset.iloc[:, -2].values
# # dataset['Location']=

print('\n Displaying DataSets for X \n', X)
print('\n Displaying DataSets for y \n', y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train
# from sklearn.linear_model import LogisticRegression

# lr = LogisticRegression(solver='liblinear',random_state=0)
# lr.fit(X_train, y_train)
# print('Accuracy of Logistic Regression', lr.score(X_test, y_test))
