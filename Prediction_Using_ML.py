#importing libraries
import numpy as np
import pandas as pd
import math
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#loading dataset
dataset = pd.read_csv('Crimes_2001_to_2022.csv',low_memory=False)
 
dataset.head(10)

dataset.dtypes


# droping the features that are not usefull
dataset=dataset.drop(columns=['IUCR','Case Number','Description','FBI Code','Updated On','Block'])
# X.shape

print('Columns in dataset: ', dataset.columns)


dataset


# count the number of rows that contain missing values:
dataset.isna().any(axis=1).sum()

# droping the null value enteries 
dataset.dropna(inplace=True)

# counts after removal of null values from dataset
np.count_nonzero(dataset.isnull())

dataset 

# Before removing Null values 1048575

# After removing Null value 1015247

# Total Null values removed 33328

# Chicago is bounded by box: 41.6439,-87.9401; 41.9437,-87.5878
# Source: https://boundingbox.klokantech.com/

#exploring location column
dataset['Location']

print('Current rows:', dataset.shape[0])
dataset = dataset[(((dataset.Latitude >= 41.64) & (dataset.Longitude <= -87.50)) | 
            ((dataset.Latitude <= 41.94) & (dataset.Longitude >= -87.94)))]
print('Rows after removing out of box points:', dataset.shape[0])

dataFrame = dataset.copy()

data = dataFrame.pivot_table(index='Year', columns='Primary Type', aggfunc='count')
print(data)

plt.figure(figsize=(12,8))
sns.countplot(x='Year',data=dataFrame)
plt.ylabel('No of Crimes')
plt.show()

# Since we do not have crimes in starting years so we drop them 2020
# selecting the dataset which starts from 2015
dataFrame=dataFrame[dataFrame['Year']>=2015]
dataFrame=dataFrame[dataFrame['Year']<2020]
dataFrame

sns.countplot(x='Year',data=dataFrame)
plt.ylabel('No of Crimes')
plt.show()

# # listing the crimes type wise with their counts
types=dataFrame['Primary Type'].value_counts().sort_values(ascending=False)
types

plt.figure(figsize=(10,8))
sns.countplot(data=dataFrame, y="Primary Type", order=dataFrame['Primary Type'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Count of Crimes')
plt.show()

fourMajorTypes=dataFrame['Primary Type'].value_counts().sort_values(ascending=False)
fourMajorTypes=fourMajorTypes[:4]
fourMajorTypes

fourMajorTypes.plot(kind='bar',color='red')
plt.ylabel('Count')
plt.show()

# major crime time
#---> Storing Major Crimes
major_crimes=['THEFT','BATTERY','CRIMINAL DAMAGE','ASSAULT']

# crime types according to their counts in dataframe
# 15 classes
# major_crimes=['THEFT','BATTERY','CRIMINAL DAMAGE','ASSAULT','OTHER OFFENSE','DECEPTIVE PRACTICE','NARCOTICS','BURGLARY','MOTOR VEHICLE THEFT'
#               ,'ROBBERY','CRIMINAL TRESPASS','WEAPONS VIOLATION','OFFENSE INVOLVING CHILDREN','PUBLIC PEACE VIOLATION','CRIM SEXUAL ASSAULT']
# 8 classes
# storing major crime types according to their counts in dataframe
# major_crimes=['THEFT','BATTERY','CRIMINAL DAMAGE','ASSAULT','OTHER OFFENSE','DECEPTIVE PRACTICE','NARCOTICS','BURGLARY']

# selecting the data form our dataset that belongs major crime classes
crime_df = dataFrame.loc[dataFrame['Primary Type'] .isin(major_crimes)]
crime_df

loc=crime_df['Location Description'].value_counts().sort_values(ascending=False)
loc=loc[:20]
loc
# tempr=yearDF.groupby('Location Description')['ID'].count().sort_values(ascending=False)
# tempr=temp[:10]
# tempr

plt.figure(figsize=(12,8))
sns.countplot(data=crime_df, x="Location Description", order=loc.index)
plt.xticks(rotation='vertical')
plt.ylabel('Count of Crimes')
plt.show()

plt.figure(figsize=(12,8))
loc.plot(kind='bar',color='red')
plt.ylabel('Count')
plt.show()

# storing four major 4 locations
major_loc = ['STREET','RESIDENCE', 'APARTMENT','SIDEWALK']

# selecting the data form our dataset that belongs major locations
crime_df = crime_df.loc[crime_df['Location Description'] .isin(major_loc)]
crime_df

graphDF=crime_df.copy()
graphDF

graphDF['Date'] = pd.to_datetime(graphDF['Date'])
graphDF

# datetime.strptime('07/28/2014 18:54:55.099000', '%m/%d/%Y %H:%M:%S.%f')

graphDF['Date'] = pd.to_datetime(graphDF['Date'],format='%m/%d/%Y %I:%M:%S %p')

# graphDF['Date'] = datetime.strptime(graphDF['Date'],'%m/%d/%Y %H:%M:%S.%f')
# 
# '%m/%d/%Y %H:%M:%S.%f'

import calendar
graphDF['Month']=(graphDF['Date'].dt.month).apply(lambda x: calendar.month_abbr[x])
graphDF.head(4)

# storing the categories of months

graphDF['Month'] = pd.Categorical(graphDF['Month'] , categories=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], ordered=True)

import numpy as np
months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
graphDF.groupby('Month')['ID'].count().plot(marker='o')
plt.xticks(np.arange(12),months)
plt.ylabel('No of Crimes')
plt.show()

graphDF.groupby(['Year','Arrest'])['ID'].count().unstack().plot(kind='bar')
plt.ylabel('No of Crimes')
plt.show()

sns.set(rc={'figure.figsize':(12,6)})
sns.countplot(x='Primary Type',hue='Arrest',data=graphDF,order=graphDF['Primary Type'].value_counts().index)
plt.xticks(rotation='vertical')
plt.ylabel('No of Crimes')
plt.show()

temp=graphDF.copy()
temp

def crime_type(t):
    if t =='THEFT': return '0'
    elif t =='BATTERY': return '1'
    elif t =='CRIMINAL DAMAGE': return '2'
    elif t == 'ASSAULT': return '3'
    else: return '-1'

# cp_crime = crime_df.copy()
temp['crimeType'] = temp['Primary Type'].map(crime_type)
temp=temp.drop(columns='Primary Type')
temp

# temp.dropna()
temp.dropna(inplace=True)
temp

# count of null values
temp.isna().any(axis=1).sum()

# values according to their class count
count=temp['crimeType'].value_counts().sort_values(ascending=False)
count

# getting the portion of our data set for random data selection
nrows= temp.shape[0]
portion=math.floor(nrows/3)
portion

first=temp.iloc[0:portion,:]
first.shape

nextp=portion+portion+1
scnd=temp.iloc[(portion+1):nextp,:]
scnd.shape

finalp=nextp+portion+1
third=temp.iloc[(nextp+1):finalp,:]
third.shape

# picking random 5k enteries from the first part
index=np.random.choice(portion,replace=False,size = 70000)
df_frst=first.iloc[index]
df_frst.shape

# picking random 5k enteries from the second half

index=np.random.choice(portion,replace=False,size = 70000)
df_scnd=scnd.iloc[index]
df_scnd

# picking random 5k enteries from the third half

index=np.random.choice(portion,replace=False,size = 70000)
df_third=third.iloc[index]
df_third

# combined all three dataframe

temp_df = pd.concat([df_frst,df_scnd],ignore_index=True)
final_df = pd.concat([temp_df,df_third],ignore_index=True)
final_df

df=final_df.copy()

# Using PCA to combine two features

from sklearn.decomposition import PCA

location = df[['Latitude','Longitude']]
pca = PCA(n_components=1,random_state=123)
locat = pca.fit_transform(location)
df['Location'] = locat
df

# convertung date column to actual date format
df.Date=pd.to_datetime(df.Date)

df.head(10)

# extracting month and weekday from date column
df['month']=df.Date.dt.month
df['weekday'] = df.Date.dt.day_of_week
df=df.drop(columns='Date')
df

cp_crime = df.copy()

# encoding our column with labels to nos
labelEncoder = LabelEncoder()

locDes_enc = labelEncoder.fit_transform(cp_crime['Location Description'])
cp_crime['Location Description'] = locDes_enc

cp_crime.head()

# encoding our column with labels to nos
labelEncoder2 = LabelEncoder()

arrest_enc = labelEncoder2.fit_transform(cp_crime['Arrest'])
cp_crime['Arrest'] = arrest_enc

cp_crime.head()

# encoding our column with labels to nos
labelEncoder3 = LabelEncoder()

domestic_enc = labelEncoder3.fit_transform(cp_crime['Domestic'])
cp_crime['Domestic'] = domestic_enc

cp_crime.head()

# feature scaling
scaler = preprocessing.MinMaxScaler()
cp_crime[['Beat']] = scaler.fit_transform(cp_crime[['Beat']])
cp_crime[['Location Description']] = scaler.fit_transform(cp_crime[['Location Description']])
cp_crime[['X Coordinate', 'Y Coordinate']] = scaler.fit_transform(cp_crime[['X Coordinate', 'Y Coordinate']])
cp_crime

# using correlation for the feature selection
corelation = cp_crime.corr()
corelation


plt.figure(figsize=(12,8))
sns.heatmap(corelation,annot=True)

# month week day have low correlation they isn't effect our results so we drop them
# since beat have high correlation with district so we drop beat
# and X cordinate have high correlation with longitube and Y cordinate with latitude and location so we drop longitude and latitude
# 'Beat'
selected_cols=['Location Description','Arrest','Domestic','Beat','Ward','X Coordinate','Y Coordinate','Year'] 


X=cp_crime[selected_cols]
Y=cp_crime['crimeType']

# Class Balancing using OverSampling

from collections import Counter
counter=Counter(Y)

# before oversampling
print(Counter(Y))

# Total classes
np.unique(Y)

from imblearn.over_sampling import SMOTE

# oversampling using SMOTE
oversample= SMOTE()
X,Y = oversample.fit_resample(X,Y)

# After overSampling
print(Counter(Y))

Y.isna().any(axis=0).sum()

Y=Y.astype(int)
Y.dtype

for c in selected_cols:
    print(f'{c}:{len(cp_crime[c].unique())}')

    
sns.set_theme(style="whitegrid")
#dropping domestic, x coordinate and y coordinate on the basis of correlation map
selected_cols=['Location Description','Arrest','Beat','Ward','Community Area','Year','Location']      
sns.boxplot(x=cp_crime['Location Description'])
plt.show()
sns.boxplot(x=cp_crime['Beat'])
plt.show()
sns.boxplot(x=cp_crime['Ward'])
plt.show()
sns.boxplot(x=cp_crime['Community Area'])
plt.show()
sns.boxplot(x=cp_crime['Year'])
plt.show()
sns.boxplot(x=cp_crime['Location'])
plt.show()

# Tarining and testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X , Y , test_size = 0.2, random_state=0)

# Models used
# 1- Logistic Regression
# 2- Naive Bayes
# 3- XG Boost
# 4- Random Forest
# 5- Knn
# 6- SVM
# 7- Ada Boost
# 8- Decision Tree Classifier (J48)

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# para=[{"max_iter":[1,10,100,100]}]
parameters = {
    'penalty' : ['l1','l2'], 
    'C'       : np.logspace(-3,3,7),
    }


logreg = LogisticRegression(max_iter=12000,class_weight='balanced')
clf = GridSearchCV(logreg,                    # model
                   param_grid = parameters,   # hyperparameters
                   scoring='accuracy',        # metric for scoring
                   cv=3)                     # number of 

print("Tuned Hyperparameters :", clf.best_params_)
print("Accuracy :",clf.best_score_)

# Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver="sag",max_iter=12000,class_weight='balanced',penalty = 'l2',C = 1.0)
lr.fit(X_train, y_train)
pred_labels = lr.predict(X_test)
print('--------------------------------------------------------')
score = lr.score(X_test, y_test)
print('Accuracy Score: ', score)
print('--------------------------------------------------------')
print(classification_report(y_test, pred_labels))
# print('Accuracy of Logistic Regression', lr.score(X_test, y_test))

# Naive Bayes

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
pred_labels = gnb.predict(X_test)

 # Use score method to get accuracy of the model
print('--------------------------------------------------------')
score = gnb.score(X_test, y_test)
print('Accuracy Score: ', score)
print('--------------------------------------------------------')
print(classification_report(y_test, pred_labels))
# print('Accuracy of Naive Bayes', gnb.score(X_test, y_test))

#  Categoric Naivee Bayes

from sklearn.naive_bayes import CategoricalNB
cnb = CategoricalNB(min_categories=4)
cnb.fit(X_train,y_train)
pred_labels = cnb.predict(X_test)
print('--------------------------------------------------------')
score = cnb.score(X_test, y_test)
print('Accuracy Score: ', score)
print('--------------------------------------------------------')
print(classification_report(y_test, pred_labels))
# print('Accuracy of Categoric Naive Byaes', cnb.score(X_test, y_test))    


error_rate = []

krange = range(1,10,1)
for i in krange:
    knn = KNeighborsClassifier(n_neighbors=i, metric='manhattan', weights = 'uniform',n_jobs= -1) 
    knn.fit(X_train,y_train)
    predicted_train = knn.predict(X_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    print(' Neighbours  ',i)
    print('Test accuracy ', metrics.accuracy_score(y_train, predicted_train))
    print('Accuracy ', metrics.accuracy_score(y_test, pred_i))
    

# elbow method to check best amount of neighbours
plt.figure(figsize=(10,6))

plt.plot(krange,error_rate, color= 'blue', linestyle= 'dashed', marker= 'o', markerfacecolor='red', markersize=1)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
# by looking at the figure at 1 error is minimum
# after that at 2 it  increases drastically than show a gradual decrease
# so we chooose K size 1


# KNN

knn = KNeighborsClassifier(n_neighbors = 1,metric='manhattan', weights = 'uniform')
knn.fit(X_train, y_train)
pred_train = knn.predict(X_train)
pred_i = knn.predict(X_test)

 # Use score method to get accuracy of the model
print('--------------------------------------------------------')
score = knn.score(X_test, y_test)
print('Accuracy Score: ', score)
print('--------------------------------------------------------')
print(classification_report(y_test, pred_i))

print('Test accuracy ', metrics.accuracy_score(y_train, pred_train))
print('Accuracy ', metrics.accuracy_score(y_test, pred_i))

from sklearn.svm import SVC
svm = SVC(gamma='auto')

# For HyperTuning
#from sklearn.model_selection import GridSearchCV
#clf = GridSearchCV( svm ,
#{
#    'C': [1,10,20],
#    'kernel': ['rbf','linear']
#}, cv=5, return_train_score=False)

#clf.fit(X_train, y_train)

#clf.cv_results_

#dataframe = pd.DataFrame(clf.cv_results_)
#dataframe

# SVM
svm = SVC(gamma='auto',kernel='rbf',class_weight='balanced')
svm.fit(X_train, y_train)
print('Accuracy of SVM', svm.score(X_test, y_test))

pred_train = svm.predict(X_train)
pred_i = svm.predict(X_test)
print('--------------------------------------------------------')
print(' SVM ')
print('--------------------------------------------------------')
print(classification_report(y_test, pred_i))

# SVM

#from sklearn.model_selection import cross_val_score
#score=cross_val_score(svm,X_train, y_train,cv=3)
#print('Cross Validation: ',score.mean())

# for xgboost
Y=Y.map({1:0,2:1,3:2,4:3})

## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}

# Calculate the accuracy
import xgboost as xgb
xgb = xgb.XGBClassifier()
#xgb.set_params(n_estimators=10)
random_search=RandomizedSearchCV(xgb,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

random_search.fit(X_train, y_train)
# Fit it to the training set

print(random_search.best_estimator_)

random_search.best_params_

xgb=xgb.set_params(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.5,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, gamma=0.1, gpu_id=-1, grow_policy='depthwise',
              importance_type=None, interaction_constraints='',
              learning_rate=0.3, max_bin=256, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=12, max_leaves=0, min_child_weight=7,
              monotone_constraints='()', n_estimators=100,
              n_jobs=0, num_parallel_tree=1, objective='multi:softprob',
              predictor='auto', random_state=0, reg_alpha=0)
xgb.fit(X_train, y_train)

# Predict the labels of the test set
preds = xgb.predict(X_test)

accuracy = float(np.sum(preds==y_test))/y_test.shape[0]

# Print the baseline accuracy
print("XG Boost accuracy:", accuracy)

pred_train = xgb.predict(X_train)
pred_i = xgb.predict(X_test)
print('--------------------------------------------------------')
print(' Xgb Report ')
print('--------------------------------------------------------')
print(classification_report(y_test, pred_i))

# print(xgb)

y_train.unique()

# importing random forest classifier from assemble module
from sklearn.ensemble import RandomForestClassifier
# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 300) 
 
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
 
# performing predictions on the test dataset
y_pred = clf.predict(X_test)
 
# metrics are used to find accuracy or error
from sklearn import metrics 
print()
 
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

pred_train = clf.predict(X_train)
pred_i = clf.predict(X_test)
print('--------------------------------------------------------')
print(' Random Forest Report ')
print('--------------------------------------------------------')
print(classification_report(y_test, pred_i))

# Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

total_datapoints = X_test.shape[0]
mislabeled_datapoints = (y_test != y_pred).sum()
correct_datapoints = total_datapoints-mislabeled_datapoints
percent_correct = (correct_datapoints / total_datapoints) * 100

print("DecisionTreeClassifier results for NSL-KDD:\n")
print("Total datapoints: %d\nCorrect datapoints: %d\nMislabeled datapoints: %d\nPercent correct: %.2f%%"
      % (total_datapoints, correct_datapoints, mislabeled_datapoints, percent_correct))


print("ACCURACY OF THE Decision Tree Classifier: ", metrics.accuracy_score(y_test, y_pred))
pred_train = tree.predict(X_train)
pred_i = tree.predict(X_test)
print('--------------------------------------------------------')
print(classification_report(y_test, pred_i))

# Decision Tree Classifier (J48)

# from sklearn.metrics import accuracy_score


# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1000)


# j48 = DecisionTreeClassifier(criterion = "gini",random_state = 1000,max_depth=500, min_samples_leaf=600)
# j48.fit(X_train, y_train)
# print(j48)
             
# clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 1000,max_depth = 500, min_samples_leaf = 600)
# clf_entropy.fit(X_train, y_train)
# print(clf_entropy)
             
# y_pred = j48.predict(X_test)
# # print("Predicted values:")
# print(y_pred)
             
# print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
# print ("Accuracy : ",accuracy_score(y_test,y_pred))
# print("Report : ",classification_report(y_test, y_pred))


from sklearn.ensemble import AdaBoostClassifier


# Create adaboost classifer object
adb = AdaBoostClassifier(n_estimators=300,  learning_rate=1)
# Train Adaboost Classifer
model = adb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)

print("Accuracy OF Ada Boost:",metrics.accuracy_score(y_test, y_pred))
pred_train = adb.predict(X_train)
pred_i = adb.predict(X_test)
print('--------------------------------------------------------')
print(classification_report(y_test, pred_i))

# Using Cross Validation
# Models used
# 1- Logistic Regression
# 2- Naive Bayes
# 3- XG Boost
# 4- Random Forest
# 5- Knn
# 6- SVM
# 7- Ada Boost
# 8- Decision Tree Classifier (J48)

# XG Boost
from sklearn.model_selection import cross_val_score
score=cross_val_score(xgb,X_train, y_train,cv=10)
score
print('XG boost Using Cross Validation: ',score.mean())

# Logistic Regression
score=cross_val_score(lr,X_train, y_train,cv=10)
score
print('Logistic Regression boost Using Cross Validation: ',score.mean())

# Naive Bayes
score=cross_val_score(gnb,X_train, y_train,cv=10)
score
print('Naive Bayes Using Cross Validation: ',score.mean())

# KNN
score=cross_val_score(knn,X_train, y_train,cv=10)
score
print('KNN Using Cross Validation: ',score.mean())

# Random Forest
score=cross_val_score(clf,X_train, y_train,cv=10)
score
print('Random Forest Using Cross Validation: ',score.mean())

# SVM
score=cross_val_score(svm,X_train, y_train,cv=10)
score
print('Random Forest Using Cross Validation: ',score.mean())

# Decision Tree Classifier (J48)
score=cross_val_score(j48,X_train, y_train,cv=10)
score
print('J46 Using Cross Validation: ',score.mean())

# Ada Boost
score=cross_val_score(adb,X_train, y_train,cv=10)
score
print('Ada BoostUsing Cross Validation: ',score.mean())





    
    
