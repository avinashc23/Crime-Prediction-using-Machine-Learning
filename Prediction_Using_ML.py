#importing libraries
import numpy as np
import pandas as pd
import math
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
# dataset.dtypes

# droping the features that are not usefull
dataset=dataset.drop(columns=['ID','Case Number','Description','Updated On','Block'])
#for checking the shape of dataset
# X.shape
print('Columns in dataset: ', dataset.columns)
#Checking the shape of dataset
print('Displaying the shape of dataset',dataset.shape)

# droping the null value enteries drop null 
dataset.dropna(inplace=True)
# Displaying DataSet
print('Displaying DataSet after removing null values',dataset)

# Before removing Null values 1048575
# After removing Null value 1015247
# Total Null values removed 33328

# ignore latitude and logitude outside of the chicago
dataset=dataset[(dataset["Latitude"] < 45)
             & (dataset["Latitude"] > 40)
             & (dataset["Longitude"] < -85)
             & (dataset["Longitude"] > -90)]

# Displaying DataSet
print('Displaying DataSet',dataset)

# listing the crimes category wise with their counts
types=dataset['Primary Type'].value_counts().sort_values(ascending=False)
# Displaying types
print('Displaying types',types)

# crime types according to their counts in dataframe
# 15 classes
# major_crimes=['THEFT','BATTERY','CRIMINAL DAMAGE','ASSAULT','OTHER OFFENSE','DECEPTIVE PRACTICE','NARCOTICS',
# 'BURGLARY','MOTOR VEHICLE THEFT','ROBBERY','CRIMINAL TRESPASS','WEAPONS VIOLATION','OFFENSE INVOLVING CHILDREN',
# 'PUBLIC PEACE VIOLATION','CRIM SEXUAL ASSAULT']
# 8 classes
# storing major crime types according to their counts in dataframe
# major_crimes=['THEFT','BATTERY','CRIMINAL DAMAGE','ASSAULT','OTHER OFFENSE','DECEPTIVE PRACTICE','NARCOTICS','BURGLARY']

# major crime time
#---> Storing Major Crimes
major_crimes=['THEFT','BATTERY','CRIMINAL DAMAGE','ASSAULT']

# Displaying major_crimes
crime_df = dataset.loc[dataset['Primary Type'] .isin(major_crimes)]
print('Displaying major_crimes',crime_df)

# since we dont have different crimes in early years so we drop data of these years
data = crime_df.pivot_table(index='Year', columns='Primary Type', aggfunc='count')
print(data)

# selecting the dataset which starts from 2015
crime_df=crime_df[crime_df['Year']>=2015]
# Displaying major_crimes from 2015
print('Displaying major_crimes from 2015',crime_df)

temp=crime_df.copy()
temp

# getting the half of our data set for random data selection
nrows= temp.shape[0]
portion=math.floor(nrows/3)
# Displaying this portion size
print('Displaying this portion size',portion)

# First half of the data
first=temp.iloc[0:portion,:]
# Displaying the first half shape
print('Displaying the first half shape',first.shape)

# Second half of the data
nextp=portion+portion+1
scnd=temp.iloc[(portion+1):nextp,:]
# Displaying the second half shape
print('Displaying the second half shape',scnd.shape)

#Third half of the data
finalp=nextp+portion+1
third=temp.iloc[(nextp+1):finalp,:]
# Displaying the third half shape
print('Displaying the third half shape',third.shape)

# picking random 80k enteries from the first half
index=np.random.choice(portion,replace=False,size = 80000)
df_frst=first.iloc[index]
# displaying the first patch shape
print('Displaying the first patch shape',df_frst.shape)

# Drawing the boxplot to check outlying values
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x=df_frst["Ward"])

# picking random 80k enteries from the second half
index=np.random.choice(portion,replace=False,size = 80000)
df_scnd=scnd.iloc[index]
# displaying the second patch
print('Displaying the second patch',df_scnd)

# picking random 80k enteries from the third half
index=np.random.choice(portion,replace=False,size = 80000)
df_third=third.iloc[index]
# displaying the third patch
print('Displaying the third patch',df_third)

# combined all three dataframe
temp_df = pd.concat([df_frst,df_scnd],ignore_index=True)
final_df = pd.concat([temp_df,df_third],ignore_index=True)
# Displaying the final dataframe
print('Displaying the final dataframe',final_df)

df=final_df.copy()

# Using PCA to combine two features
from sklearn.decomposition import PCA
location = df[['Latitude','Longitude']]
pca = PCA(n_components=1,random_state=123)
locat = pca.fit_transform(location)
df['Location'] = locat
# Displaying the dataframe
print('Displaying the dataframe', df)

# convertung date column to actual date format
df.Date=pd.to_datetime(df.Date)
# Displaying the first 10 columns
print('Displaying the first 10 columns',df.head(10))

# extracting month and weekday from date column
df['month']=df.Date.dt.month
df['weekday'] = df.Date.dt.day_of_week
df=df.drop(columns='Date')
# Displaying the dataframe
print('Displaying the dataframe',df)

# assigning crimetype 
def crime_type(t):
    if t =='THEFT': return '1'
    elif t =='BATTERY': return '2'
    elif t =='CRIMINAL DAMAGE': return '3'
    elif t == 'ASSAULT': return '4'
    else: return '0'

cp_crime = df.copy()
cp_crime['crimeType'] = cp_crime['Primary Type'].map(crime_type)
# Displaying the Cime
print('Displaying the Cime',cp_crime)

# Doing labelEncode on the Location column
labelEncoder = LabelEncoder()
locDes_enc = labelEncoder.fit_transform(cp_crime['Location Description'])
cp_crime['Location Description'] = locDes_enc
# Displaying the Cime after labelEncoding
print('Displaying the Cime after labelEncoding location column',cp_crime.head())

# Doing labelEncode on the Arrest column
labelEncoder2 = LabelEncoder()
arrest_enc = labelEncoder2.fit_transform(cp_crime['Arrest'])
cp_crime['Arrest'] = arrest_enc
# Displaying the Cime after labelEncoding
print('Displaying the Cime after labelEncoding arrest column ',cp_crime.head())

# Doing labelEncode on the Domestic column
labelEncoder3 = LabelEncoder()
domestic_enc = labelEncoder3.fit_transform(cp_crime['Domestic'])
cp_crime['Domestic'] = domestic_enc
# Displaying the Cime after labelEncoding
print('Displaying the Cime after labelEncoding domestic column',cp_crime.head())

# feature scaling
scaler = preprocessing.MinMaxScaler()
cp_crime[['Beat']] = scaler.fit_transform(cp_crime[['Beat']])
cp_crime[['X Coordinate', 'Y Coordinate']] = scaler.fit_transform(cp_crime[['X Coordinate', 'Y Coordinate']])
# Displaying the Cime after feature scaling
print('Displaying the Cime after feature scaling',cp_crime)

# using correlation for the feature selection
corelation = cp_crime.corr()
# Displaying the corelation
print('Displaying the corelation',corelation)

# Displaying the corelation graph
plt.figure(figsize=(10,7))
sns.heatmap(corelation,annot=True)

# month week day have low correlation they isn't effect our results so we drop them
# since beat have high correlation with district so we drop beat
# and X cordinate have high correlation with longitube and Y cordinate with latitude and location so we drop longitude and latitude
selected_cols=['Location Description','Arrest','Domestic','Beat','Ward','Community Area','Year','X Coordinate','Y Coordinate','Location']

X=cp_crime[selected_cols]
Y=cp_crime['crimeType']

Y=Y.astype(int)
Y.dtype

for c in selected_cols:
    print(f'{c}:{len(cp_crime[c].unique())}')


# Displaying the boxplot to check outlying values
sns.set_theme(style="whitegrid")
selected_cols=['Location Description','Arrest','Domestic','Beat','Ward','Community Area','Year','X Coordinate','Y Coordinate','Location']      
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

#for xg boost
Y=Y.map({1:0,2:1,3:2,4:3})

# Tarining and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

# Models used
# 1- Logistic Regression
# 2- Naive Bayes
# 3- XG Boost
# 4- Random Forest
# 5- Knn
# 6- SVM
# 7- Ada Boost
# 8- Decision Tree Classifier (J48)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="saga", multi_class='ovr',max_iter=12000)
lr.fit(X_train, y_train)
print('Accuracy of Logistic Regression', lr.score(X_test, y_test))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
print('Accuracy of Naive Bayes', gnb.score(X_test, y_test))

# KNN
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print('Accuracy of KNN', knn.score(X_test, y_test))
pred_train = knn.predict(X_train)
pred_i = knn.predict(X_test)
print('Test accuracy ', metrics.accuracy_score(y_train, pred_train))
print('Accuracy ', metrics.accuracy_score(y_test, pred_i))

# Xgboost
# Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# Hyper Parameter Optimization
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
# Fit it to the training set
random_search.fit(X_train, y_train)
print('Displaying the results',random_search.best_estimator_)
# Displaying random search best params
print('Displaying random search best params',random_search.best_params_)

xgb=xgb.set_params(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.5,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, gamma=0.1, gpu_id=-1, grow_policy='depthwise',
              importance_type=None, interaction_constraints='',
              learning_rate=0.15, max_bin=256, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=12, max_leaves=0, min_child_weight=5,
              monotone_constraints='()', n_estimators=100,
              n_jobs=0, num_parallel_tree=1, objective='multi:softprob',
              predictor='auto', random_state=0, reg_alpha=0)
xgb.fit(X_train, y_train)

# Predict the labels of the test set
preds = xgb.predict(X_test)
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]

# Print the baseline accuracy
print("XGboost accuracy:", accuracy)

y_train.unique()

# RandomForestClassifier
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

# SVM
from sklearn.svm import SVC
svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
print('Accuracy of SVM', svm.score(X_test, y_test))

# Decision Tree Classifier (J48)
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1000)
j48 = DecisionTreeClassifier(criterion = "gini",random_state = 1000,max_depth=500, min_samples_leaf=600)
j48.fit(X_train, y_train)
print(j48)             
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 1000,max_depth = 500, min_samples_leaf = 600)
clf_entropy.fit(X_train, y_train)
print(clf_entropy)
             
y_pred = j48.predict(X_test)
# print("Predicted values:")
# print(y_pred)
# print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
print ("Accuracy : ",accuracy_score(y_test,y_pred))
# print("Report : ",classification_report(y_test, y_pred))

# AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
# Create adaboost classifer object
adb = AdaBoostClassifier(n_estimators=300,  learning_rate=1)
# Train Adaboost Classifer
model = adb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

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








































































