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
dataset.shape

# droping the null value enteries drop null 
dataset.dropna(inplace=True)
# Displaying DataSet
dataset

# Before removing Null values 1048575
# After removing Null value 1015247
# Total Null values removed 33328

# ignore latitude and logitude outside of the chicago
dataset=dataset[(dataset["Latitude"] < 45)
             & (dataset["Latitude"] > 40)
             & (dataset["Longitude"] < -85)
             & (dataset["Longitude"] > -90)]

# Displaying DataSet
dataset

# listing the crimes category wise with their counts
types=dataset['Primary Type'].value_counts().sort_values(ascending=False)
# Displaying types
types

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
crime_df

# since we dont have different crimes in early years so we drop data of these years
data = crime_df.pivot_table(index='Year', columns='Primary Type', aggfunc='count')
print(data)

# selecting the dataset which starts from 2015
crime_df=crime_df[crime_df['Year']>=2015]
# Displaying major_crimes from 2015
crime_df

temp=crime_df.copy()
temp

# getting the half of our data set for random data selection
nrows= temp.shape[0]
portion=math.floor(nrows/3)
# Displaying this portion size
portion

# First half of the data
first=temp.iloc[0:portion,:]
# Displaying the first half shape
first.shape

# Second half of the data
nextp=portion+portion+1
scnd=temp.iloc[(portion+1):nextp,:]
# Displaying the second half shape
scnd.shape

#Third half of the data
finalp=nextp+portion+1
third=temp.iloc[(nextp+1):finalp,:]
# Displaying the third half shape
third.shape

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
