import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

dataset = pd.read_csv('Crimes_2001_to_Present (1).csv',low_memory=False)
 

# droping the features that are not usefull
dataset=dataset.drop(columns=['ID','Case Number','Description','Updated On','IUCR','Block','FBI Code'])
# X.shape
#

print('Columns in dataset: ', dataset.columns)

# dataset.shape

# dataset.dtypes

# droping the null value enteries
dataset.dropna(inplace=True)
dataset

# listing the top 10 crimes category wise
dataset['Primary Type'].value_counts().sort_values(ascending=False).head(10)

# choosing the interested crimes
interested_crimes = ['THEFT','BATTERY','CRIMINAL DAMAGE','ASSAULT']


crime_rec = dataset.loc[dataset['Primary Type'] .isin(interested_crimes)]
crime_rec

def crime_type(t):
    if t =='THEFT': return '1'
    elif t =='BATTERY': return '2'
    elif t =='CRIMINAL DAMAGE': return '3'
    elif t == 'ASSAULT': return '4'
    # elif t == 'DECEPTIVE PRACTICE': return '5'
    # elif t == 'NARCOTICS': return '6'
    # elif t == 'BURGLARY': return '7'
    # elif t == 'ROBBERY': return '8'
    else: return '0'

cp_crime = crime_rec.copy()
cp_crime['crimeType'] = cp_crime['Primary Type'].map(crime_type)
cp_crime

