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

dataset = pd.read_csv('Crimes_2001_to_Present (1).csv',low_memory=False)
dataset.head(10)
# dataset.dtypes

# droping the features that are not usefull
dataset=dataset.drop(columns=['ID','Case Number','Description','Updated On','Block'])
# X.shape
print('Columns in dataset: ', dataset.columns)
dataset.shape
