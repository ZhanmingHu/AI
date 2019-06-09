#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:03:28 2019

@author: zhanminghu
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn import preprocessing
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import  KerasRegressor
from sklearn.model_selection import GridSearchCV
import datetime
from keras.constraints import maxnorm
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
pd.options.display.max_columns = 999 #to visualize the whole grid
#read in the csv file into a dataframe
dataframe = pd.read_csv("BlackFriday.csv",
                   delimiter = ",", header =0)
#,error_bad_lines=False

print(dataframe.count()) #count the number of rows for each row
dataframe.head()
dataframe.info()
dataframe.describe() #gives statistical value for each column

#check for duplicates:
idsUnique = len(set(dataframe.User_ID))
idsTotal = dataframe.shape[0]
idsDupli = idsTotal - idsUnique
print('There are ' + str(idsDupli) + ' duplicate IDs for ' + str(idsTotal) + ' total entries')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Data Visualization
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,7))
sns.distplot(dataframe.Purchase, bins = 25)
plt.xlabel('Amount spent in Purchase')
plt.ylabel('Number of Buyers')
plt.title('Purchase amount Distribution')

print ('Skew is:', dataframe.Purchase.skew())
print('Kurtosis: %f' % dataframe.Purchase.kurt())

#check which features are numeric
numeric_features = dataframe.select_dtypes(include=[np.number])
numeric_features.dtypes
#ds = pd.Series({"Column" : dataframe})
#create plots for each predictor to assess its distribution
sns.set(style="darkgrid")
ax = sns.countplot(dataframe.Occupation)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize =7)
# kind = “bar”, “strip”, “swarm”, “box”, “violin”, or “boxen”.
sns.countplot(dataframe.Occupation, hue= dataframe.Gender) #hue determine how data is plotted
sns.countplot(dataframe.Marital_Status)
sns.countplot(dataframe.Gender)
sns.countplot(dataframe.Product_Category_1, orient = 'h')
sns.countplot(dataframe.Product_Category_2, orient = 'h')
sns.countplot(dataframe.Product_Category_3, orient = 'h')
sns.countplot(dataframe.Age)
sns.countplot(dataframe.Stay_In_Current_City_Years)
sns.countplot(dataframe.City_Category)

#Correlation between Numerical Predictors and Target variable
corr = numeric_features.corr()
print (corr['Purchase'].sort_values(ascending=False)[:10], '\n')
print (corr['Purchase'].sort_values(ascending=False)[-10:])

f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corr, vmax=.8,annot_kws={'size': 20}, annot=True)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Data Preprocessing and Feature Engineering Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#at first, we are checking the percentage of missing value for each cloumns
dataframe.isnull().sum()/dataframe.shape[0]*100
dataframe = dataframe.drop(['User_ID','Product_ID'],axis = 1) #drop the first two columns since they are irrelavant
dataframe.shape
#dataframe['Product_Category_2'] == np.nan

#fill NaN with 0 instead
dataframe['Product_Category_2']= \
dataframe['Product_Category_2'].fillna(0.0).astype("float")
dataframe['Product_Category_3']= \
dataframe['Product_Category_3'].fillna(0.0).astype("float")
#dataframe = dataframe.replace(np.nan,0)
dataframe.Product_Category_2.value_counts().sort_index()
dataframe.Product_Category_3.value_counts().sort_index()
#get the index of all columns equal to 19,20
condition = dataframe.index[(dataframe.Product_Category_1.isin([19,20]))]
data = dataframe.drop(condition)
data.Product_Category_1.value_counts().sort_index()

data.apply(lambda x: len(x.unique()))

#feature engineering
#Turn gender into binary 
gender_dict = {'F':0, 'M':1}
data['Gender'] = data['Gender'].apply(lambda line: gender_dict[line])
data['Gender'].value_counts()

#convert age to numeric values
age_dict = {'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6}
data['Age'] = data['Age'].apply(lambda line: age_dict[line])

#conver city_category to binary
city_dict = {'A':0, 'B':1, 'C':2}
data['City_Category'] = data['City_Category'].apply(lambda line: city_dict[line])

#create dummmy variables for Stay_In_Current_City_Years
le = LabelEncoder()
data['Stay_In_Current_City_Years'] = le.fit_transform(data['Stay_In_Current_City_Years'])
data = pd.get_dummies(data, columns=['Stay_In_Current_City_Years'])
data.dtypes

#we are takking 5000 sample after we shuffle the orignal data np array
dataset = data.values
np.random.shuffle(dataset)
dataset = dataset[:5000]
dataset.shape
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Data Partition Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#(data_train, data_test) = model_selection.train_test_split(dataset, train_size = 0.80, random_state = 7)
x = dataset[:,[0,1,2,3,4,5,6,7,9,10,11,12,13]]
y = dataset[:,8]
x_stand = preprocessing.StandardScaler() #standardize
y_minmax = preprocessing.MinMaxScaler()

x = np.array(x).reshape((len(x),13))
y = np.array(y).reshape((len(y),1))

x = x_stand.fit_transform(x)
y = y_minmax.fit_transform(y)

#split the data into training and testing
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, train_size = 0.80, random_state = 7)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#init-mode optimization
start_time = datetime.datetime.now()

# Function to create model, required for KerasClassifier
def create_model(init_mode='uniform'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=13, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = 'adam'
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
    return model
	
# create model
model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=10, verbose=0)

#different type of initializer mode to choose for grid search
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#the parameter has to be dictionary for Grid search
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, verbose=2)

#fit the model onto 
grid_result = grid.fit(x_test, y_test)

print("The best loss is: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)


start_time = datetime.datetime.now()
#Nueron optimization
def create_model(neurons=1):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=13, kernel_initializer='uniform', activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dense(1,activation='sigmoid'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer= 'Adamax', metrics=['mae'])
    return model
	
# create model
model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons)
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# n_jobs=-1 means using all processors (don't use).

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, verbose=1)

grid_result = grid.fit(x_train, y_train)

# print out results
print("The best loss is: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Model Selection Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
neurons = [1, 5, 10, 15, 20, 25, 30]
def create_model(neurons =1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=13, kernel_initializer='uniform', activation='linear'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = 'adam'
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse','mae'])
    return model
collection_mse = []
collection_mae = []
for i in neurons:
    model = create_model(neurons =i)
    estimator = model.fit(x_train, y_train,epochs=100,verbose=2)
    score= model.evaluate(x_test, y_test, batch_size =20, verbose =1)
    print("\nTest score:", score[0])
    print('Test accuracy:', score[1])
    pyplot.plot(estimator.history['mean_squared_error'])
    pyplot.plot(estimator.history['mean_absolute_error'])
    pyplot.title('Combined statistics with '+str(i)+' neaurons - MSE & MAE')
    pyplot.xlabel('Number of epoch')
    pyplot.show()
    



#test_mse_score, test_mae_score = model.evaluate(x_test, y_test, batch_size =20, verbose =1)
#    print(test_mse_score)
#    collection_mse.append(test_mse_score)
#    collection_mae.append(test_mae_score)

































