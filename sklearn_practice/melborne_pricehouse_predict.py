import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score,mean_absolute_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


import os

cwd=os.getcwd()
path_file=os.path.join(cwd,'melb_data.csv')
data_frame=pd.read_csv(path_file)
data_frame=data_frame.dropna()
y=data_frame.Price

melbourne_features=['Rooms','Bathroom','Landsize','BuildingArea','YearBuilt']
X=data_frame[melbourne_features]

my_imputer=Imputer()
X=my_imputer.fit_transform(X)

train_X,test_X,train_y,test_y=train_test_split(X,y,random_state=0)

def one_hot_encoder(df,colums):
    for col in colums:
        if df[col].dtype==np.dtype('object'):
            dummies=pd.get_dummies(df[col],prefix=col)
            df=pd.concat([df,dummies],axis=1)

            df.drop([col],axis=1,inplace=1)
        return df
def get_cols_with_no_nans(df,col_type='num'):

    if col_type=='num':
        predictors=df.select_dtypes(exclude=['object'])
    elif col_type=='no_num':
        predictors=df.select_dtypes(include=['object'])
    elif col_type=='all':
        predictors=df
    else :
        print('Error: choose a type(num,no_num,all)')
    cols_with_no_nans=[]
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no

def getModel(train):
    model=Sequential()
    model.add(Dense(128,kernel_initializer='normal',input_dim=train.shape[1],activation='relu'))
    model.add(Dense(256,kernel_initializer='normal',activation='relu'))
    model.add(Dense(256,kernel_initializer='normal',activation='relu'))
    model.add(Dense(256,kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,  kernel_initializer='normal',activation='linear'))
    return model

model=getModel(X)
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_absolute_error'])

model.fit(train_X,train_y,epochs=1000,batch_size=32)
model.save_weights('price.h5')


