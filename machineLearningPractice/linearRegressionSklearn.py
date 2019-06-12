#dominhkha
import sklearn 
from  sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import numpy as np 
import os

cwd =os.getcwd()

def get_training_data(numOfFeature=2,W=[1,1,1],numOfSample=10):
    np.random.seed(1)
    X=np.random.rand(numOfSample,numOfFeature)
    X=1+X*(10)
    X1=np.insert(X,numOfFeature,1,axis=1)
    y=np.dot(X1,W)
    return X,y

def get_model(model="LinearRegression"):
    if model=="LinearRegression":
        model=LinearRegression(normalize=True,copy_X=True)
        return model


X,y=get_training_data(numOfFeature=3,W=[1,2,3,4],numOfSample=20)
model=get_model(model="LinearRegression").fit(X,y)

# model.coef_ : return W
# model.intercept_ : return bias
print(model.coef_)
print(model.intercept_)
print(model.score(X,y))