

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
digits=datasets.load_digits()
images,labels=digits.data,digits.target

pca=PCA(n_components=2)
reduce_data=pca.fit_transform(digits.data)

data=scale(digits.data)
X_train,X_test,y_train,y_test,images_train,images_test=train_test_split(data,digits.target,digits.images,test_size=0.25,random_state=42)
n_samples,n_features=X_train.shape
n_digits=len(np.unique(y_train))
clf=KMeans(init='k-means++',n_clusters=10,random_state=42)
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)[3]
print(prediction,y_test[3])

