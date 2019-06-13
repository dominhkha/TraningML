# dominhkha
# import lib
import numpy as np 
import matplotlib.pyplot as plt 
import os
import sklearn
from sklearn import svm
from sklearn import datasets,metrics
from joblib import load,dump
cwd=os.getcwd()
# load data

iris=datasets.load_iris()
digits=datasets.load_digits()
images_and_label=list(zip(digits.images,digits.target))
for index,(image,label) in enumerate(images_and_label[:4]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r)
    plt.title('Training: %i' % label)

n_samples=len(digits.images)
data=digits.images.reshape((n_samples,-1))

classifier = svm.SVC(gamma=0.001)
classifier.fit(data[:n_samples//2],digits.target[:n_samples//2])
dump(classifier,"svmClf.joblib")
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
