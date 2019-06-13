
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras import datasets
from keras.models import Sequential,load_model
from keras.layers import Conv2D,MaxPool2D,AveragePooling2D,Dense,Flatten,Dropout
from keras.preprocessing.image import load_img
from keras.optimizers import Adadelta,Adam,SGD

cwd=os.getcwd()

(train_images,train_labels),(test_images,test_labels)=datasets.fashion_mnist.load_data()
image_size=train_images[0].shape

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images= train_images.astype('float32')
test_images=test_images.astype('float32')
train_images/=255
test_images/=255
def getModel():
    model=Sequential()
    model.add(Conv2D(64,(3,3),input_shape=(28,28,1),activation='relu'))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['acc'])
    return model

def plot_result(i,predictions,truelabels,imgs):
    prediction,truelabel,img=predictions[i],truelabels[i],imgs[i]
    plt.xticks([])
    plt.yticks([])
    predicted_label=np.argmax(prediction)
    plt.imshow(img)
    if predicted_label==truelabel:
        color='blue'
    else: color='red'
    plt.xlabel("{} {:2.0f}% {}".format(class_names[predicted_label],100*float(np.max(prediction)),class_names[truelabel]),color=color)


if __name__=="__main__":
    model=getModel()
    if os.path.exists("myWeight.h5")==False:
        model.fit(train_images.reshape(-1,28,28,1),train_labels,epochs=10)
        model.save("myWeight.h5")
    else: model=load_model("myWeight.h5")
    predictions=model.predict(test_images.reshape(-1,28,28,1))
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plot_result(i,predictions,test_labels,test_images)
    plt.show()






