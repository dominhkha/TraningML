#dominhkha

# import library
import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt 
import os

# implementation
cwd=os.getcwd()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def get_data_set(type_of_data="fashion_mnist"):
    if type_of_data=="fashion_mnist":
        (train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
        return train_images,train_labels,test_images,test_labels



def show_top_images(images,labels,num_of_images=25):
    plt.figure(figsize=(10,10))
    for i in range(num_of_images):
        plt.subplot(np.sqrt(num_of_images),np.sqrt(num_of_images),i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i],cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()

def load_model1(image_size=(28,28)):
    model=keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(units=128,activation=tf.nn.relu))
    model.add(keras.layers.Dense(10,activation=tf.nn.softmax))
    return model



if __name__=="__main__":
    
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images=train_images/255.0
    test_images=test_images/255.0
    image_size=train_images.shape[1:]
    '''
    model=load_model1(image_size=image_size)
    model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['acc'])
    model.fit(train_images,train_labels,epochs=5)
    model.save("weightForFashionMnist.h5")
    del model
    '''
    model=load_model(os.path.join(cwd,"weightForFashionMnist.h5"))
    predictions=model.predict(test_images)
    i=0
    print(np.argmax(predictions[i]))
    print(predictions[i])
    print(test_labels[i])
    




