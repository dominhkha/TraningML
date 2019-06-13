import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import keras
import os
from keras.preprocessing import image
from keras.applications import MobileNet
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

cwd=os.getcwd()
rootFile=os.path.join(cwd,"dataset/training_set")
def getmodel():
    baseModel=MobileNet(weights="imagenet",include_top=False)
    x=baseModel.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation="relu")(x)
    x=Dense(512,activation='relu')(x)
    x=Dense(2,activation='softmax')(x)

    model=Model(inputs=baseModel.input,outputs=x)
    model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=['acc'])
    for layer in model.layers[:20]:
        layer.trainable=False
    return model
if __name__=="__main__":

    model=getmodel()
    trainData=ImageDataGenerator(rescale=1/255)
    train_generator=trainData.flow_from_directory(rootFile,target_size=(224,224),color_mode='rgb',batch_size=32,
                                                class_mode="categorical",shuffle=True)
    history = model.fit_generator(
      train_generator,
      steps_per_epoch=20,  
      epochs=4,
      verbose=1)
    model.save("weight.h5")