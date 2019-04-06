import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization,LeakyReLU
import numpy as np
import matplotlib.pyplot as plt

from load_data import *

# Do minh kha
# In this problem I suppose the dict {0:'cat',1:'dog'}

root_path='dataset/training_set'
num_iter=10
epochs=10
batch_size=128
target_size=(64,64)
image_size=(64,64,3)
#traing_images,label_images,label_lib=load_training_data(root_path)

# build model
# model has 13 layers
print("Configuring model ....")
model = tf.keras.models.Sequential([
    # block 1
    tf.keras.layers.Conv2D(8,kernel_size=(3,3),input_shape=(128,128,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # block 2
    tf.keras.layers.Conv2D(16,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # block 3
    tf.keras.layers.Conv2D(16,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # block 4
    tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # block 5
    tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Dense
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512,activation='relu',use_bias=True),
    # 2 output [1,0] or [0,1] approximately
    tf.keras.layers.Dense(2,activation=tf.nn.softmax)
])
model.summary()
model.compile(loss="sparse_categorical_crossentropy",optimizer='Adam',metrics=['acc'])


