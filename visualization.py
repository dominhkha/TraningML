import numpy as np 
import matplotlib.pyplot as plt
import os
from load_file1 import *
cwd=os.getcwd()

def pre_visual_image(num_image,training_images,label_images,dict={0:'cat',1:'dog'}):
    values=list(dict.values())
    plt.figure(figsize=(10,10))
    training_images=np.array(training_images)
    for i in range(num_image*num_image):
        plt.subplot(num_image,num_image,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(training_images[i])
        plt.xlabel(values[label_images[i]])
    plt.show()

def plot_image_result(i,predictions_array,true_labels,img,dict={0:'cat',1:'dog'}):
    values=dict.values()
    predictions_array,true_labels,img=predictions_array[i],true_labels[i],img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    predicted_label=np.argmax(predictions_array)
    if predictions_array==true_labels:
        color='blue'
    else: color='red'
    plt.xlabel('{} {:2.0f%} {}'.format(values[predictions_array],100%np.max(predictions_array),values[true_labels],color=color))

def plot_image_array(i,predictions_array,true_label,dict={0:'cat',1:'dog'}):
    values=dict.values()
    predictions_array,true_label=predictions_array[i],true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.bar(range(10),predictions_array,color='#777777') # gray color
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')