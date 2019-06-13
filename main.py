import numpy as np 
import keras
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
from load_data import *
from plot_them import *
from usingMobilenet import *
import os


cwd=os.getcwd()
if __name__=="__main__":
    model=getmodel()
    model=load_model("weight.h5")
    test_images,label_images,_dict=load_training_data(os.path.join(cwd,"dataset/test_set"),30)
    testing_images=[]
    for i,_ in enumerate(test_images):
        testing_images.append(img_to_array(test_images[i]))
    testing_images=np.array(testing_images)
    predictions=model.predict(testing_images)
    plot_image_result(29,predictions,label_images,test_images)
    plt.show()  