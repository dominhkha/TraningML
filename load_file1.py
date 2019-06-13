import os
import numpy as np 
import matplotlib.pyplot as plt 
from keras.preprocessing.image import load_img,img_to_array
import random
# load dataset 0: cat, 1: dog
cwd=os.getcwd()
root=os.path.join("dataset")

def load_data(pathFile,num_images=-1):
    ''' load training || test || valid data for visual data'''
    folderNames=[d for d in os.listdir(pathFile)]
    training_images=[]
    label_images=[]
    fileNames=[]

    for name in folderNames:
        print("loading....{}".format(name))
        fileName=[]
        if num_images==-1: 
            fileName=[os.path.join(pathFile,name,f) for f in os.listdir(os.path.join(pathFile,name)) if f.endswith(".jpg")]
        else:
            for i,f in enumerate(os.listdir(os.path.join(pathFile,name))):
                if i>num_images/2: break
                if f.endswith('.jpg'): fileName.append(os.path.join(pathFile,name,f))
        
        for file in fileName:
            image=load_img(file,target_size=(512,512))
            image=np.array(image)
            training_images.append(image)
            if name == "cats":
                label_images.append(0)
            else:
                label_images.append(1)
        print("{} Done".format(name))
    
    rand=list(zip(training_images,label_images))
    random.shuffle(rand)
    training_images,label_images=zip(*rand)
    _dict={0:"cat",1:"dog"}
    return training_images,label_images,_dict       

print(load_data.__doc__)



