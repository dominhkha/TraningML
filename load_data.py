import tensorflow
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import random
import matplotlib.image as mpimg
def load_training_data(root_path,num_images=-1):

    names=[d for d in os.listdir(root_path) if d=='cats'or d=='dogs']
    training_images=[]
    label_images=[]
    file_names=[]
    for d in names:
        file_names=[]
        if num_images==-1:
            file_names = [os.path.join(root_path, d, path) for path in os.listdir(os.path.join(root_path, d))
                          if path.endswith(".jpg")]
        else :
            for i,path in enumerate(os.listdir(os.path.join(root_path,d))):
                if i>(num_images)/2: break
                if path.endswith('.jpg'): file_names.append(os.path.join(root_path,d,path))
                print(i)

        print("loading...".format(color='blue'))
        for path in file_names:
            image = load_img(path,target_size=(512,512))
            image=np.array(image)
            training_images.append(image)
            if d == "cats":
                label_images.append(0)
            else:
                label_images.append(1)
        print("Done...".format(color='blue'))
    rand=list(zip(training_images,label_images))
    random.shuffle(rand)
    training_images,label_images=zip(*rand)
    label_lib={0:'cat',1:'dog'}
    return training_images,label_images,label_lib

if __name__=='__main__':

    training_images,label_images,label_lib=load_training_data("dataset/training_set/")

    print(label_images[0],label_images[500])
    #print(training_images.shape,label_images.shape)