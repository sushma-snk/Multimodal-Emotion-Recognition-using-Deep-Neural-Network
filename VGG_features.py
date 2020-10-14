# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 21:55:52 2020

@author: Home-PC
"""

from keras.layers import *
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from scipy.misc import imsave
import  numpy  as  np
from keras.layers import *
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D, Conv2D, AveragePooling2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(530, 700, 3), padding='VALID'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='VALID'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Block 2
model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
model.add(AveragePooling2D(pool_size=(19, 19)))

# set of FC => RELU layers
model.add(Flatten())

#getting the summary of the model (architecture)
model.summary()

img_path = 'C:/PAC/dissertation/fer2013/FER_images/0.jpg'
img = image.load_img(img_path, target_size=(530, 700))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

vgg_feature = model.predict(img_data)
#print the shape of the output (so from your architecture is clear will be (1, 128))
#print shape
print(vgg_feature.shape)

#print the numpy array output flatten layer
print(vgg_feature.shape)