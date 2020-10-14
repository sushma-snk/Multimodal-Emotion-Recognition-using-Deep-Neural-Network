# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:32:10 2020

@author: Home-PC
"""

import pandas as pd

# Describe the data
data = pd.read_csv("C:/PAC/dissertation/fer2013.csv")

# Data Sampling and Preprocessing
print(data.head())

#Sample Distribution
print('Samples distribution across Usage:')
print(data.Usage.value_counts())

#Unique Emotions Count
print('Samples per emotion:')
print(data.emotion.value_counts())

#Pixel size of each sample

print('Number of pixels for a sample:')
print(len(data.pixels[0].split(' ')))

#Categorize dataset into traing, validation and testing
train_set = data[(data.Usage == 'Training')]
validation_set = data[(data.Usage == 'PublicTest')]
test_set = data[(data.Usage == 'PrivateTest')]

#Label the emotions and store as a list
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
num_classes = len(emotion_labels)

#Converting samples into 2D-matrix
from math import sqrt

depth = 1
height = int(sqrt(len(data.pixels[0].split())))
width = height

##Visualize some data
sample_number = 76 #@param {type:"slider", min:0, max:1000, step:1}
#
#import numpy as np
#import scipy.misc
#from IPython.display import display
#
#array = np.mat(data.pixels[sample_number]).reshape(48,48)
#image = scipy.misc.toimage(array)
#display(image)
print(emotion_labels[data.emotion[sample_number]])

# Form input for the neural network considering the 3 categorized datasets
import numpy as np

X_train = np.array(list(map(str.split, train_set.pixels))).astype(np.float32)
X_validation = np.array(list(map(str.split, validation_set.pixels))).astype(np.float32)
X_test = np.array(list(map(str.split, test_set.pixels))).astype(np.float32)

num_train = X_train.shape[0]
num_validation = X_validation.shape[0]
num_test = X_test.shape[0]

X_train = X_train.reshape(num_train, width, height, depth)
X_validation = X_validation.reshape(num_test, width, height, depth)
X_test = X_test.reshape(num_test, width, height, depth)

print('Training: ',X_train.shape)
print('Validation: ',X_validation.shape)
print('Test: ',X_test.shape)

#
from keras.utils import np_utils

y_train = train_set.emotion
y_train = np_utils.to_categorical(y_train, num_classes)

y_validation = validation_set.emotion
y_validation = np_utils.to_categorical(y_validation, num_classes)

y_test = test_set.emotion
y_test = np_utils.to_categorical(y_test, num_classes)

print('Training: ',y_train.shape)
print('Validation: ',y_validation.shape)
print('Test: ',y_test.shape)

#To view the images in a given dataset
import matplotlib
import matplotlib.pyplot as plt

#def overview(start, end, X):
#    fig = plt.figure(figsize=(20,20))
#    for i in range(start, end):
#        input_img = X[i:(i+1),:,:,:]
#        ax = fig.add_subplot(10,10,i+1)
#        ax.imshow(input_img[0,:,:,0], cmap=matplotlib.cm.gray)
#        plt.xticks(np.array([]))
#        plt.yticks(np.array([]))
#        plt.tight_layout()
#    plt.show()
#overview(0,50, X_train)

#Building Neural Network
#First 4 layers: Feature extraction
#Remaining layers: Fully Connected Layers
from keras.layers import Convolution2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten, AveragePooling2D
from keras.models import Sequential

model = Sequential()

model.add(Convolution2D(64, (3, 1), padding='same', input_shape=(48,48,1)))
model.add(Convolution2D(64, (1, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
model.add(Dropout(0.25))

model.add(Convolution2D(128, (3, 1), padding='same'))
model.add(Convolution2D(128, (1, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
model.add(Dropout(0.25))

model.add(Convolution2D(256, (3, 1), padding='same'))
model.add(Convolution2D(256, (1, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
model.add(Dropout(0.25))

model.add(Convolution2D(512, (3, 1), padding='same'))
model.add(Convolution2D(512, (1, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
model.add(Dropout(0.25))
##################################################################################
model.add(Flatten())

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7))
model.add(Activation('softmax'))

#Model Summary
model.summary()

#Training the model
from keras.preprocessing.image import ImageDataGenerator 

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,   # randomly flip images
    )


datagen.fit(X_train)
datagen.fit(X_validation)

#Define the hyper parameters
batch_size = 32
num_epochs = 25

#Define fbeta metric instead of accuracy
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

filepath='C:/PAC/dissertation/FER/Model.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

from keras import backend as K

def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 1

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))

#Compile the Model
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=[fbeta, 'acc'])

#Train the model on augmented data
train_flow = datagen.flow(X_train, y_train, batch_size=batch_size)
validation_flow = datagen.flow(X_validation, y_validation)

#Training
history = model.fit_generator(train_flow,
                    steps_per_epoch=len(X_train) / batch_size,
                    epochs=num_epochs, 
                    verbose=1, 
                    validation_data=validation_flow, 
                    validation_steps=len(X_validation) / batch_size,
                    callbacks=[checkpointer, reduce_lr, checkpointer])

#Model Evaluation on test dataset
#score = model.evaluate(X_test, y_test, steps=len(X_test) / batch_size)
#print('Evaluation loss: ', score[0])
#print('Evaluation accuracy: ', score[1])

#Plots of training and validation sets (accuracy and loss)
# summarize history for accuracy

plt.plot(history.history['acc'], color='b', label='Training')
plt.plot(history.history['val_acc'], color='g', label='Validation')
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'], color='b', label='Training')
plt.plot(history.history['val_loss'], color='g', label='Validation')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='lower left')
plt.show()

#Confusion Matrix
y_pred = model.predict_classes(X_test)
y_true = np.asarray([np.argmax(i) for i in y_test])

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
cm_normalised = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.set(font_scale=1.5) 
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(cm_normalised, annot=True, linewidths=0, square=False, 
                    cmap="Blues", yticklabels=emotion_labels, xticklabels=emotion_labels, vmin=0, vmax=np.max(cm_normalised), 
                    fmt=".2f", annot_kws={"size": 20})
ax.set(xlabel='Predicted label', ylabel='True label')

#Saving the model
FER1_model_json = model.to_json()
with open("C:/PAC/dissertation/FER/FER1_model.json","w") as json_file:
     json_file.write(FER1_model_json)

model.save('C:/PAC/dissertation/FER/FER1_weights.h5')











