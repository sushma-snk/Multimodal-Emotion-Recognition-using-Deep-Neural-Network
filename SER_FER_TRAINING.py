# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:05:03 2020

@author: Home-PC
"""

import numpy as np
import pandas as pd
import os

data1=pd.read_csv('C:/PAC/dissertation/SER_FER_LATEST/data6.csv')
data1=data1.sample(frac=1).reset_index(drop=True)
#print(data1)

Emotion = data1['emotion']
# the features
Features = data1.iloc[:,1:].values

## Normalization
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#data_rescaled = scaler.fit_transform(Features)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(Features, Emotion, test_size=0.2, random_state=0)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

x_traincnn = np.expand_dims(x_train, axis=2)
x_testcnn = np.expand_dims(x_test, axis=2)

# tensorflow imports
import tensorflow as tf
import keras

# tf.keras imports
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

from tensorflow.keras.layers import Dense, Conv1D, Activation, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization, Input, MaxPooling1D

model = Sequential()
  
conv1 = model.add(Conv1D(16,kernel_size=3, padding='same', activation='relu', input_shape=(x_traincnn.shape[1],1)))
max1 = model.add(MaxPooling1D(pool_size=2))

conv3 = model.add(Conv1D(32,kernel_size=3, padding='same', activation='relu'))
max2 = model.add(MaxPooling1D(pool_size=2))
drop1 = model.add(Dropout(0.5))

conv5 = model.add(Conv1D(64,kernel_size=3, padding='same', activation='relu'))
max3 = model.add(MaxPooling1D(pool_size=2))
  
conv7 = model.add(Conv1D(128,kernel_size=3, padding='same', activation='relu'))
max4 = model.add(MaxPooling1D(pool_size=2))
drop2 = model.add(Dropout(0.5))

conv9 = model.add(Conv1D(256,kernel_size=3, padding='same', activation='relu'))
max5 = model.add(MaxPooling1D(pool_size=2))

bn = model.add(BatchNormalization())
fl = model.add(Flatten())
drop3 = model.add(Dropout(0.5))

dense1 = model.add(Dense(256, activation='relu', kernel_regularizer = regularizers.l2(0.01)))
dense2 = model.add(Dense(7, activation='softmax'))

model.compile(optimizer = 'adam', loss = tf.keras.losses.categorical_crossentropy, metrics = ["accuracy"])
   
model.summary()

checkpoint_filepath = 'C:/PAC/dissertation/SER_FER_LATEST/Models'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

# Model weights are saved at the end of every epoch, if it's the best seen so far.

model_hist = model.fit(x_traincnn, y_train, epochs = 7500, batch_size = 16, validation_data=(x_testcnn,y_test),shuffle=True,callbacks=[model_checkpoint_callback]) 

model.load_weights(checkpoint_filepath)

import matplotlib.pyplot as plt

plt.plot(model_hist.history['loss'])
plt.plot(model_hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(model_hist.history['acc'])
plt.plot(model_hist.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Confusion Matrix
y_pred = model.predict_classes(x_testcnn)
y_true = np.asarray([np.argmax(i) for i in y_test])
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

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
SER_FER2_model_json = model.to_json()
with open("C:/PAC/dissertation/SER_FER_LATEST/Models/SER_FER_bestmodel.json","w") as json_file:
    json_file.write(SER_FER2_model_json)

model.save('C:/PAC/dissertation/SER_FER_LATEST/Models/SER_FER_bestweights.h5')
