# Feature extraction from audio signal
import os
import random
import sys
import keras
import IPython.display as ipd
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import seaborn as sns
import scipy.io.wavfile
import tensorflow as tf
py.init_notebook_mode(connected=True)
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import History, ReduceLROnPlateau, CSVLogger
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, LSTM, Input, Flatten, Dropout, Activation
from keras.layers import BatchNormalization, Conv1D, MaxPooling1D, AveragePooling2D
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm
input_duration = 3

dir_list= os.listdir('C:/PAC/dissertation/SER_DATASET/')
dir_list.sort()
print(dir_list)
data_df = pd.DataFrame(columns = ['path', 'source', 'actor', 'gender', 'intensity',
‘statement', 'repetition', 'emotion'])
count = 0
for i in dir_list:
  file_list = os.listdir('C:/PAC/dissertation/SER_DATASET/' + i)
  for f in file_list:
    nm = f.split('.')[0].split('-')
    path = 'C:/PAC/dissertation/SER_DATASET/' + i + '/' + f
    source = int(nm[1])
    actor = int(nm[-1])
    emotion = int(nm[2])
  if int(actor)%2 == 0:
    gender = "female"
  else:
    gender = "male"
  if nm[3] == '01':
    intensity = 0
  else:
    intensity = 1
  if nm[4] == '01':
    statement = 0
  else:
    statement = 1
  if nm[5] == '01':
    repetition = 0
  else:
    repetition = 1
   data_df.loc[count] = [path, source, actor, gender, intensity, statement, repetition, emotion]
  count += 1
print(len(data_df))
data_df.head()
file_name = data_df.path[1001]
print(file_name)
samples, sample_rate = librosa.load(file_name)
samples, sample_rate
len(samples), sample_rate
def log_spctgrm(audio, sample_rate, window_size = 20, step_size = 10, epsilon = 1e-10):
  nperseg = int(round(window_size * sample_rate / 1e3))
  noverlap = int(round(step_size * sample_rate / 1e3))
  frequency, time, spec = signal.spectrogram(audio, fs = sample_rate, window = 'hann',
  nperseg = nperseg, noverlap = noverlap,
  detrend = False)
  return frequency, time, np.log(spec.T.astype(np.float32) + epsilon)
frequency, time, spectrogram = log_spctgrm(samples, sample_rate)
mean = np.mean(spectrogram, axis = 0)
std = np.std(spectrogram, axis = 0)
spectrogram = (spectrogram - mean) / std
aa, bb = librosa.effects.trim(samples, top_db=30)
aa, bb
S = librosa.feature.melspectrogram(aa, sr = sample_rate, n_mels = 128)
log_S = librosa.power_to_db(S, ref = np.max)
mfcc = librosa.feature.mfcc(S = log_S, n_mfcc = 13)
# All 8 Class
label8_list = []
for i in range (len(data_df)):
  if data_df.emotion[i] == 1: # Neutral
    lb = "_neutral"
  if data_df.emotion[i] == 2: # Calm
    lb = "_calm"
  elif data_df.emotion[i] == 3: # Happy
    lb = "_happy"
  elif data_df.emotion[i] == 4: # Sad
    lb = "_sad"
  elif data_df.emotion[i] == 5: # Angry
    lb = "_angry"
  elif data_df.emotion[i] == 6: # Fearful
    lb = "_fearful"
  elif data_df.emotion[i] == 7: # Disgust
    lb = "_disgust"
  elif data_df.emotion[i] == 8: # Surprised
    lb = "_surprised"
  else:
    lb = "_none"
# Add gender to the label
label8_list.append(data_df.gender[i] + lb)
len(label8_list)
data_df['label'] = label8_list
data_df.head()
print(data_df.label.value_counts().keys())
def plot_emotion_dist(dist, color_code = '#C2185B', title = "Plot"):
  tmp_df = pd.DataFrame()
  tmp_df['Emotions'] = list(dist.keys())
  tmp_df['Count'] = list(dist)
  fig, ax =plt.subplots(figsize = (12, 6))
  ax = sns.barplot(x = "Emotions", y = "Count", color = color_code, data = tmp_df )
  ax.set_title(title)
  ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
a = data_df.label.value_counts()
plot_emotion_dist(a, "#2962FF", "Emotion Distribution")
data2_df = data_df.copy()
# Male audio feature extraction
data2_df = data2_df[data2_df.label != "male_none"]
data2_df = data2_df[data2_df.label != "female_none"].reset_index(drop=True)
data2_df = data2_df[data2_df.label != "female_neutral"]
data2_df = data2_df[data2_df.label != "female_happy"]
data2_df = data2_df[data2_df.label != "female_angry"]
data2_df = data2_df[data2_df.label != "female_sad"]
data2_df = data2_df[data2_df.label != "female_fearful"]
data2_df = data2_df[data2_df.label != "female_calm"]
data2_df = data2_df[data2_df.label != "female_positive"]
data2_df = data2_df[data2_df.label != "female_negative"].reset_index(drop=True)
tmp1 = data2_df[data2_df.actor == 21]
tmp2 = data2_df[data2_df.actor == 22]
tmp3 = data2_df[data2_df.actor == 23]
tmp4 = data2_df[data2_df.actor == 24]
data3_df = pd.concat([tmp1, tmp3],ignore_index=True).reset_index(drop=True)
data2_df = data2_df[data2_df.actor != 21]
data2_df = data2_df[data2_df.actor != 22]
data2_df = data2_df[data2_df.actor != 23].reset_index(drop=True)
data2_df = data2_df[data2_df.actor != 24].reset_index(drop=True)
# # Uncomment the below section for female audio feature extraction
# data2_df = data_df.copy()
# data2_df = data2_df[data2_df.label != "male_none"]
# data2_df = data2_df[data2_df.label != "female_none"]
# data2_df = data2_df[data2_df.label != "male_neutral"]
# data2_df = data2_df[data2_df.label != "male_happy"]
# data2_df = data2_df[data2_df.label != "male_angry"]
# data2_df = data2_df[data2_df.label != "male_sad"]
# data2_df = data2_df[data2_df.label != "male_fearful"]
# data2_df = data2_df[data2_df.label != "male_calm"]
# data2_df = data2_df[data2_df.label != "male_surprised"]
# data2_df = data2_df[data2_df.label != "male_disgust"]
# tmp1 = data2_df[data2_df.actor == 22]
# tmp2 = data2_df[data2_df.actor == 24]
# data3_df = pd.concat([tmp1, tmp2],ignore_index=True).reset_index(drop=True)
# data2_df = data2_df[data2_df.actor != 22]
# data2_df = data2_df[data2_df.actor != 24].reset_index(drop=True)                                  
print (len(data2_df))
data2_df.head()
data = pd.DataFrame(columns=['feature'])
for i in tqdm(range(len(data2_df))):
  X, sample_rate = librosa.load(data2_df.path[i], res_type='kaiser_fast', duration=input_duration, sr=22050*2,offset=0.5)
  sample_rate = np.array(sample_rate)
  mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
  feature = mfccs
  data.loc[i] = [feature]
data.head()
df3 = pd.DataFrame(data['feature'].values.tolist())
labels = data2_df.label
df3.head()
newdf = pd.concat([df3,labels], axis=1)
rnewdf = newdf.rename(index=str, columns={"0": "label"})
len(rnewdf)
rnewdf.head(10)
rnewdf.isnull().sum().sum()
rnewdf = rnewdf.fillna(0)
rnewdf.head()
def plot_time_series(data):
  fig = plt.figure(figsize=(14, 8))
  plt.title('Raw wave ')
  plt.ylabel('Amplitude')
  plt.plot(np.linspace(0, 1, len(data)), data)
  plt.show()
def noise(data):
  noise_amp = 0.005*np.random.uniform()*np.amax(data)
  data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
  return data
def shift(data):
  s_range = int(np.random.uniform(low=-5, high = 5)*500)
  return np.roll(data, s_range)
def stretch(data, rate=0.8):
  data = librosa.effects.time_stretch(data, rate)
  return data
def pitch(data, sample_rate):
  bins_per_octave = 12
  pitch_pm = 2
  pitch_change = pitch_pm * 2*(np.random.uniform())
  data = librosa.effects.pitch_shift(data.astype('float64'), sample_rate,
  n_steps=pitch_change, bins_per_octave=bins_per_octave)
  return data
def dyn_change(data):
  dyn_change = np.random.uniform(low=1.5,high=3)
  return (data * dyn_change)
def speedNpitch(data):
  length_change = np.random.uniform(low=0.8, high = 1)
  speed_fac = 1.0 / length_change
  tmp = np.interp(np.arange(0,len(data),speed_fac),np.arange(0,len(data)),data)
  minlen = min(data.shape[0], tmp.shape[0])
  data *= 0
  data[0:minlen] = tmp[0:minlen]
  return data
X, sample_rate = librosa.load(data2_df.path[150], res_type='kaiser_fast',
duration=4,sr=22050*2, offset=0.5)
plot_time_series(X)
x = pitch(X, sample_rate)
plot_time_series(x)
syn_data1 = pd.DataFrame(columns=['feature', 'label'])
for i in tqdm(range(len(data2_df))):
X, sample_rate = librosa.load(data2_df.path[i], res_type='kaiser_fast',
duration=input_duration,sr=22050*2,offset=0.5)
  if data2_df.label[i]:
    X = noise(X)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    feature = mfccs
    a = random.uniform(0, 1)
    syn_data1.loc[i] = [feature, data2_df.label[i]]
    syn_data2 = pd.DataFrame(columns=['feature', 'label'])
for i in tqdm(range(len(data2_df))):
  X, sample_rate = librosa.load(data2_df.path[i], res_type='kaiser_fast',
  duration=input_duration,sr=22050*2,offset=0.5)
  if data2_df.label[i]:
    X = pitch(X, sample_rate)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    feature = mfccs
    a = random.uniform(0, 1)
    syn_data2.loc[i] = [feature, data2_df.label[i]]
len(syn_data1), len(syn_data2)
syn_data1 = syn_data1.reset_index(drop=True)
syn_data2 = syn_data2.reset_index(drop=True)
df4 = pd.DataFrame(syn_data1['feature'].values.tolist())
labels4 = syn_data1.label
syndf1 = pd.concat([df4,labels4], axis=1)
syndf1 = syndf1.rename(index=str, columns={"0": "label"})
syndf1 = syndf1.fillna(0)
len(syndf1)
syndf1.head()
df4 = pd.DataFrame(syn_data2['feature'].values.tolist())
labels4 = syn_data2.label
syndf2 = pd.concat([df4,labels4], axis=1)
syndf2 = syndf2.rename(index=str, columns={"0": "label"})
syndf2 = syndf2.fillna(0)
len(syndf2)
syndf2.head()
combined_df = pd.concat([rnewdf, syndf1, syndf2], ignore_index=True)
combined_df = combined_df.fillna(0)
combined_df.head() 
