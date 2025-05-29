import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import os
#import IPython.display as ipd
import librosa
#import tensorflow as tf
#import tensorflow_io as tfio
#-from scipy.io import wavfile
import soundfile as sf
#from IPython.display import Audio
#import io
import random
#import torch
#import torch.nn as nn
#import torch.optim as optim
#import torchvision
#import torchvision.transforms as transforms
#import torch.nn.functional as F
import pickle as pk

# load the data its in the Train directory its not a csv its just wav files

# get every file from the Train directory
train_dir = "/home/u172702/dl_project/accent_classification/Train/"
train_files = os.listdir(train_dir)

# make csv file out of the files 
train_df = pd.DataFrame(train_files, columns=['file'])

# create a column with the file path 
train_df['file_path'] = train_df['file'].apply(lambda x: os.path.join(train_dir, x))

# create a column with the file name without the .wav extension
train_df['file_name'] = train_df['file'].str.replace('.wav', '')

# create a column with the accent from the first numer of the file name
train_df['accent'] = train_df['file_name'].str[0]

# create a column with male or female from the second incex of the file name
train_df['gender'] = train_df['file_name'].str[1]

print('train df created')

test_dir = "/home/u172702/dl_project/accent_classification/Test/"
test_files = os.listdir(test_dir)

test_df = pd.DataFrame(test_files, columns=['file'])
test_df['file_path'] = test_df['file'].apply(lambda x: os.path.join(test_dir, x))
test_df['file_name'] = test_df['file'].str.replace('.wav', '')

# create an empty column for the accent and gender
test_df['accent'] = ''
test_df['gender'] = ''
 
print('test_df created')

def load_audio_file(file_path):
    # The longest file is 286003 ms? long
    input_length = 286003
    data = librosa.core.load(file_path, sr = 16000)[0]
    # returns a numpy array which is a audio time series, in our case with 
    # only one channel.
    data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

def plot_time_series(data):
    fig = plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()

def array_to_wav_bytes(arr, rate=16000):
    buf = io.BytesIO()
    sf.write(buf, arr, rate, format='WAV')
    buf.seek(0)
    return buf.read()  # raw WAV bytes

def add_white_noise(file):
    amp = random.randrange(1, 21) * 0.005
    # amp between 0.005 and 0.1
    # creates a normally distributed array with the length of the file 
    wn = np.random.randn(len(file))
    file_wn = file + amp*wn
    if type(file_wn) != np.ndarray:
        print('problem, add_white_noise')
    return file_wn

def data_roll(file):
    rd = random.randint(1, 25)
    hz = rd * 10,000
    # range between 10,000 and 250,000
    file_roll = np.roll(file, hz)
    if type(file_roll) != np.ndarray:
        print('problem, data_roll')
    return file_roll

def stretch(file):
    rd = random.randint(1, 3)
    # range between 1 and 3 
    data_stretch = librosa.effects.time_stretch(file, rate = rd)
    if type(data_stretch) != np.ndarray:
        print('problem, data_stretch')
    return data_stretch

def augment(data):
    functions = [add_white_noise, data_roll, stretch]
    random_number = random.randint(0, 2)
    #print('augmented')
    function = functions[random_number]
    return function(data)
    #print('didn\'t augment')

def frequency_mask(data):
    spectro = data[0]
    range = random.randint(0, 30)
    band = range = random.randint(0, len(spectro))
    spectro[band: band+range] = 0
    return data

def temporal_mask(data):
    spectro = data[0]
    range = random.randint(0, 2000)
    band =  random.randint(0, 15053)
    print(band, range)
    spectro[:, band: band+range] = 0
    return data

print('functions created, skipping training data')


train_data = [] # shape is 3100 entries of [loaded file, accent]
test_data = []

for name, accent in zip(train_df['file_path'], train_df['accent']):
    scale, sr = librosa.load(name)
    train_data.append([scale, accent])

for name in test_df['file_path']:
    scale, sr = librosa.load(name)
    test_data.append(scale)

for i in range(3):
    
    for i in range(len(train_df)):
        random_number = random.randint(0, 1)
        if random_number == 0:
            augmented_data = augment(train_data[i][0])
            train_data.append([augmented_data, train_data[i][1]])

for data in train_data:
    data[0] = np.pad(data[0], (0, max(0, 286003 - len(data[0]))), "constant")

    
for data in test_data:
    data = np.pad(data, (0, max(0, 286003 - len(data))), "constant")

print('data augmented and padded, onto create spectrograms')


spectro_accent = []
spectro_accent_test = []

files = len(train_data)

for i in range(files):
    if i%20 == 0:
        print(f'we\'re at iteration {i} of the training data') 
    mel_spectrogram = librosa.feature.melspectrogram(y= train_data[i][0],sr= 16000, hop_length= 20)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    spectro_accent.append([log_mel_spectrogram, train_data[i][1]])


for i in range(len(test_data)):
    if i%20 == 0:
        print(f'we\'re at iteration {i} of the test data') 
    mel_spectrogram = librosa.feature.melspectrogram(y= test_data[i], sr= 16000, hop_length= 20)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    spectro_accent_test.append(log_mel_spectrogram)

print('spectrograms created for training set, augmenting them now')


for data in spectro_accent:
    rand = random.randint(0, 10)
    if rand < 7:
        if rand%2 == 0:
            spectro_freq = frequency_mask(data)
            spectro_freq_temp = temporal_mask(spectro_freq)
            spectro_accent.append(spectro_freq_temp)
        if rand%2 != 0 and rand > 2:
            spectro_freq = frequency_mask(data)
            spectro_accent.append(spectro_freq)
        if rand%2 != 0 and rand < 2: 
            spectro_temp = temporal_mask(data)
            spectro_accent.append(spectro_temp)


print('done augmenting spectrogram, saving file.')

with open(f'spectro_data_train_{files}_filesls.pkl', 'wb') as handle:
    pk.dump(spectro_accent, handle, protocol=pk.HIGHEST_PROTOCOL)

with open('spectro_data_test.pkl', 'wb') as handle:
    pk.dump(spectro_accent_test, handle,  protocol=pk.HIGHEST_PROTOCOL)