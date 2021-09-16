# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:45:30 2021

@author: KAgarwal
"""
import os.path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import seaborn as sns
import librosa
import utils
from sklearn.preprocessing import LabelEncoder

# Read the metadata
AUDIO_DIR = os.path.abspath("../fma_large")
tracks = utils.load('../fma_metadata/tracks.csv')
print(tracks.shape)

# Get the file path to a track
filename = utils.get_audio_path(AUDIO_DIR, 2)
print('File: {}'.format(filename))

# Load the mp3 track into a numpy array
x, sr = librosa.load(filename,sr = None, mono=True, duration = 29.5)
print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))
# Set parameters for the mel spectrogram so that
# the resulting spectrogram is of size 128 x 128
hop_length = int(sr * 29.5/128) + 1
nfft = hop_length*2
mel = librosa.feature.melspectrogram(x, sr = sr, 
                                     n_fft=nfft,
                                     hop_length=hop_length,
                                     fmax = 11025)
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(mel)
im = ax.pcolormesh(S_dB, shading = 'gouraud')
ax.set_xlabel('Frames')
ax.set_ylabel('Mel Frequency')
ax.set_title('MelSpectrogram')
fig.colorbar(im, ax=ax, format='%+2.0f dB')
                                     
mean_duration = tracks['track', 'duration'].mean()
# Get training, validation and testing subsets
train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'

print(np.sum(train))
print(np.sum(tracks.loc[train, ('track', 'duration')] >29))
print(np.sum(tracks.loc[train, ('track', 'genre_top')].notna()))

# Get the opt genres
genres = list(LabelEncoder().fit(tracks['track', 'genre_top']).classes_)
enc = LabelEncoder()
labels = tracks['track', 'genre_top']
y_train = enc.fit_transform(labels[train])
y_val = enc.transform(labels[val])
y_test = enc.transform(labels[test])