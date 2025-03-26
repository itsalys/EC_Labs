#%% Import the required libraries

import numpy as np
import matplotlib.pyplot as plt

import librosa # Used for speech feature extraction: https://librosa.org/doc/

y, sr = librosa.load("test.wav", sr=None) #Save the microphone recording as test.wav 

#%% Compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))

#%% Plot the time series and the frequency-time plot (spectrogram)
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))
ax1.plot(y)
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
img = librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                          y_axis='log', x_axis='time', sr=sr, ax=ax2)
fig.colorbar(img, ax=ax2)
ax1.set(title='Time Series')
ax2.set(title='Spectrogram')
plt.show()

#%% Chroma Estimation
S = np.abs(librosa.stft(y, n_fft=4096))**2
chroma = librosa.feature.chroma_stft(S=S, sr=sr)
fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                               y_axis='log', x_axis='time', ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].label_outer()
img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax1.set(title='Power Spectrogram')
ax2.set(title='Chromogram')
plt.show()

#%% Compute Mel-Spectrogram
S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                    fmax=8000)
fig, ax = plt.subplots()
S_mel_dB = librosa.power_to_db(S_mel, ref=np.max)
img = librosa.display.specshow(S_mel_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
plt.show()

#%% Compute MFCC
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                   fmax=8000)
fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel spectrogram')
ax[0].label_outer()
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')
plt.show()