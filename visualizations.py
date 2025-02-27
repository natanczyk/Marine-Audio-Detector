# %% [markdown]
# 

# %% [markdown]
# Waveplot and mel-spectrogram of a random killer whale sample in the dataset

# %%
import librosa
import librosa.display
import matplotlib.pyplot as plt

#Waveplot
plt.figure(figsize=(6, 8))
plt.subplot(2, 1, 1)
y,sr=librosa.load("D:\\Studia\\thesis\\data\\KillerWhale\\894050CI.wav")
librosa.display.waveshow(y,sr=sr, x_axis='time', color='purple',offset=0.0)
plt.title('Waveplot')
plt.xlabel('Time')
plt.ylabel('Amplitude')

#Mel-spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.subplot(2, 1, 2)
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')

# %% [markdown]
# Generating Mel Frequency Cepstral Coefficient representation

# %%
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('MFCCs')
plt.xlabel('Time')
plt.ylabel('MFCC Coefficients')
plt.tight_layout()
plt.show()

# %% [markdown]
# More visualizations can be found in notebook with models, They are put there because of the saved variables, on which these visualizations are built.


