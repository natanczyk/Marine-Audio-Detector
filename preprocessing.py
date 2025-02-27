# %% [markdown]
# A few information about the dataset

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

print(len(os.listdir("D:\\Studia\\thesis\\data\\KillerWhale")))

num_of_files = 0

for dirpath, dirnames, filenames in os.walk("D:\\Studia\\thesis\\data"):
    num_of_files+= len(filenames)
    
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")
print(num_of_files)

# %% [markdown]
# Firstly I am going through all elements in the dataset and I am creating dataframe with path to the sample and corresponding label.

# %%
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import wave
import tensorflow as tf


directory_path = "D:\\Studia\\thesis\\data"

#paths to subfolders
subdirectories = [os.path.join(directory_path, d) for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
print(subdirectories)

longest_sample = 0
shortest_sample = 5

#Creating dataframe out of paths and and names
df=[]
for subdirectory in subdirectories:
    #Every sample that is not killer whale is labeled 0, killer whale samples are labeled 1
    label = os.path.basename(subdirectory)
    ID=0
    if label=="KillerWhale":
        ID=1
        
    paths_and_labels = []
    
    for file in os.listdir(subdirectory):
        file_path = os.path.join(subdirectory, file)
        paths_and_labels.append([file_path, ID])
        
        #I want to check the length of shortest and longest sample
        with wave.open(file_path, 'r') as f:
            frames = f.getnframes()
            frame_rate = f.getframerate()
            duration = frames / float(frame_rate)
        if duration>longest_sample:
            longest_sample=duration
        if duration<shortest_sample:
            shortest_sample=duration
    
    dataframe = pd.DataFrame(paths_and_labels, columns=['Path', 'ID'])
    df.append(dataframe)
    
#I am creating a single dataframe of all elements  
ready_df = pd.concat(df, ignore_index = True)

#Here I am shuffling the dataset
ready_df = ready_df.sample(frac=1).reset_index(drop=True)

print("There is {} Killer Whale samples and {} samples of diffrent species".format(len(os.listdir("D:\\Studia\\thesis\\data\\KillerWhale")), num_of_files-len(os.listdir("D:\\Studia\\thesis\\data\\KillerWhale"))))
print("longest sample duration:", longest_sample, "seconds")
print("shortest sample duration:", shortest_sample, "seconds")


# %% [markdown]
# I the cell below I am converting audio samples into MFCCs. 

# %%
import librosa
import mat
import json
from sklearn.preprocessing import StandardScaler

list_of_paths = ready_df["Path"].tolist()
list_of_ID = ready_df["ID"].tolist()

json_path_scaled = "MFCC_data_scaled.json"

duration = 1 #seconds
sample_rate = 16000
samples_per_track=sample_rate * duration

data = {"mfcc" : [],
        "label" : [],
        "frequency": [],
        "duration": [],
        "pitch": []
        }

data_scaled = {"mfcc" : [],
               "label" : [],
               "frequency": [],
               "duration": [],
               "pitch": []
              }

#Next three functions are supposed to extract frequency, duration and pitch from the given sample.
def extract_frequency(signal, sr):
    fourier_transform = np.fft.rfft(signal)
    abs_fourier_transform = np.abs(fourier_transform)
    frequency = np.argmax(abs_fourier_transform)
    return frequency

def extract_duration(signal, sr):
    duration = len(signal) / sr
    return duration

def extract_pitch(signal, sr):
    pitches, magnitudes = librosa.piptrack(y=signal, sr=sr)
    # I am trying to find the index of the maximum value in magnitudes along the frequency. when I am using pitch = pitches[np.argmax(magnitudes)], it returns the position of the maximum value as if the 2D array was stretched out into a single line, which leads to an error.
    # On Stackoverflow website I found function np.unravel_index. Cited meaning: "function takes the flattened position and converts it back into two positions: one for the time dimension and one for the frequency dimension. These two positions should correctly point to the maximum value in the 2D array." 
    f_index, t_index = np.unravel_index(np.argmax(magnitudes), magnitudes.shape) 
    pitch = pitches[f_index, t_index]
    return pitch  
     
     
#This is a function which should add noise to given sample. It is used for data augmentation.   
def add_noise(signal):
    noise = np.random.normal(0, 0.005, signal.shape)
    return signal + noise



#This function calculates the MFCCs of the audio file located at path and appends them, along with other features, to the data_scaled dictionary. 
#It also segments the audio signal and computes MFCCs for each segment. If augmentation_fn is provided, it applies this function to the signal for data augmentation.
def spectro(path, i, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5, augmentation_fn=None):

    signal, sr = librosa.load(path, sr=sample_rate)
    while len(signal) < samples_per_track:
        signal = np.concatenate((signal, signal[:samples_per_track - len(signal)]))
    num_samples_per_segment=int(samples_per_track / num_segments)
    expected_num_mfcc_per_segment= math.ceil(num_samples_per_segment / hop_length)
    
    if augmentation_fn:
        signal = augmentation_fn(signal)
    
    #process every segment
    for s in range(num_segments):
        start_sample = num_samples_per_segment * s
        finish_sample = start_sample + num_samples_per_segment
    
        #transforming audio into mfcc
        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length
                                    )
        mfcc = mfcc.T
        if len(mfcc) == expected_num_mfcc_per_segment:
            scaler = StandardScaler()
            mfcc_scaled = scaler.fit_transform(mfcc)
            data_scaled["mfcc"].append(mfcc_scaled.tolist()) #numpy array changed into a list
            data_scaled["label"].append(list_of_ID[i])
            data_scaled["frequency"].append(extract_frequency(signal, sr))
            data_scaled["duration"].append(extract_duration(signal, sr))
            data_scaled["pitch"].append(extract_pitch(signal, sr))
            
#This variable determines how many times the minority class will be oversampled.       
upsampling_factor = 4        
 
#For each file, it extracts features and appends them to data_scaled.
#If the label is 1 (indicating the minority class), the signal is augmented and the features are extracted again multiple times, according to the upsampling factor.       
for i in range(0, len(list_of_paths)-1):
    path = list_of_paths[i]
    label = list_of_ID[i]
    spectro(path,i)
    
    if label == 1:
        for j in range(upsampling_factor-1):
            spectro(path, i, augmentation_fn=add_noise)
    

# %% [markdown]
# In this code we save numpy dataframe into a JSON file.

# %%
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


    
with open(json_path_scaled, "w") as fp:
    json.dump(data_scaled, fp, cls=NumpyEncoder, indent=4)


