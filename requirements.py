# %% [markdown]
# Requirements

# %%
import numpy as np
import pandas as pd
import sklearn.model_selection
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import wave
import tensorflow as tf
import math
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')
import tensorflow
from tensorflow.keras.layers import LSTM, Dense
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from math import floor
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import statistics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier



