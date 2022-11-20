# Usual Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn

# Librosa (the mother of audio files)
import librosa
import librosa.display
import IPython.display as ipd
import os
from sklearn.ensemble import RandomForestClassifier
# Importing 1 file
def model(atributes):
    pickled_model = pickle.load(open('rforest.pkl', 'rb'))
    return pickled_model.predict(x_new)[0]


