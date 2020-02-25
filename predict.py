import pickle
import os
import numpy as np
import pandas as pd
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
from sklearn.metrics import accuracy_score

def build_predictions(audio_dir):
    
    y_pred = []
    print('Extracting features from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        y_prob = []
        
        for i in range(0, wav.shape[0]-config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=1103)
            x = (x - config.min) / (config.max - config.min)
            if x.shape[0] == 6:
                x = np.append(x, np.zeros([3, 13]), 0)
            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode == 'time':
                x = np.expand_dims(x, axis=0)
            y_hat = model.predict(x)
            y_prob.append(y_hat)

        y_pred.append(y_prob)
    return y_pred

df = pd.read_csv('instruments.csv')
classes = list(np.unique(df.label))
fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('pickles', 'conv.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)

model = load_model(config.model_path)

y_pred = build_predictions('clean')
print(len(y_pred), len(y_pred[0]), len(y_pred[1]))