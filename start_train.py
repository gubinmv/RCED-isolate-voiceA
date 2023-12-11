# модуль обучучения нейронной сети
# автор: Губин М.В.

import numpy as np
from numpy import array, arange, zeros, linalg, log10, abs as np_abs

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling2D, Convolution2D
from keras.callbacks import EarlyStopping

import h5py
import scipy
from scipy import signal
import tensorflow as tf
from sys import argv
import soundfile as sf
import librosa
import os, math

from keras.utils.generic_utils import get_custom_objects
from keras import backend as K

from param_project import args
import param_project
import my_loss_functions

class CustomSaver(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):

        k = 0.05
        lrate = self.model.optimizer.learning_rate * math.exp(-k*epoch)
        tf.keras.backend.set_value(self.model.optimizer.lr, lrate)


        print("model.optimizer.learning_rate", self.model.optimizer.learning_rate)


    def on_epoch_end(self, epoch, logs={}):

        global max_val_my_get_SDR
        self.model.save(args.path_model)

        out_string = "\n epoch =" + str(epoch) + " loss =" + str(logs["loss"]) + " my_get_SDR =" + str(logs["my_get_SDR"]) +" val_my_get_SDR =" + str(logs["val_my_get_SDR"])
        print(out_string)

        f = open('./epochs_metrics.txt', 'a+')
        f.write(out_string)
        f.close()

def learning_rate(x):
    L_rate = x
    return L_rate


#keras model param
batch_size = args.batch_size
epochs = args.epochs

#Parametrs of audio and TF
fs      = args.fs
window  = args.window
overlap = args.overlap
step_wave = args.step_wave
img_rows, img_cols = args.img_rows , args.img_cols
maxRazmer = args.maxRazmer

path_history = args.path_history
model_path = args.path_model

saver = CustomSaver()
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose=1, patience = 20)

#create model CNN
model=load_model(model_path, custom_objects={'my_get_SDR': my_loss_functions.my_get_SDR, 'my_get_SDR_1': my_loss_functions.my_get_SDR_1})
model.summary()

print("\n Loading DataSet ...")

path = "./TrainSet.npz"
with np.load(path) as data:
    x_train = data['DataX']
    y_train = data['DataY']

path = "./TestSet.npz"
with np.load(path) as data:
    x_val = data['DataX']
    y_val = data['DataY']

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0, validation_data=(x_val,y_val), shuffle = True, callbacks=[saver])

#save model CNN
model.save(model_path)
print('Saved trained model at %s ' % model_path)

#save history
file_histiry = open(path_history,'w')
print(history.history, file=file_histiry)
file_histiry.close()

print("\n End programm")


