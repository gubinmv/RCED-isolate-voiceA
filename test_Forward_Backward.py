# модуль окончательного тестирования обученных нейронных сетей
# автор: Губин М.В.

import numpy as np
from numpy import array, arange, zeros, linalg, log10, abs as np_abs

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv1D

import h5py
import scipy
from scipy import signal
import tensorflow as tf
from sys import argv

from keras.utils.generic_utils import get_custom_objects
from keras import backend as K

from pystoi import stoi
from pesq import pesq

from param_project import args
import param_project
import my_loss_functions

#Parametrs of audio and TF
fs = args.fs
window      = args.window
k_overlap   = args.k_overlap
overlap     = int(k_overlap * window)
step_wave   = int(window - overlap)

# size of spectrogram
img_rows    = 1 + window // 2
img_cols    = args.img_cols
method      = args.model_name

maxRazmer = (window - overlap) * (img_cols - 1) + window
batch_size = 64

path        = "./Out/"
path_model  = "./"

if (method == "Forward"):
    wave_file_voice = path + "test_data_y.wav"
    wave_file_mix   = path + "test_data_x.wav"
    wave_file_out   = path + "out.wav"
else:
    wave_file_voice = path + "test_voice.wav"
    wave_file_mix   = path + "test_data_x.wav"
    wave_file_out   = path + "out.wav"


#open model
model_path = path_model + "model_conv1d-15skip.hdf5"
model=load_model(model_path, custom_objects={'my_get_SDR': my_loss_functions.my_get_SDR, 'my_get_SDR_1': my_loss_functions.my_get_SDR_1})
model.summary()

#load wav-file
sr, wave_data = scipy.io.wavfile.read(wave_file_mix)
_, _, in_stft = scipy.signal.stft(wave_data, fs=fs, nperseg=window, noverlap = overlap, window='hann')

in_stft_amp = np.maximum(np.abs(in_stft*100), 1e-5)
in_data = in_stft_amp
phase_data = in_stft / in_stft_amp

in_data = in_stft_amp

num_samples = in_data.shape[1]-img_cols
sftt_frame = np.array([in_data[:, i:i+img_cols] for i in range(0, num_samples, 1)])

pred = model.predict(sftt_frame, verbose=1)
pred = pred.reshape((len(pred),img_rows))

pred = np.transpose(pred)

out_len = pred.shape[1]

if (method == "Forward"):
    out_stft = pred[:,:out_len] * phase_data[:,img_cols - 1:out_len + img_cols - 1]
else:
    out_stft = in_stft[:,img_cols - 1:out_len + img_cols - 1] - pred[:,:out_len] * phase_data[:,img_cols - 1:out_len + img_cols - 1]


_, out_audio = signal.istft(out_stft, fs=fs, noverlap = overlap, nperseg=window, window='hann')

scipy.io.wavfile.write(wave_file_out, 8000, out_audio)

fs, clean = scipy.io.wavfile.read(wave_file_voice)

start_segment = step_wave * (img_cols - 1)
num_segments = min(len(clean),len(out_audio))-start_segment

SDR_l2 = param_project.get_SDR(clean[start_segment:num_segments+start_segment], out_audio[:num_segments],)
print ("\n SDR_l2 = ", SDR_l2)

STOI = stoi(clean[start_segment:num_segments+start_segment], out_audio[:num_segments], fs, extended=False)
print(" STOI = ", STOI)

num_segments = min(len(clean),len(out_audio))-8000
num_segments = min(500000, num_segments)

PESQ = pesq(fs, clean[start_segment:num_segments + start_segment], out_audio[:num_segments], 'nb')
print(" PESQ = ", PESQ)

SI_SNR_ = param_project.SI_SNR(out_audio[:num_segments], clean[start_segment:num_segments+start_segment])
print(" SI_SNR = ", SI_SNR_.item())

f = open("out_test_metric.txt", 'a+')
f.write("\n  SDR = " + str(SDR_l2) + " STOI = " + str(STOI) + " pesq = " + str(PESQ) + " SI_SNR = " + str(SI_SNR_))
f.close()

