# модуль создания обучающей и тестирующей выборок
# автор: Губин М.В.

import os, sys
import numpy as np
import pandas

from param_project import args
import param_project

def get_mix(noise, voice):

    len_noise = len(noise)
    len_voice = len(voice)

    if (len_voice > len_noise):
        k_ = 1 + len_voice // len_noise
        noise = [noise] * k_
        noise = np.concatenate(noise, axis=0)
    else:
        k_ = 1 + len_noise // len_voice
        voice = [voice] * k_
        voice = np.concatenate(voice, axis=0)

    tot_len = min(len(voice), len(noise))

    mix = voice[:tot_len] + noise[:tot_len]

    return voice[:tot_len], noise[:tot_len], mix


def get_data_set(path_to_file_npz, list_noise, list_voice, prefix):
    data_X = []
    data_Y = []

    fs = args.fs
    window = args.window
    overlap = args.overlap
    step_wave = args.step_wave

    img_rows, img_cols = args.img_rows, args.img_cols
    maxRazmer = args.maxRazmer

    print("\n Loading wav files ...")

    list_wav_data_noise = param_project.get_wav_files(list_noise)
    list_wav_data_voice = param_project.get_wav_files(list_voice)

    noise = np.concatenate(list_wav_data_noise, axis=0)
    voice = np.concatenate(list_wav_data_voice, axis=0)

    voice, noise, mix = get_mix(noise, voice)

    noisePower = sum(noise**2)/len(noise)
    signalPower = sum(voice**2)/len(voice)

    snr_ = args.snr_in * 0.1 / 2
    n_rate = np.sqrt(signalPower / noisePower)
    k_mix_noise = np.abs(n_rate / 10**snr_)
    print("\n k_mix_noise = ", k_mix_noise)
    noise = k_mix_noise * noise
    mix = voice + noise

    noisePower = sum(noise**2)/len(noise)

    snr = 10*np.log10(signalPower/noisePower)
    print("\n SNR(dB) = ", snr)

    path_wav_out = args.path_wav_out + prefix
    print("path as ", path_wav_out)

    if not os.path.exists(args.path_wav_out):
        os.makedirs(args.path_wav_out)

    if (model_name == 'Backward'):
        param_project.save_wav_file(path_wav_out + 'data_x.wav', mix)
        param_project.save_wav_file(path_wav_out + 'data_y.wav', noise)
        param_project.save_wav_file(path_wav_out + 'voice.wav', voice)

        stft_y = param_project.get_stft_samples(wav_file=noise)
        stft_x = param_project.get_stft_samples(wav_file=mix)

    elif (model_name == 'Forward'):
        param_project.save_wav_file(path_wav_out + 'data_x.wav', mix)
        param_project.save_wav_file(path_wav_out + 'data_y.wav', voice)
        param_project.save_wav_file(path_wav_out + 'noise.wav', noise)

        stft_y = param_project.get_stft_samples(wav_file=voice)
        stft_x = param_project.get_stft_samples(wav_file=mix)
    else:
        print('\n Stop! not method model \n')
        exit() #os._exit(0)


    x_train = np.array([])
    x_train = stft_x

    y_train = np.array([])
    y_train = stft_y[:, :, img_cols-1]

    print('\n shape baze')
    print("x_train.shape = ",x_train[0].shape)
    print("y_train.shape = ",y_train[0].shape)
    print('len train records x = ', len(x_train))
    print('len train records y = ', len(y_train))

    tot_len = min(len(x_train),len(y_train))
    y_train = y_train[:tot_len]
    x_train = x_train[:tot_len]

    #reshape tensor from conv_1D_inverse
    data_X = x_train.reshape(x_train.shape[0], img_rows, img_cols)
    data_Y = y_train.reshape(y_train.shape[0], img_rows, 1)

    print("save in npz-file ...")
    np.savez(path_to_file_npz, DataX=data_X, DataY=data_Y)

    return 0

def get_list_files(dataset_class, n_records, order = 1):

    list_files = pandas.DataFrame(columns=['file_name', 'class', 'length'])
    if (order == 1):
        n_records = n_records // len(dataset_class)
        for i in range(len(dataset_class)):
            list_file = df[df['class'].isin([dataset_class[i]])]
            list_files = pandas.concat([list_files, list_file[0:n_records]])
    else:
        list_files = df[df['class'].isin(dataset_class)].sample(n_records)[['file_name', 'class', 'length']]

    print("\n list fragments")
    print(list_files.groupby('class').count())

    return list_files

#start programm
k_mix_noise = args.k_mix_noise
model_name = args.model_name

#open dataset
dataset_csv = args.path_source_dataset_csv
df = pandas.read_csv(dataset_csv, delimiter=',')
pandas.set_option('display.max_rows', None)
print(df['class'].value_counts())

#отобрать записи из датасета
#noise train
dataset_class = args.class_noise
n_records = args.n_records_noise
list_csv_noise_train = get_list_files(dataset_class = dataset_class, n_records = n_records)

#noise test
dataset_class = args.class_noise_test
n_records = args.n_records_noise_test
list_csv_noise_test = get_list_files(dataset_class = dataset_class, n_records = n_records)

#voice train
dataset_class = args.class_voice
n_records = args.n_records_voice
list_csv_voice_train = get_list_files(dataset_class = dataset_class, n_records = n_records)

#voice test
dataset_class = args.class_voice_test
n_records = args.n_records_voice_test
list_csv_voice_test = get_list_files(dataset_class = dataset_class, n_records = n_records)

# create npz for train phase
print("\n Creating spectrogram for train")
path_to_file_npz = args.path_to_file_train_npz
hlp = get_data_set(path_to_file_npz, list_csv_noise_train, list_csv_voice_train, 'train_')

# create npz for test phase
print("\n Creating spectrogram for test")
path_to_file_npz = args.path_to_file_test_npz
hlp = get_data_set(path_to_file_npz, list_csv_noise_test, list_csv_voice_test, 'test_')

print("\n End programm creating spectrogramm.")
