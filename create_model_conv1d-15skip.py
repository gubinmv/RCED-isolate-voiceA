# модуль создания нейронной сети с 15 скрытыми слоями
# автор: Губин М.В.

import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Conv1D, BatchNormalization
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.layers.merge import concatenate, add
from keras import optimizers
from keras import regularizers
import h5py
import os
import tensorflow as tf
from keras.layers import Layer
import tensorflow as tf

from param_project import args
import param_project

from tensorflow.keras.utils import plot_model

import my_loss_functions

img_rows, img_cols = args.img_rows, args.img_cols
input_shape = (img_rows, img_cols)
input_img = Input(input_shape, name='img')

batch_size = 64
l2_lambda = 1e-5

def my_conv1d(input_layer, filters, kernel_size, use_bn=True):
    my_layer = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        strides=1,
        padding='same',
        use_bias=True,
        kernel_regularizer=regularizers.l2(1e-5),
        bias_regularizer=regularizers.l2(1e-5),
        activity_regularizer=regularizers.l2(1e-5)
        )(input_layer)
    if use_bn:
        my_layer_out = BatchNormalization()(my_layer)
    else:
        my_layer_out = my_layer

    return my_layer_out

model = Sequential()
conv0 = my_conv1d(input_img, filters=10, kernel_size=11)
conv1 = my_conv1d(conv0, filters=12, kernel_size=7)
conv2 = my_conv1d(conv1, filters=14, kernel_size=5)
conv3 = my_conv1d(conv2, filters=15, kernel_size=5)
conv4 = my_conv1d(conv3, filters=19, kernel_size=5)
conv5 = my_conv1d(conv4, filters=21, kernel_size=5)
conv6 = my_conv1d(conv5, filters=23, kernel_size=7)
conv7 = my_conv1d(conv6, filters=25, kernel_size=11)
conv8 = my_conv1d(conv7, filters=23, kernel_size=7)
conv8_1 = add([conv6, conv8])
conv9 = my_conv1d(conv8_1, filters=21, kernel_size=5)
conv9_1 = add([conv5, conv9])
conv10 = my_conv1d(conv9_1, filters=19, kernel_size=5)
conv10_1 = add([conv4, conv10])
conv11 = my_conv1d(conv10_1, filters=15, kernel_size=5)
conv11_1 = add([conv3, conv11])
conv12 = my_conv1d(conv11_1, filters=14, kernel_size=5)
conv12_1 = add([conv2, conv12])
conv13 = my_conv1d(conv12_1, filters=12, kernel_size=7)
conv13_1 = add([conv1, conv13])
conv14 = my_conv1d(conv13_1, filters=10, kernel_size=11)
conv14_1 = add([conv0, conv14])
conv15 = Conv1D(filters = 1, kernel_size = 1, strides=1, activation='relu', padding='same')(conv14_1)

model = Model(inputs=[input_img], outputs=[conv15])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
model.compile(loss=my_loss_functions.my_get_SDR_1, optimizer=optimizer, metrics=[my_loss_functions.my_get_SDR, keras.metrics.MeanSquaredError()])
model.summary()

model_path='model_conv1d-15skip.hdf5'
model.save(model_path)
print('Saved no-trained model at %s ' % model_path)