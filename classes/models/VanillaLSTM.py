# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from classes.params import simul_param


class VanillaLSTM:

    def __init__(self, input_shape, output_shape):

        self.input_shape = input_shape# (train_X.shape[1], train_X.shape[2])
        self.output_shape = output_shape
    # Model definition
    def return_model(self):

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.LSTM(50, activation='relu', input_shape=self.input_shape, return_sequences=True))#
        #model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(self.output_shape , activation='softmax'))

        return model

