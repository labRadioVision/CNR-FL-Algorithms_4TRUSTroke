# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# MacOs
# import os
# import plaidml.keras
# plaidml.keras.install_backend()
# os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
# import keras
from classes.params import simul_param


class Mobinet:
    
    def __init__(self, input_shape, output_shape):
        print('MobileNetV2 model')
        if input_shape[0] < 32 or input_shape[1] < 32:
            raise ValueError('MobileNetV2 model is designed for at least 32x32 images.')
        self.input_shape = input_shape
        self.output_shape = output_shape

    # Model definition
    def return_model(self):
        model = tf.keras.applications.MobileNetV2(self.input_shape, classes=self.output_shape, weights=None)
        
        return model
