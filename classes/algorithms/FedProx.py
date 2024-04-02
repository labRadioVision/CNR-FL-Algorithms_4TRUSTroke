# Import parameters
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from classes.algorithms.algorithms_utils import interpolate_weights
from classes.algorithms.FedAvg import FedAvg
from classes.params import fl_param, simul_param
import tensorflow as tf
import numpy as np
import copy

class FedProx(FedAvg):

    def __init__(self, model, id=None):
        super(FedProx, self).__init__(model, id)
        self.name = "FedProx"
        if id is not None:
            self.mu = fl_param.MU
        print(f'{self.name} in use with mu={fl_param.MU}')

    
    """ CLIENT: Parameter Server Based Architecture """
    def train_step(self, data_sample, masks):
        with tf.GradientTape() as tape:
            predictions = self.model_client(data_sample, training=True)
            loss = self.loss_function(masks, predictions)
            
            for layer_local, layer_global in zip(self.model_client.get_weights(), self.weights_global):
                loss += self.mu/2*tf.reduce_sum(tf.square((tf.cast(layer_local, dtype=tf.float32) - tf.cast(layer_global, dtype=tf.float32))))
            
            #prox_term = 0
            #for layer_local, layer_global in zip(self.model_client.get_weights(), self.weights_global):
            #    prox_term += tf.reduce_sum(tf.square((tf.cast(layer_local, dtype=tf.float32) - tf.cast(layer_global, dtype=tf.float32))))
            #print("prox_term:{:.3f}".format(self.mu/2*prox_term))                         
            #loss += self.mu/2*prox_term
        gradients = tape.gradient(loss, self.model_client.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model_client.trainable_variables))
        return loss