# Import parameters
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from classes.algorithms.algorithms_utils import interpolate_weights, flatten_weights
from classes.algorithms.FedAvg import FedAvg
from classes.params import fl_param, simul_param
import tensorflow as tf
import numpy as np
import copy

class FedDyn(FedAvg):
    def __init__(self, model, id=None):
        super(FedDyn, self).__init__(model, id)
        self.name = "FedDyn"
        self.alpha = fl_param.ALPHA_DYN
        if id is None: # Parameter Server attribute initialization
            self.h = copy.copy(self.model_struct) # server state
        else: # Client attribute initialization
            self.deltaLk = copy.copy(self.model_struct)
        
        assert  not flatten_weights(self.weights_global).all() # Must not be init as zeros
        print(f"{self.name} in use with alpha={self.alpha}")


    """ SERVER: Parameter Server Based Architecture """
    def aggregate_weights(self, active_check):
        # Update weights
        sample_clients = self.clients[active_check]
        total_samples = np.sum([self.samples[client_]  for  client_ in sample_clients])
        
        weights_agg = []
        for layer_ in range(len(self.model_struct)):
            self.h[layer_] = self.h[layer_] - (self.alpha/fl_param.NUM_CLIENTS) * np.sum([(self.weights_clients[client_][layer_] - self.weights_global[layer_]) for client_ in sample_clients], axis=0)
            weights_layer = 1/total_samples * (np.sum([self.samples[client_]*self.weights_clients[client_][layer_]  for  client_ in sample_clients], axis=0))  - 1/self.alpha * self.h[layer_]
            weights_agg.append(weights_layer)
        
        self.weights_global = interpolate_weights(self.weights_global, weights_agg)
    
    """ CLIENT: Parameter Server Based Architecture """
    def get_param_client(self):
        for layer_ in range(len(self.model_struct)):# Update local gradients
            self.deltaLk[layer_] -= self.alpha * (self.weights_global[layer_] - self.model_client.get_weights()[layer_])
        detObj = super().get_param_client()
        return detObj
    
    
    """ CLIENT: Parameter Server Based Architecture """   
    def train_step(self, data_sample, masks):
        with tf.GradientTape() as tape:
            predictions = self.model_client(data_sample, training=True)
            loss = self.loss_function(masks, predictions)

            for layer_local, layer_global, loc_grad in zip(self.model_client.get_weights(), self.weights_global, self.deltaLk):
                loss += self.alpha/2*tf.reduce_sum(tf.square((tf.cast(layer_local, dtype=tf.float32) - tf.cast(layer_global, dtype=tf.float32))))
                loss -= tf.reduce_sum(tf.math.multiply(tf.cast(layer_local, dtype=tf.float32), tf.cast(loc_grad, dtype=tf.float32)))
            
        gradients = tape.gradient(loss, self.model_client.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model_client.trainable_variables))
        return loss


