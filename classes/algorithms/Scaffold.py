# Import parameters
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from classes.algorithms.algorithms_utils import interpolate_weights, flatten_weights
from classes.algorithms.FedAvg import FedAvg
from classes.params import fl_param, simul_param
import tensorflow as tf
import numpy as np
import copy

class Scaffold(FedAvg):
    def __init__(self, model, id=None):
        super(Scaffold, self).__init__(model, id)
        self.name = "Scaffold"
        self.c_global = copy.copy(self.model_struct)
        if id is None: # Parameter Server attribute initialization
            self.c_local_diff = [copy.copy(self.model_struct) for _ in range(fl_param.NUM_CLIENTS)]
        else: # Client attribute initialization
            self.c_i = copy.copy(self.model_struct)
            self.c_i_plus = copy.copy(self.model_struct)
            self.K = 0
        print(f'{self.name} in use\n')

    """ SERVER: Parameter Server Based Architecture """
    def aggregate_weights(self, active_check):
        super().aggregate_weights(active_check)
        # Update control variables
        sample_clients = self.clients[active_check]
        total_samples = np.sum([self.samples[client_]  for  client_ in sample_clients])
    
        for layer_ in range(len(self.model_struct)):
            self.c_global[layer_] += 1/total_samples * (np.sum([self.c_local_diff[client_][layer_] * self.samples[client_]  for  client_ in sample_clients], axis=0))
            # or 
            #self.c_global[layer_] += sum(active_check)/fl_param.NUM_CLIENTS * (np.sum([self.c_local_diff[client_][layer_] for  client_ in sample_clients], axis=0))

    """ SERVER: Parameter Server Based Architecture """        
    def set_param_server(self, payload):
        super().set_param_server(payload)
        client_id = payload['device']
        for layer_ in range(len(self.model_struct)):
            self.c_local_diff[client_id][layer_] = np.asarray(payload[f'control_diff_layer{layer_}'])
    
    
    """ CLIENT: Parameter Server Based Architecture """
    def set_param_client(self, payload):
        super().set_param_client(payload)
        for layer_ in range(len(self.model_struct)):
            self.c_global[layer_] = np.asarray(payload[f'global_control_layer{layer_}'])    
    
    
    """ CLIENT: Parameter Server Based Architecture """
    def get_param_client(self):
        self.K = 1 if self.K == 0 else self.K# to avoid division by zero when you only have to send
        detObj = super().get_param_client()
        weights_client = self.model_client.get_weights()
        for layer_ in range(len(self.model_struct)):
            self.c_i_plus[layer_] = self.c_i[layer_] - self.c_global[layer_] + 1/(self.K*simul_param.LR)*(self.weights_global[layer_] - weights_client[layer_] )
            detObj['control_diff_layer{}'.format(layer_)] = (
                        np.array(self.c_i_plus[layer_]) - np.array(self.c_i[layer_]))
            self.c_i[layer_] =self.c_i_plus[layer_]  # update control local
        self.K = 0
        
        return detObj
    
    
    def get_param_server(self):
        detObj = super().get_param_server()
        for layer_ in range(len(self.model_struct)):
            detObj['global_control_layer{}'.format(layer_)] = self.c_global[layer_]  # .tolist()
        return detObj
    

    
    def train_step(self, data_sample, masks):
        with tf.GradientTape() as tape:
            predictions = self.model_client(data_sample)
            loss = self.loss_function(masks, predictions)
        gradients = tape.gradient(loss, self.model_client.trainable_variables)  # g_i(y_i)
        updated_gradients = [g - c_i + c for g, c_i, c in
                                zip(gradients, self.c_i, self.c_global)]
        self.optimizer.apply_gradients(zip(updated_gradients, self.model_client.trainable_variables))
        self.K+=1
        return loss

