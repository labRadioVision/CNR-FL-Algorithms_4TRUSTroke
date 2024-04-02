# Import parameters
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from classes.algorithms.algorithms_utils import interpolate_weights, flatten_weights
from classes.algorithms.FedAvg import FedAvg
from classes.params import fl_param, simul_param
import tensorflow as tf
import numpy as np
import copy

class FedNova(FedAvg):

    def __init__(self, model, id=None):
        super(FedNova, self).__init__(model, id)
        self.name = "FedNova"
        if id is None: # Parameter Server attribute initialization
            self.tau = [0 for _ in range(fl_param.NUM_CLIENTS)]
        else: # Client attribute initialization
            self.tau_i = 0
        print(f"{self.name} in use")

    
    """ SERVER: Parameter Server Based Architecture """
    def aggregate_weights(self, active_check):
        # Update weights
        sample_clients = self.clients[active_check]
        total_samples = np.sum([self.samples[client_]  for  client_ in sample_clients])
        weights_agg = []        
        tau_eff = np.sum([self.tau[client_] * self.samples[client_] for client_ in sample_clients])/total_samples
        for layer_ in range(len(self.model_struct)):
            
            norm_grad = np.sum([self.samples[client_]/ self.tau[client_] * (self.weights_global[layer_] - self.weights_clients[client_][layer_]) for client_ in sample_clients], axis=0)/total_samples
            weights_layer = self.weights_global[layer_] - tau_eff * norm_grad
            weights_agg.append(weights_layer) 
        
        self.weights_global = interpolate_weights(self.weights_global, weights_agg)

    
    """ CLIENT: Parameter Server Based Architecture """
    def get_param_client(self):
        detObj = super().get_param_client()# detObj['tau_i'] = algorithm.tau_i
        detObj['tau_i'] = self.tau_i
        self.tau_i = 0
        return detObj
    
    
    """ SERVER: Parameter Server Based Architecture """        
    def set_param_server(self, payload):
        super().set_param_server(payload)
        
        client_id = payload['device']
        self.tau[client_id] = payload['tau_i']    
    
    
    """ CLIENT: Parameter Server Based Architecture """   
    def train_step(self, data_sample, masks):
        loss = super().train_step(data_sample, masks)
        self.tau_i+=1
        return loss