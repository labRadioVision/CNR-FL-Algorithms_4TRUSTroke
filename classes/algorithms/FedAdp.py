# Import parameters
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from classes.algorithms.algorithms_utils import interpolate_weights, flatten_weights
from classes.algorithms.FedAvg import FedAvg
from classes.params import fl_param, simul_param
import numpy as np
import math
import copy

class FedAdp(FedAvg):

    def __init__(self, model, id=None):
        super(FedAdp, self).__init__(model, id)
        self.name = "FedAdp"
        if id is None: # Parameter Server attribute initialization
            self.alpha = fl_param.ALPHA_ADP
            self.gradients_global = copy.copy(self.model_struct)
            self.gradients_clients = [copy.copy(self.model_struct) for _ in range(fl_param.NUM_CLIENTS)]
            self.theta_tilde = [None for _ in range(fl_param.NUM_CLIENTS)]
            self.psi_tilde = [None for _ in range(fl_param.NUM_CLIENTS)]
            self.round = 0        
        print(f"{self.name} in use with alpha={fl_param.ALPHA_ADP}")
            
        
    """ SERVER: Parameter Server Based Architecture """        
    def set_param_server(self, payload):
        super().set_param_server(payload)
        self.round = payload['round'] + 1 # needed for theta_tildes
        
    
    """ SERVER: Parameter Server Based Architecture """
    def aggregate_weights(self, active_check):
        # Update weights
        sample_clients = self.clients[active_check]
        total_samples = np.sum([self.samples[client_]  for  client_ in sample_clients])

        # compute local gradients
        for client_ in sample_clients:
            for layer_ in range(len(self.model_struct)):
                self.gradients_clients[client_][layer_] = - 1/simul_param.LR * (self.weights_clients[client_][layer_] - self.weights_global[layer_])
                
        #compute global gradient
        for layer_ in range(len(self.model_struct)):
            self.gradients_global[layer_] = 1/total_samples * (np.sum([self.gradients_clients[client_][layer_] * self.samples[client_]  for  client_ in sample_clients], axis=0))
        
        # Compute smooth instantaneous angle
        for client_ in sample_clients:  
            flat_global = flatten_weights(self.gradients_global)
            flat_client = flatten_weights(self.gradients_clients[client_])
            inner = np.inner(flat_global, flat_client)
            norms = np.linalg.norm(flat_global)*np.linalg.norm(flat_client)
            theta = np.arccos(np.clip(inner / norms, -1.0, 1.0))
        
            if self.theta_tilde[client_] == None:
                self.theta_tilde[client_] = float(theta)
            self.theta_tilde[client_] = ((self.round-1)/self.round)*self.theta_tilde[client_] + (1/(self.round))*theta
            
        # Compute aggregation weights
        sum_phi = 0
        for client_ in sample_clients:
            f = self.alpha*(1-math.exp(-math.exp(-self.alpha*(self.theta_tilde[client_]-1))))
            self.psi_tilde[client_] = math.exp(f)*self.samples[client_]
            sum_phi+=self.psi_tilde[client_]
        #print('psi_tilde:{}'.format(1/sum_phi*np.array(self.psi_tilde)[active_check]))   
        # Update weights
        weights_agg = []
        for layer_ in range(len(self.model_struct)):
            weights_layer = 1/sum_phi*(np.sum([self.weights_clients[client_][layer_] * self.psi_tilde[client_]  for  client_ in sample_clients], axis=0))
            weights_agg.append(weights_layer) 
        self.weights_global = interpolate_weights(self.weights_global, weights_agg)

    
        
    
