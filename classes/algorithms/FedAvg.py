# Import parameters
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from classes.algorithms.algorithms_utils import interpolate_weights
from classes.params import fl_param, simul_param
import tensorflow as tf
import numpy as np
import copy

class FedAvg:

    def __init__(self, model, id=None):
        self.model_struct = [np.zeros_like(w) for w in model.get_weights()]
        self.weights_global = copy.copy(model.get_weights())# problems if zero for feddyn
        self.name = "FedAvg"
        if id is None: # Parameter Server attribute initialization
            print("Initializing PS")
            self.clients = np.arange(fl_param.NUM_CLIENTS)
            self.samples = [0 for _ in range(fl_param.NUM_CLIENTS)]
            self.weights_clients = [copy.copy(self.model_struct) for _ in range(fl_param.NUM_CLIENTS)]
        else: # Client attribute initialization
            print(f"Initializing Client {id}")
            self.id = id
            self.model_client = copy.copy(model)
            self.optimizer = simul_param.OPTIMIZER(learning_rate=simul_param.LR)
            self.loss_function = simul_param.LOSS
            self.metrics = simul_param.METRICS
        print(f"{self.name} in use")


    """ SERVER: Parameter Server Based Architecture """
    def aggregate_weights(self, active_check):
        # Update weights
        sample_clients = self.clients[active_check]
        total_samples = np.sum([self.samples[client_]  for  client_ in sample_clients])

        weights_agg = []
        for layer_ in range(len(self.model_struct)):
            weights_layer = 1/total_samples * (np.sum([self.weights_clients[client_][layer_] * self.samples[client_]  for  client_ in sample_clients], axis=0))
            weights_agg.append(weights_layer) 
        
        self.weights_global = interpolate_weights(self.weights_global, weights_agg)
        
    
    """ SERVER: Parameter Server Based Architecture """        
    def set_param_server(self, payload):
        client_id = payload['device']
        self.samples[client_id] = payload['samples']
        for layer_ in range(len(self.model_struct)):
            self.weights_clients[client_id][layer_] = np.asarray(payload[f'model_layer{layer_}'])
        
        
    """ CLIENT: Parameter Server Based Architecture """
    def set_param_client(self, payload):
        weights_received = []
        for layer_ in range(len(self.model_struct)):
            weights_received.append(np.asarray(payload[f'global_model_layer{layer_}']))     
        self.weights_global = weights_received  
        self.model_client.set_weights(weights_received)

    
    """ CLIENT: Parameter Server Based Architecture """
    def get_param_client(self):
        detObj = {}
        weights_client = self.model_client.get_weights()

        for layer_ in range(len(self.model_struct)):
            detObj['model_layer{}'.format(layer_)] = weights_client[layer_].tolist()
        return detObj

        
    def get_param_server(self):
        detObj = {}
        weights_client = self.weights_global
        for layer_ in range(len(self.model_struct)):
            detObj['global_model_layer{}'.format(layer_)] = weights_client[layer_].tolist()
        return detObj
    
    
    """ CLIENT: Parameter Server Based Architecture """   
    def train_step(self, data_sample, masks):
        with tf.GradientTape() as tape:
            predictions = self.model_client(data_sample, training=True)
            loss = self.loss_function(masks, predictions)
        gradients = tape.gradient(loss, self.model_client.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model_client.trainable_variables))
        return loss



