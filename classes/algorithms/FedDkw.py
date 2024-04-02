### todo
# Import parameters
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from classes.algorithms.algorithms_utils import interpolate_weights, flatten_weights
from classes.Datasets.dataset_utils import kl_divergence
from classes.algorithms.FedAvg import FedAvg
from classes.params import fl_param, simul_param
import tensorflow as tf
import numpy as np
import copy

class FedDkw(FedAvg):
    def __init__(self, model, id=None):
        super(FedDkw, self).__init__(model, id)
        self.name = "FedDkw"
        self.n_classes = model.output_shape[-1]

        if id is None: # Parameter Server attribute initialization
            self.Dg = np.zeros(self.n_classes) # distribution of the global data
            self.U = []
            #self.D_clients = [0 for _ in range(fl_param.NUM_CLIENTS)]
            self.D_clients = [np.zeros(self.n_classes) for _ in range(fl_param.NUM_CLIENTS)]
            self.D_clients_t = copy.copy(self.D_clients)
        else: # Client attribute initialization
            self.D_t_i = np.zeros(self.n_classes)# labels used for training
            self.D_i = copy.copy(self.D_t_i)# labels available in the dataset
            self.sent_state = False
        print(f'{self.name} in use\n')

        

    """ SERVER: Parameter Server Based Architecture """
    def aggregate_weights(self, active_check):
        # Update weights
        sample_clients = self.clients[active_check]
        #print('self.D_clients', self.D_clients)
        self.Dg = 1 / len(self.U) * np.sum(np.array(self.D_clients), 0)
        #print('Dg', self.Dg)    
        KLR = [0 for _ in range(fl_param.NUM_CLIENTS)]
        for client_ in sample_clients:
            KLR[client_] = 1/np.sum(np.where(self.D_clients_t[client_] != 0, self.D_clients_t[client_] * np.log(self.D_clients_t[client_] / self.Dg), 0))
        
        pi = [KLR[client_]/sum(KLR) for client_ in range(fl_param.NUM_CLIENTS)]
        print('Pi', np.array(pi)[active_check])
        
        weights_agg = []
        for layer_ in range(len(self.model_struct)):
            weights_layer = (np.sum([self.weights_clients[client_][layer_] * pi[client_]  for  client_ in sample_clients], axis=0))
            weights_agg.append(weights_layer) 
        
        self.weights_global = interpolate_weights(self.weights_global, weights_agg)
        
    """ CLIENT: Parameter Server Based Architecture """
    def get_param_client(self):
        detObj = super().get_param_client()
        detObj['D_t_i'] = self.D_t_i/sum(self.D_t_i)
        self.D_t_i = np.zeros(self.n_classes)
        
        if self.sent_state == False:
            detObj['D_i'] = batch_count(self.D_i, self.n_classes)/self.D_i.shape[0]
            self.sent_state = True
        return detObj
        
    
    """ SERVER: Parameter Server Based Architecture """        
    def set_param_server(self, payload):
        super().set_param_server(payload)
        
        client_id = payload['device']
        if client_id not in self.U: # global distribution
            self.U.append(client_id)
            self.D_clients[client_id] = payload['D_i']
        self.D_clients_t[client_id] = payload['D_t_i']
        #print('D_clients[client_id]', self.D_clients[client_id])
        #print('D_clients_t[client_id]', self.D_clients_t[client_id])
        
            
            
    """ CLIENT: Parameter Server Based Architecture """   
    def train_step(self, data_sample, masks):
        loss = super().train_step(data_sample, masks)
        self.D_t_i+=batch_count(masks, self.n_classes)
        return loss


#@classmethod
def batch_count(ohe_batches, n_classes):
    
    label_counts = np.zeros(n_classes)
    if n_classes == 1:# Binary!!!!
        labels = ohe_batches.flatten()
    else:
        labels = np.argmax(ohe_batches, axis=1)
    #a = np.argmax(ohe_batches, axis=1)
    for label, count in zip(*np.unique(labels, return_counts=True)):
        label_counts[label] = count
    return label_counts