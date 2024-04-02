# Import parameters
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from classes.algorithms.algorithms_utils import interpolate_weights, flatten_weights
from sklearn.metrics import mean_squared_error, accuracy_score
from classes.algorithms.FedAvg import FedAvg
from classes.params import fl_param, simul_param
from classes.Datasets.dataset_client import Dataset
import numpy as np
import tensorflow as tf
import math
import copy
import xgboost as xgb

class FedXGBllr(FedAvg):

    def __init__(self, model, id=None):
        super(FedXGBllr, self).__init__(model, id)
        self.name = 'FedXGBllr'
        self.trees_client = 30
        # objective = "binary:logistic"
        if id is None: # Parameter Server attribute initialization
            self.trees_stored = 0
            self.total_trees = [0 for _ in range(fl_param.NUM_CLIENTS*self.trees_client)]
            self.ensemble_sent = False
        else: # Client attribute initialization
            self.hyperparams = {
                "objective": "binary:logistic",
                "n_estimators": self.trees_client,
                "max_depth": 7,
                "learning_rate": 0.1,
                #"base_score": 0.5,  
                "random_state": 34,
            }
            self.xgb_trained = False
            self.total_trees = None
        print(f'{self.name} in use')
    
    """ CLIENT: Parameter Server Based Architecture """
    def set_param_client(self, payload):
        print('set_param_client')
        if self.xgb_trained == False:
            print('Training local XGB')
            data = Dataset(self.id)
            self._train_local_xgb(data.x_train_local, data.y_train_local)
            self.xgb_trained = True
            return
        if self.total_trees == None:
            self.total_trees = payload['final_ensemble']
            #print(self.total_trees)
            print('Received full ensemble')
        print('Setting global model')
        weights_received = []
        for layer_ in range(len(self.model_struct)):
            weights_received.append(np.asarray(payload[f'global_model_layer{layer_}']))     
        self.weights_global = weights_received  
        self.model_client.set_weights(weights_received)
    
    
    
    """ CLIENT: Parameter Server Based Architecture """ 
    def _train_local_xgb(self, x_train, y_train):
        
        self.ensemble = xgb.XGBClassifier(**self.hyperparams)
        self.ensemble.fit(x_train, y_train)
        #y_pred = reg.predict(x_valid)
        print(f'Local model accuracy: {self.ensemble.score(x_train, y_train)}')
        
        #self.trees = [booster for booster in self.ensemble.get_booster()]
    
    """ CLIENT: Parameter Server Based Architecture """ 
    def sent_local_xgb(self):# client sends his local model
        detObj = {}
        #detObj['trees'] = self.trees
        detObj['xgb'] = self.ensemble
        detObj['device'] = self.id
        return detObj
    
    
    """ SERVER: Parameter Server Based Architecture """ 
    def store_trees_ps(self, payload):# server stores the recieved local model
        client_id = payload['device']
        print(f'Storing local trees from {client_id}')
        trees = [booster for booster in payload['xgb'].get_booster()]
        #for booster in payload['xgb'].get_booster():
        #    self.total_trees[client_id*self.trees_client:(client_id+1)*self.trees_client] = booster
        self.total_trees[client_id*self.trees_client:(client_id+1)*self.trees_client] = trees
        print('done')
        self.trees_stored+=len(trees)
        print('Total trees:', self.trees_stored)
    
    """ SERVER: Parameter Server Based Architecture """ 
    def send_full_ensemble(self): # server sends the full ensemble to the client
        detObj = {}
        detObj['final_ensemble'] = self.total_trees
        self.ensemble_sent = True
        return detObj
    
    
    """ CLIENT: Parameter Server Based Architecture """ 
    def trees_predictions(self, x):
        #, base_margin=np.zeros(len(x), dtype=np.float32))
        xm = xgb.DMatrix(x)
        #print('test xgb')
        #_ = self.ensemble.predict(xm)
        #print('worked')
        #print(self.total_trees[0])
        trees_predictions = np.array(
        [booster.predict(xm) for booster in self.total_trees]
        ).T
        
        #print(trees_predictions.shape   )
        #print(trees_predictions)
        return trees_predictions

    def get_param_server(self):
        detObj = {}
        weights_client = self.weights_global
        for layer_ in range(len(self.model_struct)):
            detObj['global_model_layer{}'.format(layer_)] = weights_client[layer_].tolist()
        return detObj
    
    
    """ CLIENT: Parameter Server Based Architecture """   
    def train_step(self, data_sample, masks):
        #print('Training step')
        with tf.GradientTape() as tape:
            predictions = self.model_client(data_sample, training=True)
            #print('Predictions:', predictions)
            #print('Masks:', masks)
            loss = self.loss_function(masks, predictions)
        gradients = tape.gradient(loss, self.model_client.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model_client.trainable_variables))
        return loss
    
    
    
    
    