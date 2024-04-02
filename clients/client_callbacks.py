import pickle
import numpy as np
import sys
import zlib
import _pickle as cPickle
import time
import zlib
import socket
import warnings
import paho.mqtt.client as mqtt
from classes.params import simul_param, mqtt_param
import zlib
from queue import Queue

class Client_handler:
    def __init__(self, mqttc, model, algorithm, id):
        #self.model = model
        self.id = id
        self.num_layers = len(model.get_weights())
        self.algorithm = algorithm
        self.round = 0
        self.training_end = False
    
        
    def on_message_from_ps(self, client, userdata, message):
        if mqtt_param.COMPRESSION_ENABLED:
            payload = pickle.loads(zlib.decompress(message.payload))  # rx message and uncompress
        else:
            payload = pickle.loads(message.payload)  # rx message
        
        #print('aaaaaaaaaaaaaaaaaaaaaaaaaaaa', self.algorithm.total_trees != None)
        if self.algorithm.name != 'FedXGBllr' or (self.algorithm.name == 'FedXGBllr' and self.algorithm.total_trees != None):
            self.round = payload['round']
            print('Received global ' + str(self.round) + '\n')
            
            if payload['training_end']:
                self.training_end = True
            
        self.algorithm.set_param_client(payload)
        
        rc = client.disconnect()
        if rc != mqtt.MQTT_ERR_SUCCESS:
            try:
                time.sleep(0.5)
                print("attempting to disconnect failed")
                client.disconnect()
            except (socket.error, self.mqtt.WebsocketConnectionError):
                pass
            
    def prepare_payload_client(self, device_index, num_samples):
        # store the algorithm parameters
        detObj = self.algorithm.get_param_client()
        # store other simulation param
        detObj['device'] = device_index
        detObj['round'] = self.round
        detObj['samples'] = num_samples
        detObj['training_end'] = self.training_end
        return detObj 
    
    @staticmethod        
    def on_publish(client, userdata, result):  # create function for callback
        print("data published \n")
        time.sleep(0.5)

    @staticmethod 
    def on_disconnect(client, userdata, result):  # create function for callback
        print("disconnected \n")