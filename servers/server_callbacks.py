import pickle
import time
import socket
import zlib
import os
import numpy as np
import paho.mqtt.client as mqtt
from classes.params import mqtt_param, fl_param, simul_param

class Server_handler:
    def __init__(self, mqttc, model, algorithm, scheduler):
        self.model_global = model
        self.num_layers = len(model.get_weights())
        self.algorithm = algorithm
        self.round = 0
        self.training_end = False
        self.mqttc = mqttc
        self.scheduler = scheduler
        
        self.base_path = 'output/'+ mqtt_param.config["Name_of_federation"]+'/server'
        os.makedirs(self.base_path, exist_ok=True)
        
    def PS_callback(self, client, userdata, message):
        if mqtt_param.COMPRESSION_ENABLED:
            st = pickle.loads(zlib.decompress(message.payload))  # rx message and uncompress
        else:
            st = pickle.loads(message.payload)
        rx_dev = st['device']
        
        if self.algorithm.name == 'FedXGBllr' and self.algorithm.trees_stored<=(fl_param.NUM_CLIENTS-1)*self.algorithm.trees_client:# or (self.algorithm.name == 'FedXGBllr':#: and payload['round'] >=):
            print('from', rx_dev, 'trees received')
            
            self.algorithm.store_trees_ps(st)
            if self.algorithm.trees_stored != fl_param.NUM_CLIENTS*self.algorithm.trees_client:
                return
            print('received all trees')
                
        if self.algorithm.name == 'FedXGBllr' and not self.algorithm.ensemble_sent:
            print('sending ensemble and the model')
            ens = self.algorithm.send_full_ensemble()
            detObj= self.prepare_payload_server() 
            detObj.update(ens)
            for i in range(fl_param.NUM_CLIENTS):
                if mqtt_param.COMPRESSION_ENABLED:
                    self.mqttc.publish(mqtt_param.server_weights_topic + str(i), zlib.compress(pickle.dumps(detObj)), qos=mqtt_param.QOS, retain=False)
                else:
                    self.mqttc.publish(mqtt_param.server_weights_topic + str(i), pickle.dumps(detObj), qos=mqtt_param.QOS, retain=False)    
            print('sent')
            return
        
        print("received from learner {}".format(rx_dev))
        print('training_end', st['training_end'])
        
        if st['training_end']:  # WE ARE DONE -> SAVE THE MODEL OF A CLIENT, WHIH THE BEST PERFORMANCE
            self.training_end = True
            self.algorithm.set_param_server(st)
            self.algorithm.weights_global = self.algorithm.weights_clients[rx_dev]  # TODO: CHANGE, BECAUSE TECHNICALLY THIS MODEL WAS SENT BEFORE
            
        if self.scheduler.selected(rx_dev) and not self.training_end:
            # received a local model from a learner in the scheduling_tx list
            self.scheduler._update_received(rx_dev)
            ######################## payload extraction########################
            self.algorithm.set_param_server(st)
            ######################## payload extraction########################
            
        if self.scheduler.received_all() or self.training_end:
            # prepare the global model update
            if not self.training_end:  # update according to the algorithm
                print('Updating global model')
                self.algorithm.aggregate_weights(self.scheduler.active_check)
            self.round += 1  # increase round
            print('Global round {}, check training end: {}\n'.format(self.round, self.training_end))
            
            detObj = self.prepare_payload_server()            
            
            if self.training_end:
                self.round -= 1  # we increase before in the code -> it is not an epoch, just sending
                self.model_global.set_weights(self.algorithm.weights_global)
                self.model_global.save(self.base_path+f'/global_model_{self.algorithm.name}_final.h5')
                # disconnect
                rc = self.mqttc.disconnect()
                if rc != mqtt.MQTT_ERR_SUCCESS:
                    try:
                        time.sleep(0.5)
                        print("attempting to disconnect failed")
                        self.mqttc.disconnect()
                    except (socket.error, mqtt.WebsocketConnectionError):
                        pass
            else:  # send the global model
                client_ids = self.scheduler.select_clients(self.round)
                print('Sending to:', client_ids.flatten())
                for i in client_ids:
                    if mqtt_param.COMPRESSION_ENABLED:
                        self.mqttc.publish(mqtt_param.server_weights_topic + str(i), zlib.compress(pickle.dumps(detObj)), qos=mqtt_param.QOS, retain=False)
                    else:
                        self.mqttc.publish(mqtt_param.server_weights_topic + str(i), pickle.dumps(detObj), qos=mqtt_param.QOS, retain=False)
                #self.model_global.save(self.base_path+f'/model_{simul_param.ALGORITHM_NAME}_{self.round}.h5')
                
    def prepare_payload_server(self):
        if self.algorithm.name == 'FedXGBllr' and self.algorithm.trees_stored != fl_param.NUM_CLIENTS*self.algorithm.trees_client:
            print('sending nothing')
            return {}    
        detObj = self.algorithm.get_param_server()
        # store other simulation param
        detObj['round'] = self.round
        detObj['training_end'] = self.training_end
        return detObj    
        
    @staticmethod
    def on_publish(client, userdata, result):  # create function for callback
        print("data published \n")
        time.sleep(0.5)

    def on_disconnect(self, client, userdata, rc):
        print('disconnecting')
        
        if self.training_end:
            # prepare session closure
            client_py = "PS"
            mqttc = mqtt.Client(client_id=client_py, clean_session=False)
            mqttc.connect(host=mqtt_param.ADDRESS, port=mqtt_param.PORT, keepalive=20)
            
            detObj = self.prepare_payload_server()
            
            t_end = time.time() + 10 # stay active for 20 seconds
            print("Broadcasting the finalized model")
            while time.time() < t_end:
                for i in range(fl_param.NUM_CLIENTS):
                    if mqtt_param.COMPRESSION_ENABLED:
                        mqttc.publish(mqtt_param.server_weights_topic + str(i), zlib.compress(pickle.dumps(detObj)), qos=1, retain=True)
                    else:
                        mqttc.publish(mqtt_param.server_weights_topic + str(i), pickle.dumps(detObj), qos=1, retain=True)
                    time.sleep(0.5)
                time.sleep(4)

            print("disconnected \n")
        else:
            print("Unexpected MQTT disconnection")
            
    def start_server(self):
        client_ids = self.scheduler.select_clients(self.round)
        detObj = self.prepare_payload_server()
        if self.algorithm.name == 'FedXGBllr' and self.algorithm.trees_stored != fl_param.NUM_CLIENTS*self.algorithm.trees_client:
            print('INIT')
            client_ids = np.array(range(fl_param.NUM_CLIENTS)) # ask everyone for the trees 
        print('Sending to:', client_ids.flatten())
        for i in client_ids:
            if mqtt_param.COMPRESSION_ENABLED:
                self.mqttc.publish(mqtt_param.server_weights_topic + str(i), zlib.compress(pickle.dumps(detObj)), qos=mqtt_param.QOS, retain=False)
            else:
                self.mqttc.publish(mqtt_param.server_weights_topic + str(i), pickle.dumps(detObj), qos=mqtt_param.QOS, retain=False)
            #training_end_signal = False  # to do: reactivate the server"""
