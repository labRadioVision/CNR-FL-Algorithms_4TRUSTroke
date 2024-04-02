from __future__ import division

import os
import pickle

import glob
import math
import socket
import time
import numpy as np

import argparse
import warnings
import paho.mqtt.client as mqtt
import time  # TO MEASURE THE EXECUTION
import importlib
import sys
import zlib

from classes.params import fl_param, simul_param, mqtt_param
from clients.client_utils import save_results, train_model, test_model
from clients.client_callbacks import Client_handler
from classes.Datasets.dataset_client import Dataset

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow warnings

parser = argparse.ArgumentParser()
parser.add_argument("-ID", default=0, help="device/learner identifier", type=int)
parser.add_argument('-alg', choices=['FedAvg', 'FedAdp', 'FedProx', 'Scaffold', 'FedDyn', 'FedDkw', 'FedNova', 'FedXGBllr'], default='FedXGBllr', help='FL algorithm', type=str)
parser.add_argument("-run", default=0, help="run number", type=int)
parser.add_argument('-target_loss', default=0.001, help="sets the target loss to stop federation", type=float)
parser.add_argument('-target_acc', default=0.99, help="sets the target acc to stop federation", type=float)
args = parser.parse_args() 


MODEL = getattr(importlib.import_module(simul_param.MODULE_MODEL_NAME), simul_param.MODEL_NAME)
if args.alg == 'FedXGBllr':
    MODEL = getattr(importlib.import_module('classes.models.CNN_1D'), 'CNN_1D')
ALGORITHM = getattr(importlib.import_module(simul_param.ALG_MODULE+f'.{args.alg}'), args.alg)

# check federation
if fl_param.FEDERATION:
    print("Federation enabled")
else:
    print("Federation disabled")
    
    
if fl_param.ALL_DATA:
    print('All train data is used')
else:
    print('Using the given number of batches')

if mqtt_param.COMPRESSION_ENABLED:
    print('Compression enabled')

# set device id, batch and batches
id = args.ID
run = args.run
#batch_size = fl_param.BATCH_SIZE
#number_of_batches = fl_param.NUM_BATCHES
target_loss = args.target_loss
target_acc = args.target_acc
local_epochs = fl_param.NUM_EPOCHS
rounds = fl_param.NUM_ROUNDS

print('DEVICE ', id)


# Initialize dataset
data_handle = Dataset(id)
data_handle._info()

#assert data_handle.num_samples >= batch_size*number_of_batches, "Not enough samples for training"
#number_of_batches = math.ceil(data_handle.num_samples/data_handle.batch_size)
number_of_batches = math.ceil(data_handle.num_samples/data_handle.batch_size) if fl_param.ALL_DATA else fl_param.NUM_BATCHES
batch_size = data_handle.batch_size

print("Number of batches for learning {}, size {}:".format(number_of_batches, batch_size))


input_shape, output_shape = data_handle.return_input_output()
print(f'input shape:{input_shape}, output shape:{output_shape}')
model = MODEL(input_shape, output_shape).return_model()  # init model structure
num_layers = len(model.get_weights())

# Initialize algorithm
algorithm = ALGORITHM(model, id)


detObj = {}


# -------------------------    MAIN   -----------------------------------------

if __name__ == "__main__":
    print(simul_param.ARCHITECTURE[0], "\n")
    #path_results = "results/matlab"
    base_path = 'output/' + mqtt_param.config["Name_of_federation"]+f'/client{id}' #os.path.join(os.getcwd(), 'results')
    #dir = f'data/client_{i}/{type}/'
    results_path = base_path + '/results'
    models_path = base_path + '/models'
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    
    AWAKEN = False  # train for 0 model or after you did at least one training before not to lose the statistic

    client_py = "Client" + str(id)
    PS_mqtt_topic = mqtt_param.server_weights_topic + str(id)  # Client receives the global model published on this topic
    device_topic = mqtt_param.client_weights_topic
    mqttc = mqtt.Client(client_id=client_py, clean_session=True)
    
    handler = Client_handler(mqttc, model, algorithm, id)
    mqttc.on_publish = handler.on_publish
    mqttc.on_disconnect = handler.on_disconnect
    
    metr_names = ['Loss', 'Acc', 'Prec', 'Rec', 'AUC']
    
    metrics = {}
    for name in metr_names:
        metrics[name] = []
    metrics['Rounds'] = []

    start_time = time.time()
    mqttc.connect(host=mqtt_param.ADDRESS, port=mqtt_param.PORT, keepalive=60)
    mqttc.subscribe(PS_mqtt_topic, qos=mqtt_param.QOS)
    # auth = AUTH_, tls = TLS_,
    # protocol = mqtt.MQTTv311
    mqttc.message_callback_add(PS_mqtt_topic, handler.on_message_from_ps)
    mqttc.loop_forever()
    
    if algorithm.name == 'FedXGBllr':
        mqttc.connect(host=mqtt_param.ADDRESS, port=mqtt_param.PORT, keepalive=60)
        mqttc.subscribe(PS_mqtt_topic, qos=mqtt_param.QOS)
        mqttc.message_callback_add(PS_mqtt_topic, handler.on_message_from_ps)
        print('Sending local trees\n')
        
        detObj = algorithm.sent_local_xgb()
        if mqtt_param.COMPRESSION_ENABLED:
            res=mqttc.publish(device_topic, zlib.compress(pickle.dumps(detObj)), qos=mqtt_param.QOS, retain=False)
        else:
            res=mqttc.publish(device_topic, pickle.dumps(detObj), qos=mqtt_param.QOS, retain=False)
        print('sent')
        #mqttc.loop(2.0)
        
        # wait for the global ensemble
        print("waiting for ensemble and init model...")
        mqttc.connect(host=mqtt_param.ADDRESS, port=mqtt_param.PORT, keepalive=60)
        mqttc.subscribe(PS_mqtt_topic, qos=mqtt_param.QOS)
        mqttc.message_callback_add(PS_mqtt_topic, handler.on_message_from_ps)
        mqttc.loop_forever()
        
        print('THIS OVERHEAD IS FINALLY OVER, OMG, I CANT BELIEVE THAT')
        
        # convert data to trees outputs
        print('Converting data to trees outputs')
        data_handle.x_train_local = np.expand_dims(algorithm.trees_predictions(data_handle.x_train_local), axis=-1)
        data_handle.x_valid = np.expand_dims(algorithm.trees_predictions(data_handle.x_valid), axis=-1)
    while True:
        if AWAKEN:          
            if not handler.training_end:# and not skip_train:
                print('Starting training locally')
                start_train = time.time()
                train_model(algorithm, data_handle, number_of_batches)
            
                stop_train = time.time()
                print('Trained in {:.2f} seconds\n'.format(stop_train - start_train))
                #model.save(models_path+f'/model_{id}_run_{run}.h5')
            ####################################################### TX A MODEL
            detObj = handler.prepare_payload_client(id, data_handle.num_samples)
            print('training_end', handler.training_end)

            # encryption
            # mqttc.tls_set(ca_certs=ca_certificate, certfile=client_certificate, keyfile=client_certificate)
            # mqttc.tls_insecure_set(False)
            rc = mqttc.connect(host=mqtt_param.ADDRESS, port=mqtt_param.PORT, keepalive=60)
            if rc != mqtt.MQTT_ERR_SUCCESS:
                try:
                    time.sleep(0.5)
                    print("reconnecting")
                    mqttc.reconnect()
                except (socket.error, mqtt.WebsocketConnectionError):
                    pass
            
            if handler.training_end:
                print("final model")
                # qos = 1 is needed as the client has problems for qos >= 2 for the final message
                if mqtt_param.COMPRESSION_ENABLED:
                    mqttc.publish(device_topic, zlib.compress(pickle.dumps(detObj)), qos=1, retain=True)
                else:
                    mqttc.publish(device_topic, pickle.dumps(detObj), qos=1, retain=True)
                mqttc.disconnect()
                time.sleep(2)
                break
            
            mqttc.subscribe(PS_mqtt_topic, qos=mqtt_param.QOS)
            mqttc.message_callback_add(PS_mqtt_topic, handler.on_message_from_ps)
            print('Sending the model\n')
            if mqtt_param.COMPRESSION_ENABLED:
                mqttc.publish(device_topic, zlib.compress(pickle.dumps(detObj)), qos=mqtt_param.QOS, retain=True)
            else:
                mqttc.publish(device_topic, pickle.dumps(detObj), qos=mqtt_param.QOS, retain=True)
            #size_TX_model = sys.getsizeof(zlib.compress(pickle.dumps(detObj)))
            #size_TX_model_uc = sys.getsizeof(pickle.dumps(detObj))
            #print('mqtt payload size (uncompressed) in B: {}, KB: {}, MB: {}'.format(size_TX_model_uc, size_TX_model_uc / 10 ** 3, size_TX_model_uc / 10 ** 6))
            #print('mqtt payload size (zlib compressed) in B: {}, KB: {}, MB: {}'.format(size_TX_model, size_TX_model / 10 ** 3, size_TX_model / 10 ** 6))
            if fl_param.FEDERATION:
                print("waiting for updates...")
                mqttc.loop_forever(retry_first_connection=True)
            else:
                mqttc.disconnect()
        ##############################################################################################
        print('Testing the received model')
        start_test = time.time()
        test_out = test_model(algorithm, data_handle)
        stop_test = time.time()

        for name, metric in zip(metr_names, test_out):
            metrics[name].append(metric)
        metrics['Rounds'].append(handler.round)
        #print(metrics['Rounds'])
        print(f'ID:{id} | Round={handler.round}') 
        print(f'Loss {test_out[0]:.2f} Acc {test_out[1]:.2f} Prec {test_out[2]:.2f} Rec {test_out[3]:.2f} AUC {test_out[4]:.2f}')
        print('Tested in {:.2f} seconds\n'.format(stop_test - start_test))
        
        running_loss = np.mean(metrics['Loss'][-1:])
        running_acc = np.mean(metrics['Acc'][-1:])
        AWAKEN = True
        
        if running_loss <= target_loss or running_acc >= target_acc or handler.round >= rounds:
            stop_time = time.time()
            message = "Solved" if (running_loss <= target_loss or running_acc >= target_acc) else "Unsolved"
            handler.training_end = True
            sim_time = stop_time - start_time
            print(f'{message} for device {id} at round {handler.round}!')
            save_results(algorithm.model_client, metrics, id, run, algorithm.name, results_path, models_path)

