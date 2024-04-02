from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.base import clone
import numpy as np
import random
from classes.Datasets.Stroke import Stroke_kaggle
from sklearn.metrics import accuracy_score
# predicted = rf.predict(X_test)
# accuracy = accuracy_score(y_test, predicted)
from sklearn.metrics import confusion_matrix
import pandas as pd
import argparse
import warnings
import joblib
import json
import paho.mqtt.client as mqtt
import datetime
import pickle
from classes.params.param import *

# MQTT
MQTT_broker = broker_address
MQTT_port = MQTT_port

# DEVICES
devices = NUM_CLIENTS  # TOTAL CLIENTS
active = NUM_CLIENTS

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('-timeout', default=1, help="MQTT loop forever timeout in sec", type=int)
parser.add_argument("-topic_PS", default=server_weights_topic, help="FL with PS topic", type=str)
parser.add_argument("-file", default="results/model_global.h5", help="filename used to save the global model", type=str)
parser.add_argument("-qos", default=2, help="qos level", type=int)
parser.add_argument("-topic_post_model", default=client_weights_topic, help="post model topic (do not modify)", type=str)
args = parser.parse_args()

active_check = np.zeros(devices, dtype=bool)  # initialize active_check
scheduling_tx = np.zeros((devices, 1), dtype=int)  # initialize scheduling table
local_trees = 5
rand_trees = 4
leaf = 6
rf_combined =  RandomForestClassifier(n_estimators=local_trees, min_samples_leaf=leaf)
rf_combined.estimators_ = []
# local_trees = 1
# rand_trees = 1  # number of random trees to be considered during aggregation (on the PS)
# leaf = 6 # min samples in every leaf
# fed_train_size = 100 # max 150, number of training sample per client
#


# aggregate random forests with random sampling of trees (might be implemented on the PS)
def fed_avg_rfs(rf_global, rf_local_param, rand_trees):
    local_trees_sampling = random.sample(range(0, local_trees), rand_trees) # random subset of the trees
    rf_local_sel = clone(rf_global)
    rf_local_sel.estimators_ = [] # create an empty tree list
    for k in range(rand_trees):
        rf_local_sel.estimators_.append(rf_local_param[local_trees_sampling[k]]) # append a random subset of the trees
    rf_global.estimators_ += rf_local_sel.estimators_ # merge with global
    rf_global.n_estimators = len(rf_global.estimators_)
    return rf_global

# simply combine a pair of random forests without sampling (general procedure, might be implemented on the PS)
# def combine_rfs(rf_a, rf_b):
#     rf_a.estimators_ += rf_b.estimators_
#     rf_a.n_estimators = len(rf_a.estimators_)
#     return rf_a

# test with stroke database

# the combined model scores better than *most* of the component models
# (notice that it seems that the global tree ìs only better than most of local trees, but not always the best one)

def PS_callback(client, userdata, message):
    global mqttc
    global training_end, train_start
    global checkpointpath2
    global epoch_count
    global rf_combined
    global scheduling_tx
    global active_check
    global counter

    st = pickle.loads(message.payload)
    rx_dev = st['device']
    print("received from learner {}".format(rx_dev))
    detObj = {}

    if scheduling_tx[rx_dev, 0] == 1 and not training_end:
        # received a local model from a learner in the scheduling_tx list
        # updating the WEIGHTS_CLIENTS
        if not active_check[rx_dev]:
            counter += 1
            active_check[rx_dev] = True
        if counter == 1:
            rf_combined.estimators_ = st['rf_model']
        else:
            rf_combined = fed_avg_rfs(rf_combined,st['rf_model'], rand_trees) # change with the correct algorithm

    if (counter == active) or (counter == devices) or training_end:
        # prepare the global model update
        # if not training_end:  # update according to the algorithm
        #     print('Updating global model')
        #     algorithm.update_weights_Server(epoch_count, NUM_CLIENTS, update_factor, num_layers, active_check)
        #     model_global.set_weights(algorithm.WEIGHTS_GLOBAL)
        epoch_count += 1  # increase epoch
        active_check = np.zeros(devices, dtype=bool)  # reset active checks
        counter = 0  # reset counters
        device_indices = list(range(0, devices))
        random.shuffle(device_indices)
        inds = np.array(device_indices[:active])  # LIST OF CLIENTS TO TX TO
        scheduling_tx = np.zeros((devices, 1), dtype=int)
        scheduling_tx[inds, 0] = 1

        detObj['global_rf_model'] = rf_combined.estimators_
        detObj['estimators'] = rf_combined.n_estimators
        detObj['global_epoch'] = epoch_count
        detObj['random_trees'] = rand_trees
        detObj['training_end'] = training_end


        # ONCE THE MODELS ARE AGGREGATED!
        print('Global epoch count {}, check training end: {}\n'.format(epoch_count, training_end))
        print('Sending to:', inds.flatten())
        for i in inds:
            mqttc.publish(PS_mqtt_topic + str(i), pickle.dumps(detObj), qos=args.qos, retain=False)
        joblib.dump(rf_combined,checkpointpath2, compress=0)
        rf_combined = RandomForestClassifier(n_estimators=local_trees, min_samples_leaf=leaf)
        rf_combined.estimators_ = []



if __name__ == "__main__":
    training_end = False  # checking the end of training
    counter = 0
    client_py = "PS"
    mqttc = mqtt.Client(client_id=client_py, clean_session=False)
    mqttc.connect(host=MQTT_broker, port=MQTT_port, keepalive=20)
    PS_mqtt_topic = args.topic_PS
    # mqttc.on_publish = on_publish
    # mqttc.on_disconnect = on_disconnect

    device_topic = args.topic_post_model
    mqttc.subscribe(device_topic, qos=args.qos)
    mqttc.message_callback_add(device_topic, PS_callback)
    checkpointpath1 = 'results/model_global.h5'
    checkpointpath2 = 'results/global_model_dump.ckpt'

    epoch_count = 0

    # Initialize filesystem

    train_start = True
    SERVER_WAITS = True
    print("start PS\n")


    scheduling_tx = np.ones((devices, 1), dtype=int) # wait for all of the clients
    # print(scheduling_tx)
    mqttc.loop_forever()
