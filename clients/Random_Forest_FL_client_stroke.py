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
import scipy.io as sio
import copy
import argparse
import warnings
import joblib
import json
import paho.mqtt.client as mqtt
import datetime
import pickle
from classes.params.param import *

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("-run", default=0, help="run number", type=float)
parser.add_argument("-topic_PS", default=server_weights_topic, help="FL with PS topic", type=str)
parser.add_argument("-topic_post_model", default=client_weights_topic, help="post models", type=str)
parser.add_argument("-ID", default=0, help="device/learner identifier", type=int)
parser.add_argument("-qos", default=2, help="qos level", type=int)
parser.add_argument('-samp', default=200, help="sets the total number samples per device", type=int)
parser.add_argument('-gamma', default=0.5, help=" gamma for non-iid", type=float)
args = parser.parse_args()



# clients = 10
local_trees = 5
# rand_trees = 4  # number of random trees to be considered during aggregation (on the PS)
leaf = 6 # min samples in every leaf
# fed_train_size = 100 # max 150, number of training sample per client


# set device id, batch and batches
device_index = args.ID

gamma = args.gamma  # for non-iid

print('DEVICE ', device_index)

training_set_per_device = args.samp  # NUMBER OF TRAINING SAMPLES PER DEVICE
start_index = device_index * training_set_per_device# REDUNDANT
data_stroke = Stroke_kaggle(device_index, start_index, training_set_per_device)

# generate a single small RF (with 5 trees)
def generate_rf(X_train, y_train, X_test, y_test, i):
    rf = RandomForestClassifier(n_estimators=local_trees, min_samples_leaf=leaf)
    rf.fit(X_train, y_train)
    predicted = rf.predict(X_test)
    # p = rf.fit(X_train, y_train).predict_proba(X_train)
    cm = pd.DataFrame(confusion_matrix(y_test, predicted)).to_numpy()
    TPR = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    TNR = cm[1,1]/(cm[1,0]+cm[1,1])
    ACC = accuracy_score(y_test, predicted)
    SCORE = rf.score(X_test, y_test)
    print("Node {}: Local model score {}, accuracy {}, TPR {}, TNR {} ".format(i, SCORE, ACC, TPR, TNR))
    return rf, TNR, TPR, ACC, SCORE

def PS_callback(client, userdata, message):
    global data_stroke
    global rfs
    global mqttc
    global checkpointpath1
    global rf_combined, random_trees
    st = pickle.loads(message.payload)  # rx message

    rf_combined = copy.deepcopy(rfs)
    # rf_combined.estimators_ = st['global_rf_model']
    rf_global_trees = st['global_rf_model']
    # the combined model scores better than *most* of the component models
    # (notice that it seems that the global tree ìs only better than most of local trees, but not always the best one)
    n_est = st['estimators']
    rf_combined.estimators_ = rf_global_trees
    rf_combined.n_estimators = n_est
    random_trees = st['random_trees']
    rc = mqttc.disconnect()
    if rc != mqtt.MQTT_ERR_SUCCESS:
        try:
            time.sleep(0.5)
            print("attempting to disconnect failed")
            mqttc.disconnect()
        except (socket.error, mqtt.WebsocketConnectionError):
            pass

# aggregate random forests with random sampling of trees (might be implemented on the PS)
def fed_avg_rfs(rf_global, rf_local, rand_trees):
    local_trees_sampling = random.sample(range(0, local_trees), rand_trees) # random subset of the trees
    rf_local_sel = clone(rf_local)
    rf_local_sel.estimators_ = [] # create an empty tree list
    for k in range(rand_trees):
        rf_local_sel.estimators_.append(rf_local.estimators_[local_trees_sampling[k]]) # append a random subset of the trees
    rf_global.estimators_ += rf_local_sel.estimators_ # merge with global
    rf_global.n_estimators = len(rf_global.estimators_)
    return rf_global

# simply combine a pair of random forests without sampling (general procedure, might be implemented on the PS)
def combine_rfs(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a

# test with stroke database
if __name__ == "__main__":
    # create the locla random forest classifier model
    training_end = False
    print("RANDOM FOREST FEDERATION", "\n")
    checkpointpath1 = 'results/RFmodel{}.h5'.format(device_index)
    outfile = 'results/RFdump_train_variables{}.npz'.format(device_index)
    outfile_models = 'results/RFdump_train_model{}.npy'.format(device_index)
    checkpointpath2 = 'results/RFlocal_model_{}_dump.ckpt'.format(device_index)
    rf_combined = []
    # n_est = 0
    random_trees = 0
    rfs, TNR, TPR, ACC, SCORE = generate_rf(data_stroke.x_train, data_stroke.y_train.to_numpy(), data_stroke.x_test, data_stroke.y_test.to_numpy(),device_index)
    rf_parameters = rfs.estimators_
    joblib.dump(rfs,checkpointpath2, compress=0)

    #
    #
    # if args.initialization == 0:  # clear all files
    #     # DELETE TEMPORARY CACHE FILES
    #     fileList = glob.glob(checkpointpath1, recursive=False)
    #     for filePath in fileList:
    #         try:
    #             os.remove(filePath)
    #         except OSError:
    #             print("Error while deleting file")
    #
    # # check for backup variables on start
    # if args.initialization == 1:  # clear all files
    #     print("Initializing using saved local model")
    #     loaded_rf = joblib.load(checkpointpath2) # to be completed

    # transmit over mqtt
    client_py = config["Name_of_federation"] + "learner " + str(device_index)
    PS_mqtt_topic = args.topic_PS + str(device_index)  ####################################################### NEW
    mqttc = mqtt.Client(client_id=client_py, clean_session=True)
    #  mqttc.tls_set(ca_certs='./broker.emqx.io-ca.crt')
    #  mqttc.on_publish = on_publish
    rc = mqttc.connect(host=broker_address, port=MQTT_port, keepalive=60)

    if rc != mqtt.MQTT_ERR_SUCCESS:
        try:
            time.sleep(0.5)
            print("reconnecting")
            mqttc.reconnect()
        except (socket.error, mqtt.WebsocketConnectionError):
            pass

    mqttc.subscribe(PS_mqtt_topic, qos=args.qos)
    mqttc.message_callback_add(PS_mqtt_topic, PS_callback)
    print('Sending the model\n')
    detObj = {}
    detObj['rf_model'] = rf_parameters
    detObj['device'] = device_index
    detObj['samples'] = training_set_per_device
    detObj['training_end'] = training_end
    detObj['TNR'] = TNR
    detObj['TPR'] = TPR
    detObj['ACC'] = ACC
    detObj['SCORE'] = SCORE

    mqttc.publish(args.topic_post_model, pickle.dumps(detObj), qos=args.qos, retain=True)
    time.sleep(3)

    if training_end:
        print("final model sent")
        time.sleep(3)
        mqttc.disconnect()

    if FEDERATION:
        print("waiting for updates...")
        # mqttc.loop_forever(retry_first_connection=True)
        mqttc.loop_forever()
    else:
        mqttc.disconnect()

    #rf_combined = copy.deepcopy(rfs)
    #rf_combined.estimators_ = rf_global_trees
    #rf_combined.n_estimators = n_est
    predicted_global = rf_combined.predict(data_stroke.x_test)
    cm = pd.DataFrame(confusion_matrix(data_stroke.y_test, predicted_global)).to_numpy()
    print("Node {}: Global model score {}, accuracy {}, TPR {}, TNR {} ".format(device_index,
                                                                                rf_combined.score(data_stroke.x_test,
                                                                                                  data_stroke.y_test),
                                                                                accuracy_score(data_stroke.y_test,
                                                                                               predicted_global),
                                                                                cm[0, 0] / (cm[0, 0] + cm[0, 1]),
                                                                                cm[1, 1] / (cm[1, 0] + cm[1, 1])))
    joblib.dump(rf_combined, checkpointpath1, compress=0)
    dict_1 = {"global_score": rf_combined.score(data_stroke.x_test,data_stroke.y_test),
              "global_accuracy":  accuracy_score(data_stroke.y_test,predicted_global),
              "global_TPR": cm[0, 0] / (cm[0, 0] + cm[0, 1]),
              "global_TNR": cm[1, 1] / (cm[1, 0] + cm[1, 1]),
               "local_TNR": TNR,
               "local_TPR": TPR,
               "local_ACC": ACC,
               "local_SCORE": SCORE,
    }
    sio.savemat(
        "results/matlab/Device_{}_rand_trees_{}_run_{}.mat".format(
            device_index, random_trees, args.run), dict_1)

    ##############################################################################################

# # combine the RF models, so combine the list of random forest models into one "global" model
# rf_combined = rfs[0]
# for i in range(1, clients):
#     rf_combined = fed_avg_rfs(rf_combined, rfs[i], rand_trees)
#
# # the combined model scores better than *most* of the component models
# # (notice that it seems that the global tree ìs only better than most of local trees, but not always the best one)
# for i in range(1, clients):
#     predicted_global = rf_combined.predict(data_stroke[i].x_test)
#     cm = pd.DataFrame(confusion_matrix(data_stroke[i].y_test, predicted_global)).to_numpy()
#     print("Node {}: Global model score {}, accuracy {}, TPR {}, TNR {} ".format(i,
#                                                                                 rf_combined.score(data_stroke[i].x_test,
#                                                                                                   data_stroke[
#                                                                                                       i].y_test),
#                                                                                 accuracy_score(data_stroke[i].y_test,
#                                                                                                predicted_global),
#                                                                                 cm[0, 0] / (cm[0, 0] + cm[0, 1]),
#                                                                                 cm[1, 1] / (cm[1, 0] + cm[1, 1])))
