from __future__ import division
import time
import argparse
import warnings
import json
import paho.mqtt.client as mqtt
import importlib
from classes.params import fl_param, simul_param, mqtt_param
from servers.server_callbacks import Server_handler
from servers.server_utils import Scheduler

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('-alg', choices=['FedAvg', 'FedAdp', 'FedProx', 'Scaffold', 'FedDyn', 'FedDkw', 'FedNova', 'FedXGBllr'], default='FedXGBllr', help = 'FL algorithm', type=str)
parser.add_argument('-run', default=0, help="run number", type=int)
args = parser.parse_args()


with open('data/server/data_info.json', 'r') as openfile:
    # Reading from json file
    data_info = json.load(openfile)

####################################### MODEL and ALGORITHM
MODEL = getattr(importlib.import_module(simul_param.MODULE_MODEL_NAME), simul_param.MODEL_NAME)
model_global = MODEL(data_info["input_shape"], data_info["num_classes"]).return_model()
num_layers = len(model_global.get_weights())

ALGORITHM = getattr(importlib.import_module(simul_param.ALG_MODULE+f'.{args.alg}'), args.alg)
algorithm = ALGORITHM(model_global)
##############################################
print("PS warmup: {} devices out of {} active per round. PS update factor: {}".format(fl_param.CLIENTS_SELECTED, fl_param.NUM_CLIENTS, fl_param.EPSILON_GLOBAL))

if fl_param.SHED_TYPE == 'Robin':
    print('Round robin client selection\n')
elif fl_param.SHED_TYPE == 'Rand':
    print('Random client selection\n')
########################################################################


# -------------------------    MAIN   -----------------------------------------
if __name__ == "__main__":
    client_py = "PS"
    mqttc = mqtt.Client(client_id=client_py, clean_session=False)
    # encryption
    # mqttc.tls_set(ca_certs=ca_certificate, certfile=client_certificate, keyfile=client_certificate)
    # mqttc.tls_insecure_set(False)
    mqttc.connect(host=mqtt_param.ADDRESS, port=mqtt_param.PORT, keepalive=20)
    device_topic = mqtt_param.client_weights_topic
    mqttc.subscribe(device_topic, mqtt_param.QOS)
    
    print("start PS\n")
    time.sleep(7)

    scheduler = Scheduler()
    handler = Server_handler(mqttc, model_global, algorithm, scheduler)
    mqttc.on_publish = handler.on_publish
    mqttc.on_disconnect = handler.on_disconnect
    mqttc.message_callback_add(device_topic, handler.PS_callback)
    handler.start_server()
    mqttc.loop_forever()
