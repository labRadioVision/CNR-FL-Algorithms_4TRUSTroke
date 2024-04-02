MQTT installation (see mosquitto release folder and install mosquitto server)

PYTHON installation see environment folder (yaml file)

SIMPLE TEST (LOCALHOST)
python quickstart_FL_MQTT_synch.py -mosquitto_launch_command 'C:\Program Files\mosquitto2\mosquitto.exe"' -algorithm 1
usage: quickstart_FL_MQTT_synch.py [-h]
                                   [-mosquitto_launch_command MOSQUITTO_LAUNCH_COMMAND]
                                   [-algorithm ALGORITHM]
                                   [-samples_per_device SAMPLES_PER_DEVICE]
                                   [-batches BATCHES] [-batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -mosquitto_launch_command MOSQUITTO_LAUNCH_COMMAND (replace with mosquitto installation folder, example 'C:\Program Files\mosquitto2\mosquitto.exe"')
                        moqsuitto server launch command
  -algorithm ALGORITHM  choose - 0: FedAvg, 1: FedAdp, 2: FedProx, 3:
                        Scaffold, 4: FedDyn, 5: FedDkw, 6: FedNova
  -samples_per_device SAMPLES_PER_DEVICE
                        choose the max number of samples per device (default
                        for MNIST)
  -batches BATCHES      sets the number of batches per local learning round
                        (default for MNIST)
  -batch_size BATCH_SIZE
                        sets the batch size per local learning round (default
                        for MNIST)

TESTING ALL ALGORITHMS (optimized for MNIST)
python quickstart_FL_MQTT_all_algorithms.py -mosquitto_launch_command 'C:\Program Files\mosquitto2\mosquitto.exe"' -algorithm 1

usage: quickstart_FL_MQTT_all_algorithms.py [-h]
                                            [-mosquitto_launch_command MOSQUITTO_LAUNCH_COMMAND]

optional arguments:
  -h, --help            show this help message and exit
  -mosquitto_launch_command MOSQUITTO_LAUNCH_COMMAND (replace with mosquitto installation folder, example 'C:\Program Files\mosquitto2\mosquitto.exe"')
                        moqsuitto server launch command

TESTING CONSENSUS FL
python quickstart_Consensus_FL_MQTT.py -mosquitto_launch_command 'C:\Program Files\mosquitto2\mosquitto.exe"'
usage: quickstart_Consensus_FL_MQTT.py [-h]
                                       [-mosquitto_launch_command MOSQUITTO_LAUNCH_COMMAND]

optional arguments:
  -h, --help            show this help message and exit
  -mosquitto_launch_command MOSQUITTO_LAUNCH_COMMAND (replace with mosquitto installation folder, example 'C:\Program Files\mosquitto2\mosquitto.exe"')
                        moqsuitto server launch command

MAIN LEARNER SCRIPT (FL with PS)
python learner_v2.py
usage: learner_v2.py [-h] [-initialization INITIALIZATION] [-run RUN]
                     [-max_epochs MAX_EPOCHS] [-topic_PS TOPIC_PS]
                     [-topic_post_model TOPIC_POST_MODEL] [-ID ID]
                     [-local_rounds LOCAL_ROUNDS] [-target TARGET]
                     [-target_acc TARGET_ACC] [-samp SAMP] [-batches BATCHES]
                     [-batch_size BATCH_SIZE]
                     [-noniid_assignment NONIID_ASSIGNMENT] [-gamma GAMMA]

optional arguments:
  -h, --help            show this help message and exit
  -initialization INITIALIZATION
                        set 1 to resume from a previous simulation, or retrain
                        on an update dataset (continual learning), 0 to start
                        from random initialization
  -run RUN              run number
  -max_epochs MAX_EPOCHS
                        sets the max number of epochs
  -topic_PS TOPIC_PS    FL with PS topic
  -topic_post_model TOPIC_POST_MODEL
                        post models (overwritten by param.py)
  -ID ID                device/learner identifier
  -local_rounds LOCAL_ROUNDS
                        number of local rounds between two FL updates
  -target TARGET        sets the target loss to stop federation
  -target_acc TARGET_ACC
                        sets the target acc to stop federation
  -samp SAMP            sets the total number samples per device
  -batches BATCHES      sets the number of batches per local learning round
  -batch_size BATCH_SIZE
                        sets the batch size per local learning round
  -noniid_assignment NONIID_ASSIGNMENT
                        set 0 for iid assignment, 1 for non-iid random
  -gamma GAMMA          gamma for non-iid (see Dirichlet model for non-iid distribution representation)

MAIN LEARNER SCRIPT (FL without PS)
python learner_consensus.py
usage: learner_consensus.py [-h] [-initialization INITIALIZATION] [-run RUN]
                            [-max_epochs MAX_EPOCHS]
                            [-topic_post_model TOPIC_POST_MODEL] [-ID ID]
                            [-local_rounds LOCAL_ROUNDS] [-target TARGET]
                            [-target_acc TARGET_ACC] [-samp SAMP]
                            [-batches BATCHES] [-batch_size BATCH_SIZE]
                            [-noniid_assignment NONIID_ASSIGNMENT]

optional arguments:
  -h, --help            show this help message and exit
  -initialization INITIALIZATION
                        set 1 to resume from a previous simulation, or retrain
                        on an update dataset (continual learning), 0 to start
                        from the beginning
  -run RUN              run number
  -max_epochs MAX_EPOCHS
                        sets the max number of epochs
  -topic_post_model TOPIC_POST_MODEL (overwritten by param.py)
                        post models
  -ID ID                device/learner identifi er
  -local_rounds LOCAL_ROUNDS
                        number of local rounds between two FL updates
  -target TARGET        sets the target loss to stop federation
  -target_acc TARGET_ACC
                        sets the target loss to stop federation
  -samp SAMP            sets the total number samples per device
  -batches BATCHES      sets the number of batches per local learning round
  -batch_size BATCH_SIZE
                        sets the batch size per local learning round
  -noniid_assignment NONIID_ASSIGNMENT
                        set 0 for iid assignment, 1 for non-iid random


MAIN PS script (synch version)
python PS_server_synch_v2.py
usage: PS_server_synch_v2.py [-h] [-initialization INITIALIZATION]
                             [-timeout TIMEOUT] [-topic_PS TOPIC_PS]
                             [-file FILE] [-max_epochs MAX_EPOCHS]
                             [-update_factor UPDATE_FACTOR]
                             [-topic_post_model TOPIC_POST_MODEL]

optional arguments:
  -h, --help            show this help message and exit
  -initialization INITIALIZATION
                        set 1 to start from a previous initialization, 0 to
                        start without initialization, all h5 files will be
                        deleted
  -timeout TIMEOUT      MQTT loop forever timeout in sec
  -topic_PS TOPIC_PS    FL with PS topic (overwritten by param.py)
  -file FILE            filename used to save the global model
  -max_epochs MAX_EPOCHS
                        sets the max number of epochs
  -update_factor UPDATE_FACTOR
                        sets the update factor for PS (1 faster but no memory
                        of the global model, <1 slower with memory, use with
                        initialiazion = 1)
  -topic_post_model TOPIC_POST_MODEL(overwritten by param.py)
                        post model topic (do not modify)

MQTT topics:
client_weights_topic: a single topic to be used by clients posting their local model
server_weights_topic#i: PS server topics (one per device) used by the PS to post the global model to individual devices

DOCKER
docker build -t ...server... -f ./docker/....server .
docker build -t ...client... -f ./docker/....client .  
docker run ...server...
docker run ...client... -ID xx -gamma xx