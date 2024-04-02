# docker x win:
## wsl --update
### DISM /Online /Enable-Feature /All /FeatureName:Microsoft-Hyper-V

# brokers. Default broker is the public test.mosquitto.org  port 1883 (no auth implemented). 
# Other brokers
docker run -p 8080:8080 -p 1883:1883 hivemq/hivemq4
mosquitto in the folder 
docker run --rm --name hivemq-ce-test -e HIVEMQ_LOG_LEVEL=INFO -d -p 1883:1883 hivemq/hivemq-ce (use the bind address obtained at startup)

# to change the broker configure MQTT_broker_config_local.json and modify the broker address and port
### usr/app/src % contains the source files
### usr/app/src/Classes/Params/param.py % configuration file

### PARAMETER SERVER BASED FL #########
docker build -t test-docker-client_cnr -f ./docker/Dockerfile.client .
docker build -t test-docker-server_cnr -f ./docker/Dockerfile.server .

## start the server first, manually terminate the server when the FL process is completed on the clients (final model sent)
docker run test-docker-server_cnr -algorithm <from 0 to 6> -clients 3 -active_clients 3 (default algorithm 0, 3 clients, 3 active)
docker run test-docker-server_cnr -h % for help

## start each client individually with a unique ID
### note feddkw and fednove algorithms MUST use -gamma 100 option (or larger, data must be non iid)
docker run test-docker-client_cnr -ID <from 0 to xx> -algorithm <from 0 to xx> [optional: -gamma xx]
docker run test-docker-client_cnr -h % for help

# example 3 clients running fedavg
docker run test-docker-server_cnr -algorithm 0 -clients 3 -active_clients 3 # start PS for fedavg
docker run test-docker-client_cnr -ID 0 -algorithm 0 # start client 0 on fedavg
docker run test-docker-client_cnr -ID 1 -algorithm 0 # start client 1 on fedavg
docker run test-docker-client_cnr -ID 2 -algorithm 0 # start client 2 on fedavg

## CONSENSUS FL (no PS server) 
## start each client individually with a unique ID
docker build -t test-docker-consensus_cnr -f ./docker/Dockerfile.consensus .

## start a broker on the host machine (mosquitto or hivemq)  
docker run test-docker-consensus_cnr -ID <from 0 to xx> -broker_address xxx -MQTT_port xxx

## example 3 clients with broker address host.docker.internal, port 1885
docker run test-docker-consensus_cnr -ID 0 -broker_address host.docker.internal -MQTT_port 1885  
docker run test-docker-consensus_cnr -ID 1 -broker_address host.docker.internal -MQTT_port 1885
docker run test-docker-consensus_cnr -ID 2 -broker_address host.docker.internal -MQTT_port 1885


