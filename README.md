## TRUSTroke FL platform

FL platform for TRUSTroke project and research.

### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages (python >=3.7).

```bash
pip install -r requirements.txt
```

### Conda Environment
Install the conda environment selecting the correct OS from the folder Environments):

```
conda env create -f environment_windows.yml
```


### TEST algorithms
```
MASTER BRANCH - SUPPORTED algorithms=("FedAvg" "FedAdp" "FedProx" "Scaffold" "FedDyn" "FedDkw" "FedNova")
FEDXGBOOST BRANCH - SUPPORTED algorithms=("FedAvg" "FedAdp" "FedProx" "Scaffold" "FedDyn" "FedDkw" "FedNova" "FedXGBllr")
```
### Federated data: create a federated dataset and distribute to clients 
```python
python -m classes.Datasets.data_generator -data $data -samples $samples -data $data -niid $n_iid -alpha 0.1
$data choices=['mnist', 'stroke'], default='stroke', help = 'Type of data',
$samples default=100, help="sets a fixed number samples per device"
$n_iid choices=['iid', 'label', 'sample', 'feature'], default='iid', help="Heterogeneity type"
NOTE: stroke dataset can be shared upon request
```

### MQTT
```
MQTT installation (example: see mosquitto release folder and install mosquitto server)
For localhost configuration use:
[text](MQTT_broker_config_local.json)
For remote testing (test.mosquitto.org) - unencrypted and unauthenticated:
[text](MQTT_broker_config.json)
```
### MQTT BROKER
```
START BROKER: optionally use:
python Launcher_mosquitto.py 
```

### Clients, SIMPLE STARTUP
---
In the [classes/params/fl_param.py] file, set the number of clients in NUM_CLIENTS and the number of clients to be selected at each round by the Server in CLIENTS_SELECTED.
In the [classes/params/fl_param.py] file, set the maximum number of FL rounds in NUM_ROUNDS (example 50 rounds)
In the [classes/params/simul.param.py] set the ML model parameter in model_id
MODEL = TYPES_MODELS[model_id] according to
```python
TYPES_MODELS = {
    0: ['classes.models.CNN', 'CNN'],
    1: ['classes.models.VanillaMLP', 'VanillaMLP'],
    2: ['classes.models.Mobinet','Mobinet'],
    3: ['classes.models.VanillaLSTM', 'VanillaLSTM'],
    4: ['classes.models.CNN_1D', 'CNN_1D']
}

Example: 
use 0 for mnist and neural network models
use 1 for stroke and neural network models
use 4 for stroke and fedxgboostllr
```


### Start clients
```python
python -m clients.client -ID $i -alg $alg -run $run -target_loss $target_loss
$alg, choices=['FedAvg', 'FedAdp', 'FedProx', 'Scaffold', 'FedDyn', 'FedDkw', 'FedNova', 'FedXGBllr'], default='FedXGBllr', help='FL algorithm'
$ID, default=0, help="device/learner identifier", type=int (must be unique)
$run, default=0, help="run number", type=int
$target_loss, default=0.001, help="sets the target loss to stop federation", type=float
$target_acc, default=0.99, help="sets the target acc to stop federation", type=float
```

### Start PS server
```python
python -m servers.server -alg $alg -run $run"
$alg, choices=['FedAvg', 'FedAdp', 'FedProx', 'Scaffold', 'FedDyn', 'FedDkw', 'FedNova', 'FedXGBllr'], default='FedXGBllr', help='FL algorithm'
$run, default=0, help="run number", type=int
```

## License

[MIT](https://choosealicense.com/licenses/mit/)