FL system designed for research and algorithms performance comparison.
Funded by Horizon EU project TRUSTroke in the call HORIZON-HLTH-2022-STAYHLTH-01-two-stage under GA No. 101080564.

### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages (python >=3.7, 3.8 preferable).

```bash
pip install -r requirements.txt
```

### TEST algorithms
Algorithms can be tested using tabular data (stroke public dataset https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) and image dataset MNIST. Currently supported algorithms:
```
Tabular data: "FedAvg" "FedAdp" "FedProx" "Scaffold" "FedDyn" "FedDkw" "FedNova" "FedXGBllr"
MNIST data: "FedAvg" "FedAdp" "FedProx" "Scaffold" "FedDyn" "FedDkw" "FedNova"
```
All algorithm implementations are in [/classes/algorithms/]

### Federated data: create a federated dataset and distribute to clients 
```python
python -m classes.Datasets.data_generator -data $data -samples $samples -data $data -niid $n_iid -alpha 0.1
$data choices=['mnist', 'stroke'], default='stroke', help = 'Type of data',
$samples default=100, help="sets a fixed number samples per device"
$n_iid choices=['iid', 'label', 'sample', 'feature'], default='iid', help="Heterogeneity type"
NOTE: stroke dataset can be shared upon request
```

### MQTT
The FL system runs on a MQTT platform. Simple mosquitto installation can be found in https://mosquitto.org/download/. Installation file can be provided upon request.
```
START BROKER (optionally) use:
python Launcher_mosquitto.py 
For simple localhost configuration use:
MQTT_broker_config_local.json
For remote testing (example: test.mosquitto.org) - unencrypted and unauthenticated:
MQTT_broker_config.json
```

### Clients, example of a simple startup
```
In the [classes/params/fl_param.py] file, set the number of clients in NUM_CLIENTS and the number of clients to be selected at each round by the Server in CLIENTS_SELECTED.
In the [classes/params/fl_param.py] file, set the maximum number of FL rounds in NUM_ROUNDS (example 50 rounds)
In the [classes/params/simul.param.py] set the ML model parameter in model_id MODEL = TYPES_MODELS[model_id] according to the list below (further models can be added)
```

```python
TYPES_MODELS = {
    0: ['classes.models.CNN', 'CNN'],
    1: ['classes.models.VanillaMLP', 'VanillaMLP'],
    2: ['classes.models.Mobinet','Mobinet'],
    3: ['classes.models.VanillaLSTM', 'VanillaLSTM'],
    4: ['classes.models.CNN_1D', 'CNN_1D']
}
```
Example:
```
use 0 for mnist and neural network models
use 1 for stroke and neural network models
use 4 for stroke and fedxgboostllr
```

### Start clients
```python
python -m clients.client -ID $i -alg $alg -run $run -target_loss $target_loss -target_acc $target_acc
$alg, choices=['FedAvg', 'FedAdp', 'FedProx', 'Scaffold', 'FedDyn', 'FedDkw', 'FedNova', 'FedXGBllr'], default='FedXGBllr', help='FL algorithm'
$ID, default=0, help="device/learner identifier", type=int (must be unique for each deployed client)
$run, default=0, help="run number", type=int
$target_loss, default=0.001, help="sets the target loss to stop federation", type=float
$target_acc, default=0.99, help="sets the target acc to stop federation", type=float
```
Example (3 clients)
```python
python -m clients.client -ID 0 -alg 'Scaffold' -run 0
python -m clients.client -ID 1 -alg 'Scaffold' -run 0
python -m clients.client -ID 2 -alg 'Scaffold' -run 0
```

### Start PS server
```python
python -m servers.server -alg $alg -run $run"
$alg, choices=['FedAvg', 'FedAdp', 'FedProx', 'Scaffold', 'FedDyn', 'FedDkw', 'FedNova', 'FedXGBllr'], default='FedXGBllr', help='FL algorithm'
$run, default=0, help="run number", type=int
```
Example 
```python
python -m servers.server -alg 'Scaffold' -run 0
```
### Notes
The folder [output] contains the stored local and global models in h5 format. Use sklearn.metrics for accuracy score evaluation

## License

[MIT](https://choosealicense.com/licenses/mit/)
