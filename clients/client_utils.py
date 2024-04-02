import os
import scipy.io as sio
from tqdm  import tqdm 
from classes.params.fl_param import NUM_EPOCHS
from classes.params import simul_param

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow warnings

def compile_model(model):
    #print("Compiling model")
    model.compile(optimizer=simul_param.OPTIMIZER(learning_rate=simul_param.LR), loss=simul_param.LOSS, metrics=simul_param.METRICS) 


def train_model(algorithm, dataset, num_batches):
    compile_model(algorithm.model_client)
    train_dataset = dataset.get_train_dataset(num_batches)
    if algorithm.name == 'FedDkw':
        algorithm.D_i = dataset.y_train_local# TODO: Encapsulate
    for epoch in range(NUM_EPOCHS):
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            for x_batch, y_batch in train_dataset:
                loss = algorithm.train_step(x_batch, y_batch)
                tepoch.update(1)  # Update progress bar for each batch
            tepoch.set_postfix(loss=float(loss))
    
                
def test_model(algorithm, data_handle):
    compile_model(algorithm.model_client)
    test_dataset = data_handle.get_test_dataset()
    # Evaluate the model on the test dataset
    loss, accuracy, precision, recall, auc = algorithm.model_client.evaluate(test_dataset, verbose=0)
    return loss, accuracy, precision, recall, auc

def save_results(model, metrics, id, run, alg_name, results_path, models_path):
    #model.save(models_path, include_optimizer=False, save_format='h5')
    model.save(models_path+f'/model_{id}_run_{run}_final.h5')
    file_name = results_path + f"/Device {id}_run_{run}_{alg_name}.mat"
    sio.savemat(file_name, metrics)