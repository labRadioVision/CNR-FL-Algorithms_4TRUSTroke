import numpy as np
import copy
from classes.params import fl_param, simul_param

def interpolate_weights(weights_old, weights_new):
    weights_interpolated = []
    for layer_old, layer_new in zip(weights_old, weights_new):
        weights_interpolated.append(fl_param.EPSILON*layer_new + (1-fl_param.EPSILON)*layer_old)
    return weights_interpolated

def assign_global(weights_global):
    return [copy.copy(weights_global) for _ in range(fl_param.NUM_CLIENTS)]


def flatten_weights(weights):
    weights_flat = []
    for layer_ in range(len(weights)):
        weights_flat.append(weights[layer_].flatten())
    return np.concatenate(weights_flat)


def nan_weights(weights, id=None):
    if np.isnan(flatten_weights(weights)).any():
        print(f"NAN WEIGHTS in {id}")