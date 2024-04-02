
import numpy as np
import random

from classes.params import simul_param, mqtt_param, fl_param

class Scheduler:
    def __init__(self):
        self.scheduling_tx = None
        self.active_check = None
        self.counter = 0
        
        
    def selected(self, rx_dev):
        if self.scheduling_tx[rx_dev, 0] == 1:
            return True
        return False
        
    def _update_received(self, rx_dev):
        if not self.active_check[rx_dev]:
            self.counter += 1
            self.active_check[rx_dev] = True
            
    def received_all(self):
        if (self.counter == fl_param.CLIENTS_SELECTED):
            return True    
        return False
    
    def _reset_scheduler(self):
        self.active_check = np.zeros(fl_param.NUM_CLIENTS, dtype=bool)
        self.counter = 0
    
    def select_clients(self, round):
        self.active_check = np.zeros(fl_param.NUM_CLIENTS, dtype=bool)  # reset active checks
        self.counter = 0  # reset counters
        if fl_param.SHED_TYPE == 'Robin':  # round robin
            sr = fl_param.NUM_CLIENTS  # used for scheduling table generation
            sr2 = round % sr
            clients = np.sort(np.arange(sr2, fl_param.CLIENTS_SELECTED + sr2) % sr)  # LIST OF CLIENTS TO TX TO
            self.scheduling_tx = np.zeros((fl_param.NUM_CLIENTS, 1), dtype=int)
            self.scheduling_tx[clients, 0] = 1
        elif fl_param.SHED_TYPE == 'Rand':
            device_indices = list(range(0, fl_param.NUM_CLIENTS))
            random.shuffle(device_indices)
            clients = np.array(device_indices[:fl_param.CLIENTS_SELECTED])  # LIST OF CLIENTS TO TX TO
            self.scheduling_tx = np.zeros((fl_param.NUM_CLIENTS, 1), dtype=int)
            self.scheduling_tx[clients, 0] = 1
        return clients