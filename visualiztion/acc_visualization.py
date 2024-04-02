import scipy.io
import matplotlib.pyplot as plt

algorithms = ['FEDAVG', 'FEDADP', 'FEDPROX', 'SCAFFOLD']
device = '0'
samples = '200'
batches = '3'
size = '50'
run = str(float(1))
loc = '7'
data = 'Mnist_non_iid'

# LOSS
for alg in algorithms:
    try:
        mat = scipy.io.loadmat(
            'Device_{}_samples_{}_batches_{}_size_{}_loc_{}_run_{}_dat_{}_alg_{}.mat'.format(device, samples, batches,
                                                                                             size, loc, run, data, alg))
        plt.plot(mat['epoch_loss_history'][0], label=alg)
    except:
        print('No ' + alg)

plt.xlabel('Rounds')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
# plt.savefig('Loss_Device_{}_samples_{}_batches_{}_size_{}_loc_{}_run_{}_dat_{}.png'.format(device, samples, batches,
#                                                                                           size, loc, run, data))

plt.show()

# accuracy
for alg in algorithms:
    try:
        mat = scipy.io.loadmat(
            'Device_{}_samples_{}_batches_{}_size_{}_loc_{}_run_{}_dat_{}_alg_{}.mat'.format(device, samples, batches,
                                                                                             size, loc, run, data, alg))
        plt.plot(mat['epoch_acc_history'][0], label=alg)
    except:
        print('No ' + alg)

plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
# plt.savefig('Accuracy_Device_{}_samples_{}_batches_{}_size_{}_loc_{}_run_{}_dat_{}.png'.format(device, samples, batches,
#                                                                                               size, loc, run, data))

plt.show()
