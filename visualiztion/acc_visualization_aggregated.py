import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
file_param = {
    'algorithms': ['FEDAVG','FEDADP','FEDPROX','SCAFFOLD','FEDDYN','FEDDKW', 'FEDNOVA'],
    'samples': '500',
    'batches': '4',
    'size': '25',
    'loc': '5',
    'data': 'Mnist_non_iid_full_Split'
}
runs_done = 3  # how many runs were done to compute the confidence interval (starting from zero)
devices = 9
def plot_data(file_param, metric):
    """for each algorithm compute get statistics out of 'runs_done' runs. Each run
    comprises aggregated devices, since if case C<K the global model wasn't validated on a specific
    device every epoch (sometimes a device was not selected when the evaluation occurred"""
    metric_dict = {'epoch_loss_history': 'Loss', 'epoch_acc_history': 'Accuracy'}
    for alg in file_param['algorithms']:# FOR EACH ALGORITHM
        print(alg)
        agg = []
        for r in range(0, runs_done): # FOR EACH RUN
            print('run', r)
            X = []
            Y = []
            for d in range(devices):# FOR EACH DEVICE
                print('Dev', d)
                try:
                    mat = scipy.io.loadmat(
                        'Device_{}_samples_{}_batches_{}_size_{}_loc_{}_run_{}_dat_{}_alg_{}.mat'.format(str(d), file_param['samples'],file_param['batches'],
                                                                                                         file_param['size'],file_param['loc'],str(float(r)),file_param['data'],alg))
                    X.append(mat['epochs_history'][0])
                    Y.append(mat[metric][0])
                except:
                    pass
            XY_dict = {}
            for i in range(len(X)):
                for j, _ in enumerate(X[i]):
                    if X[i][j] in XY_dict:
                        XY_dict[X[i][j]].append(Y[i][j])
                    else:
                        XY_dict[X[i][j]] = [Y[i][j]]

            # Step 10: Compute median of overlapping Y values
            X_sorted = sorted(XY_dict.keys())
            Y_median = []
            for x in X_sorted:
                Y_values = XY_dict[x]
                if len(Y_values) == 1:
                    Y_median.append(Y_values[0])
                else:
                    Y_median.append(np.median(Y_values))
            print(np.array(Y_median).shape)
            agg.append(Y_median)
        agg = np.array(agg, dtype=object)

        max_length = max(len(a) for a in agg)

        if agg.shape[0] > 1:
            # pad arrays with zeros to make them all the same length
            agg_with_zeros = np.array([np.pad(a, (0, max_length - len(a)), 'constant') for a in agg])
            # calculate the mean of non-zero elements along each column
            mean_agg = np.divide(np.sum(agg_with_zeros, axis=0), np.sum(agg_with_zeros != 0, axis=0), where=np.sum(agg_with_zeros != 0, axis=0) != 0).astype(float)


            # replace values with single element with that element itself
            agg_with_last = np.array([np.pad(a, (0, max_length - len(a)), 'edge') for a in agg])
            # calculate the standard deviation of non-zero elements along each column
            std_agg = np.sqrt(np.divide(np.sum(np.square(agg_with_last - mean_agg), axis=0), np.sum(agg_with_last != 0, axis=0),
                                        where=np.sum(agg != 0, axis=0) != 0).astype(float))
            confidence_interval = 1.96 * std_agg / np.sqrt(max_length)
        else:
            mean_agg = agg.flatten()
            confidence_interval = 0 * agg.flatten()

        plt.plot(range(1, max_length+1), mean_agg, label=alg)
        plt.fill_between(range(1, max_length+1), mean_agg - confidence_interval, mean_agg + confidence_interval, alpha=0.2)

    plt.xlabel('Rounds')
    plt.ylabel(metric_dict[metric])
    plt.grid(True)
    plt.legend()
    #    plt.savefig(metric_dict[metric]+'_Device_{}_samples_{}_batches_{}_size_{}_loc_{}_run_{}_dat_{}.png'.format(device, samples, batches,
    #                                                                                             size, loc, run, data))
    plt.show()

plot_data(file_param, 'epoch_loss_history')
plot_data(file_param, 'epoch_acc_history')

