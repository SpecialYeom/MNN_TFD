from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.utils.data as data_utils

import numpy as np
import matplotlib.pyplot as plt
plt.ion()


def LoadData(fname):
    """Loads data from an NPZ file.

    Args:
        fname: NPZ filename.

    Returns:
        data: Tuple {inputs, target}_{train, valid, test}.
              Row-major, outer axis to be the number of observations.
    """
    npzfile = np.load(fname)

    inputs_train = npzfile['inputs_train'] / 255.0
    inputs_valid = npzfile['inputs_valid'] / 255.0
    inputs_test = npzfile['inputs_test']/ 255.0
    target_train = npzfile['target_train']
    target_valid = npzfile['target_valid']
    target_test = npzfile['target_test']
    train = data_utils.TensorDataset(torch.from_numpy(inputs_train), torch.from_numpy(target_train))
    valid = data_utils.TensorDataset(torch.from_numpy(inputs_valid), torch.from_numpy(target_valid))
    test = data_utils.TensorDataset(torch.from_numpy(inputs_test), torch.from_numpy(target_test))
    return train, valid, test



def DisplayPlot(train, valid, ylabel, number=0):
    """Displays training curve.

    Args:
        train: Training statistics.
        valid: Validation statistics.
        ylabel: Y-axis label of the plot.
    """
    plt.figure(number)
    plt.clf()
    train = np.array(train)
    valid = np.array(valid)
    plt.plot(train[:, 0], train[:, 1], 'b', label='Train')
    plt.plot(valid[:, 0], valid[:, 1], 'g', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.draw()

    
def SaveStats(fname, data):
    """Saves the model to a numpy file."""
    print('Writing to ' + fname)
    np.savez_compressed(fname, **data)
    
def LoadStats(fname):
    """Loads model from numpy file."""
    print('Loading from ' + fname)
    return dict(np.load(fname))