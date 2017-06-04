import numpy as np
import torch
import torch.utils.data as data_utils

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
