'''
Utility functions for preparing datasets, model training and evaluation.
'''

import torch

def numpy_to_dataset(X, y, regression=False):
    '''
    Aseemble numpy arrays to torch tensor data set
    
    Parameters
    ----------
    X: np.array
        features
    y: np.array
        targets
    regression: bool, default as False
        whether a regression task or not
        
    Returns
    ----------
    data: torch.utils.data.TensorDataset
        assembled data set
    '''
    X = torch.stack([torch.Tensor(i) for i in X])
    y = torch.Tensor(y)
    if not regression:
        y = y.type(torch.LongTensor)
    data = torch.utils.data.TensorDataset(X, y)
    return data