import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from vat import VAT, RPT
import matplotlib.pyplot as plt

plt.style.use("myline")


def evaluate_regressor(regressor, loader, device, y_scaler):
    '''
    Evaluate the model performance on a single regression task

    Parameters
    ----------
    regressor: torch.nn.Module
        trained regression model
    loader: torch.utils.data.DataLoader
        your torch dataloader
    device: torch.device
        the device at which this evaluation is performed
    y_scaler: sklearn.preprocessing.StandardScaler
        the scaler that you normalize the label of training data

    Returns
    ----------
    mae: float
        MAE
    scaled_mae: float
        scaled MAE
    rval: float,
        Pearson's coefficient
    R2: float
        Coefficient of determination
    '''
    assert isinstance(regressor, torch.nn.Module)
    assert isinstance(loader, torch.utils.data.DataLoader)
    assert isinstance(device, torch.device)

    regressor.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            _pred = regressor(x.to(device))
            preds.append(_pred.cpu().numpy())
            labels.append(y.cpu().numpy())
    y = y_scaler.inverse_transform(labels[0].reshape(-1, 1)).reshape(-1, )
    y_hat = y_scaler.inverse_transform(preds[0][:, 0].reshape(-1, 1)).reshape(-1, )
    mae = mean_absolute_error(y_hat, y)
    scaled_mae = mae/(np.max(y) - np.min(y))
    # mae = mean_absolute_error(labels[0].reshape(-1, ), preds[0][:, 0])
    # scaled_mae = mae/(np.max(labels[0].reshape(-1, )) - np.min(labels[0].reshape(-1, )))
    R2 = r2_score(labels[0].reshape(-1, ), preds[0][:, 0])
    rval = pearsonr(labels[0].reshape(-1, ), preds[0][:, 0])[0]
    regressor.train()
    return mae, scaled_mae, rval, R2
