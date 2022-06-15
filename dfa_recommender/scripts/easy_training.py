'''
Training procedure for the Behler-Parrinello type gated networks with
density fitting features as inputs
'''

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from dfa_recommender.net import GatedNetwork
from dfa_recommender.dataset import SubsetDataset
from dfa_recommender.sampler import InfiniteSampler
from dfa_recommender.ml_utils import numpy_to_dataset
from dfa_recommender.evaluate import evaluate_regressor

import numpy as np
import pickle
import pandas as pd
import copy

def weighted_mse_loss(input, target, weights = 1):
    out = torch.absolute(input-target) * weights
    loss = out.mean() 
    return loss


f = 'pbe0-dh'
atoms  = ["X", "H", "C", "N", "O", "F", "Cr", "Mn", "Fe", "Co"]

torch.set_num_threads(4)
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cpu')
num_workers = 0
bz = 16
stop_iter = 30

# ---load data---
X_org = pickle.load(open("<path-to-feature-array>", "rb"))
df_org = pd.read_csv("<path-to-data-csv-file")
y_scalers = pickle.load(open("<path-to-target-scaler>", "rb"))

# ---data processing---
_df = df_org.dropna(subset=["delta.%s.vertsse"%f])
_df_tr = _df[_df["train"] == 1]
_df_te = _df[_df["train"] == 0]
tr_inds, te_inds = _df_tr["index"].values, _df_te["index"].values
X_tr, X_te = X_org[tr_inds], X_org[te_inds]
print(X_tr.shape, X_te.shape)
y_scaler = y_scalers[f]
y_t = np.abs(df_org["delta.%s.vertsse"%f].values)
y_t = y_scaler.transform(y_t.reshape(-1, 1)).reshape(-1, )
y_tr, y_te = y_t[tr_inds], y_t[te_inds]

data_tr, data_te = numpy_to_dataset(X_tr, y_tr, regression=True), numpy_to_dataset(X_te, y_te, regression=True)
tr_l = SubsetDataset(data_tr, list(range(len(data_tr))))
te_l = SubsetDataset(data_te, list(range(len(data_te))))
print("sub labeled dataset length: ", len(tr_l), len(te_l))

# ---build and train--- 
cls = GatedNetwork(nin=58*2, n_out=10, n_hidden=96, 
                   n_layers=4, droprate=0.2,
                   elements=list(range(len(atoms)))).to(device)  # vertsse
cls.train()
optimizer = AdamW(list(cls.parameters()),
                  lr=2e-4,
                  betas=(0.90, 0.999),
                  weight_decay=1e-2,
                  amsgrad=True,
                  )
scheduler = ExponentialLR(optimizer, gamma=0.999)
l_tr_iter = iter(DataLoader(tr_l, bz, num_workers=num_workers,
                            sampler=InfiniteSampler(len(tr_l))))
l_te_iter = iter(DataLoader(te_l, bz, num_workers=num_workers,
                            sampler=InfiniteSampler(len(te_l))))
te_loader = DataLoader(te_l, len(te_l), num_workers=num_workers)
tr_l_loader = DataLoader(tr_l, len(tr_l), num_workers=num_workers)

mae_list, scaled_mae_list, rval_list = [], [], []
min_scale_mae = 10000
for epoch in range(2000):
    for niter in range(0, 1 + int(len(data_tr)/bz)):
        l_x, l_y = next(l_tr_iter)
        l_x, l_y = l_x.to(device), l_y.to(device)

        sup_reg_loss = weighted_mse_loss(cls(l_x), l_y, torch.ones(bz))
        
        unsup_reg_loss = sup_reg_loss
        loss = sup_reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    mae, scaled_mae, rval = evaluate_regressor(
        cls, te_loader, device, y_scaler)
    tr_mae, tr_scaled_mae, tr_rval = evaluate_regressor(
        cls, tr_l_loader, device, y_scaler)
    scaled_mae_list.append(scaled_mae)
    mae_list.append(mae)
    rval_list.append(rval)
        
    print('Iter {} train_mae {:.3} train_scaled_mae {:.3} train_rval {:.3} mae {:.3} scaled_mae {:.3} rval {:.3} SupLoss {:.3} UnsupLoss {:.3}'.format(
        epoch, tr_mae, tr_scaled_mae, tr_rval, mae, scaled_mae, rval, sup_reg_loss.item(), unsup_reg_loss.item()))
    if scaled_mae < min_scale_mae:
        print("copying best model with scaled mae: ", scaled_mae)
        min_scale_mae = scaled_mae
        best_model = copy.deepcopy(cls)
    if len(scaled_mae_list) > stop_iter:
        if min(scaled_mae_list[-stop_iter:]) - min(scaled_mae_list[:-stop_iter]) > 0:
            print("EarlyStopping.", min(
                scaled_mae_list[-stop_iter:]), min(scaled_mae_list[:-stop_iter]))
            break

# ---save model---
with open("model-%s.pkl"%f, "wb") as fo:
    pickle.dump(best_model, fo)