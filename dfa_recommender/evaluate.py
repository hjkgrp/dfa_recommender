import copy
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from vat import VAT, RPT
from utils import get_latent_space, do_pca, do_tsne, do_umap
import matplotlib.pyplot as plt

plt.style.use("myline")


def _evaluate_classifier(classifier, x, y, device):
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(device, torch.device)

    classifier.eval()
    n_err = 0
    preds = []
    labels = []
    with torch.no_grad():
        prob_y = F.softmax(classifier(x.to(device)), dim=1)
        preds.append(prob_y.cpu().numpy())
        labels.append(y.cpu().numpy())
        pred_y = torch.max(prob_y, dim=1)[1]
        pred_y = pred_y.to(torch.device('cpu'))
        n_err += (pred_y != y).sum().item()
    _labels = np.eye(max(labels[0]) + 1)[labels[0]]
    val_auc = roc_auc_score(_labels, preds[0])
    classifier.train()
    return n_err, val_auc, np.mean(pred_y.tolist()), np.mean(preds[0][:, 1].tolist())


def evaluate_classifier(classifier, loader, device):
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(device, torch.device)

    classifier.eval()
    top1_correct = 0
    top3_correct = 0
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            prob_y = F.softmax(classifier(x.to(device)), dim=1)
            preds += list(prob_y.cpu().numpy())
            labels += list(y.cpu().numpy())
            pred_y = torch.topk(prob_y, 1, dim=1)[1] ## for top n, torch.topk(prob_y, n, dim=1)[1]
            pred_y = pred_y.to(torch.device('cpu'))
            top1_correct += (pred_y[:, 0] == y).sum().item()
            # top3_correct += (pred_y[:, 0] == y).sum().item()
            # top3_correct += ((pred_y[:, 0] == y).sum().item() + (pred_y[:, 1] == y).sum().item() + (pred_y[:, 2] == y).sum().item())
    # print(preds, labels)
    val_auc = roc_auc_score(labels, preds, average='macro', multi_class="ovo") # multiclass
    # sardines
    # val_auc = roc_auc_score(labels, np.array(preds)[:, 1])
    classifier.train()
    return top1_correct*1./len(labels), top3_correct*1./len(labels), val_auc


def evaluate_regressor(regressor, loader, device, y_scaler):
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
    return mae, scaled_mae, rval


def averaged_label(classifier, x, y, device):
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(device, torch.device)

    classifier.eval()
    preds = []
    probs = []
    with torch.no_grad():
        prob_y = F.softmax(classifier(x.to(device)), dim=1).cpu().numpy()
        y_pred = np.argmax(prob_y, axis=1).tolist()
        preds += y_pred
        probs += prob_y[:, 1].tolist()
    y_avrg = np.mean(preds)
    prob_avrg = np.mean(probs)
    classifier.train()

    return y_avrg, prob_avrg


def adversarial_loss_final(classifier, loader, device, eps, xi, alpha, use_entmin):
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(loader, torch.utils.data.DataLoader)
    assert isinstance(device, torch.device)

    # classifier.eval()
    ce = 0
    count = 0
    criterion = VAT(device, eps=eps, xi=xi, alpha=alpha, use_entmin=use_entmin)
    # criterion = RPT(device, eps=eps, xi=xi, use_entmin=use_entmin)
    for x, y in loader:
        x = x.to(device)
        count += 1
        _ce, _ce_array = criterion(classifier, x)
        ce += _ce
        if not "ce_array" in locals():
            ce_array = _ce_array.copy()
        else:
            ce_array = np.concatenate((ce_array, _ce_array), axis=0)
    ce /= count
    return ce.detach().numpy()


def latent_unsuploss(layer, classifier, x, y, device, criterion, figname="tmp.pdf"):
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(device, torch.device)

    x = x.to(device)
    y = y.to(device)
    y_ul = y.detach().numpy().copy()
    _y_ul = y.detach().numpy().copy()
    y_ul[_y_ul == 0] = 2
    y_ul[_y_ul == -1] = 1
    y_ul[_y_ul == 1] = 0
    c0 = x[:, 6]
    ce, ce_array = criterion(classifier, x)
    classifier.eval()
    prob_y = F.softmax(classifier(x), dim=1).detach().numpy()
    y_pred = np.argmax(prob_y, axis=1)
    latent_space = get_latent_space(classifier, x)[layer]
    classifier.train()
    trans_lspace = do_tsne(latent_space)  ### A Fixed random state MATTERS!!!!!

    fig = plt.figure(figsize=(12, 10))
    fig.add_subplot(2, 2, 1)
    plt.scatter(trans_lspace[:, 0], trans_lspace[:, 1], c=c0, s=16, cmap='jet')
    cb = plt.colorbar()
    fig.add_subplot(2, 2, 2)
    plt.scatter(trans_lspace[:, 0], trans_lspace[:, 1], c=ce_array, s=16, cmap='binary')
    cb = plt.colorbar(ticks=np.array([0, 0.1, 0.2]))
    plt.clim(0, 0.2)
    fig.add_subplot(2, 2, 3)
    plt.scatter(trans_lspace[:, 0], trans_lspace[:, 1], c=prob_y[:, 1], s=16, cmap='bwr')
    cb = plt.colorbar(ticks=np.array([0, 1]))
    plt.clim(0, 1)
    fig.add_subplot(2, 2, 4)
    plt.scatter(trans_lspace[:, 0], trans_lspace[:, 1], c=_y_ul, s=16, cmap='rainbow')
    cb = plt.colorbar(ticks=np.array([-1, 0, 1]))
    plt.clim(-1, 1)
    plt.tight_layout()
    fig.savefig(figname)
