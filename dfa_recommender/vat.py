'''
Virtual adversarial training
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_normalize(d):
    '''
    Normalize d.

    Parameters
    ----------
    d: torch.Tensor
        random perturbation in the input space

    Returns
    ----------
    dn: torch.Tensor
        normalized random perturbation in the input space
    '''
    d = d.cpu().numpy()
    if len(d.shape) == 4:
        d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape(
            (-1, 1, 1, 1)) + 1e-16)
    elif len(d.shape) == 3:
        d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape(
            (-1, 1, 1)) + 1e-16)
    elif len(d.shape) == 2:
        d /= (np.sqrt(np.sum(d ** 2, axis=(1))).reshape(
            (-1, 1)) + 1e-16)
    else:
        raise ValueError("Dimension is not encoded yet.")
    dn = torch.from_numpy(d)
    return dn


def _entropy(logits):
    '''
    Calculation of cross entropy
    '''
    return -torch.mean(torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1))


def _entropy_array(logits):
    '''
    Calculation of cross entropy as an array (not averaged)
    '''
    return np.abs(torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1).detach().cpu().numpy())


class VAT(object):
    '''
    Implementation of virtual adversarial training. 
    See https://arxiv.org/abs/1704.03976 for more details.
    '''
    def __init__(self, device, eps, xi, alpha, k=1, use_entmin=False):
        self.device = device
        self.xi = xi
        self.eps = eps
        self.alpha = alpha
        self.k = k
        self.kl_div = nn.KLDivLoss(reduction='none').to(device)
        self.use_entmin = use_entmin

    def __call__(self, model, X):
        logits = model(X, update_batch_stats=False)
        prob_logits = F.softmax(logits.detach(), dim=1)
        d = _l2_normalize(torch.randn(X.size())).to(self.device)
        # d = _l2_normalize(torch.ones(X.size())).to(self.device)

        for ip in range(self.k):
            X_hat = X + d * self.xi
            X_hat.requires_grad = True
            logits_hat = model(X_hat, update_batch_stats=False)

            adv_distance = torch.mean(self.kl_div(
                F.log_softmax(logits_hat, dim=1), prob_logits).sum(dim=1))
            adv_distance.backward()
            d = _l2_normalize(X_hat.grad).to(self.device)

        logits_hat = model(X + self.eps * d, update_batch_stats=False)
        LDS = self.alpha * torch.mean(self.kl_div(F.log_softmax(logits_hat, dim=1), prob_logits).sum(dim=1))
        LDS_array = self.alpha * np.abs(
            self.kl_div(F.log_softmax(logits_hat, dim=1), prob_logits).detach().cpu().numpy()[:, 0])

        if self.use_entmin:
            LDS += _entropy(logits_hat)
            ent_array = _entropy_array(logits_hat)
            LDS_array += ent_array
        return LDS, LDS_array


class RPT(object):
    '''
    Implementation of random perturbation training. 
    See https://arxiv.org/abs/1704.03976 for more details.
    '''
    def __init__(self, device, eps, xi, k=10, use_entmin=False):
        self.device = device
        self.xi = xi
        self.eps = eps
        self.k = k
        self.kl_div = nn.KLDivLoss(reduction='none').to(device)
        self.use_entmin = use_entmin

    def __call__(self, model, X):
        LDS = 0
        LDS_array = np.zeros(shape=(X.shape[0],))
        for _ in range(self.k):
            logits = model(X, update_batch_stats=False)
            prob_logits = F.softmax(logits.detach(), dim=1)
            d = _l2_normalize(torch.randn(X.size())).to(self.device)
            X_prime = X + self.eps * d
            logits_hat = model(X + self.eps * d, update_batch_stats=False)
            LDS += torch.mean(self.kl_div(F.log_softmax(logits_hat, dim=1), prob_logits).sum(dim=1)).detach().numpy()
            LDS_array += np.abs(self.kl_div(F.log_softmax(logits_hat, dim=1), prob_logits).detach().numpy()[:, 0])
        LDS_array /= self.k
        LDS /= self.k
        # print(LDS_array, LDS)
        return LDS, LDS_array


def df_l2_normalize(d, l_x, cut=True):
    '''
    Normalize d with a zero masking.

    Parameters
    ----------
    d: torch.Tensor
        random perturbation in the input space
    l_x: torch.Tensor
        a tensor based on which the mask is created
    cut: bool, default as True
        whether applying the mask or not

    Returns
    ----------
    dn: torch.Tensor
        normalized random perturbation in the input space
    '''
    if cut:
        r = d[:, :, :-1]
        sample_size = l_x.shape
        rand_size = (sample_size[0], sample_size[1], sample_size[2] - 1)
        cat_size = (sample_size[0], sample_size[1], 1)
        zeros_mat = torch.zeros(rand_size)
        dn = torch.where(l_x[:, :, :-1] != 0, r, zeros_mat)
        dn = torch.cat((dn, torch.zeros(cat_size)), -1)
        dn = dn.cpu().numpy()
    else:
        dn = d.cpu().numpy() 
    if len(d.shape) == 3:
        dn /= (np.sqrt(np.sum(dn ** 2, axis=(1, 2))).reshape(
            (-1, 1, 1)) + 1e-16)
    else:
        raise ValueError("Dimension is not encoded yet.")
    dn = torch.from_numpy(dn)
    return torch.from_numpy(dn)


class regVAT(object):
    '''
    Implementation of virtual adversarial training in a regression task
    The only difference compared to VAT is the change of KL divergence to MSE
    in measuring the original and perturbed point.
    '''
    def __init__(self, device, eps, xi, alpha, k=1, cut=True):
        self.device = device
        self.xi = xi
        self.eps = eps
        self.alpha = alpha
        self.k = k
        self.cut = cut
        self.metric = torch.nn.L1Loss(reduction='none').to(device)

    def __call__(self, model, X, return_adv=False):
        model.eval()
        logits = model(X, update_batch_stats=False)
        prob_logits = logits.detach()
        d = df_l2_normalize(torch.randn(X.size()), X, self.cut).to(self.device)

        for __ in range(self.k):
            X_hat = X + d * self.xi
            X_hat.requires_grad = True
            logits_hat = model(X_hat, update_batch_stats=False)

            adv_distance = torch.mean(self.metric(logits_hat, prob_logits))
            adv_distance.backward()
            d = df_l2_normalize(X_hat.grad, X, self.cut).to(self.device)

        if return_adv:
            model.train()
            return d
        logits_hat = model(X + self.eps * d, update_batch_stats=False)
        LDS = self.alpha * torch.mean(self.metric(logits_hat, prob_logits))
        model.train()
        return LDS