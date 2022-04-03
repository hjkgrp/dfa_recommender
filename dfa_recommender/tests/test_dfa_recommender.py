"""
Unit and regression test for the dfa_recommender package.
"""

# Import package, test suite, and other packages as needed
import sys
import pickle
import pytest
import dfa_recommender
import numpy as np
import pandas as pd
from pkg_resources import resource_filename, Requirement


def test_dfa_recommender_imported():
    '''
    Sample test, will always pass so long as import statement worked
    '''
    assert "dfa_recommender" in sys.modules
    
def test_psi4_import():
    '''
    Test whether psi4 can be imported
    '''
    try:
        import psi4
        assert "psi4" in sys.modules
    except ImportError:
        assert 0

def test_torch_import():
    '''
    Test whether torch can be imported
    '''
    try:
        import torch
        assert "torch" in sys.modules
    except ImportError:
        assert 0
        
def test_load_model():
    '''
    Test model loading
    '''
    from dfa_recommender.net import GatedNetwork, MySoftplus, TiledMultiLayerNN, MLP, finalMLP, ElementalGate
    
    if __name__ == '__main__':
        basepath = resource_filename(Requirement.parse("dfa_recommender"), "/dfa_recommender/data/")
        _ = pickle.load(open(basepath + "/models-trends/mergedG10-abs-reg-b3lyp.pkl", "rb"))
        
def test_build_nn():
    '''
    Test building GatedNetwork
    '''
    import torch
    from dfa_recommender.net import GatedNetwork, MySoftplus, TiledMultiLayerNN, MLP, finalMLP, ElementalGate
    
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cpu')
    
    model = GatedNetwork(nin=58*3+7, n_out=10, n_hidden=96,
                         n_layers=4, droprate=0,
                         elements=list(range(10)))
    x = torch.zeros((1, 20, 58*3+7+1))
    assert np.abs(model(x).detach().numpy().reshape(-1)[0] - -0.041750696) < 1e-6
    x = torch.ones((1, 20, 58*3+7+1))
    assert np.abs(model(x).detach().numpy().reshape(-1)[0] - -0.06381429) < 1e-6
    
    
def test_vat():
    '''
    Test VAT
    '''
    import torch
    from dfa_recommender.net import GatedNetwork, MySoftplus, TiledMultiLayerNN, MLP, finalMLP, ElementalGate
    from dfa_recommender.vat import regVAT, VAT
    
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cpu')
    
    model = GatedNetwork(nin=58*3+7, n_out=10, n_hidden=96,
                         n_layers=4, droprate=0,
                         elements=list(range(10)))
    x = torch.ones((1, 20, 58*3+7+1))
    eps = 1.
    xi = 1e-3
    alpha = 1.
    cut = True
    vat_criterion = regVAT(device, eps, xi, alpha, k=3, cut=cut)
    d_x = vat_criterion(model, x, return_adv=True).numpy()
    
    assert np.abs(np.max(d_x) - 0.036961406) < 1e-4
    assert np.abs(np.min(d_x) - -0.04693729) < 1e-4
    
    
def test_evaluator():
    '''
    Test the model evaluation functions
    '''
    import torch
    from dfa_recommender.net import GatedNetwork, MySoftplus, TiledMultiLayerNN, MLP, finalMLP, ElementalGate
    from sklearn.preprocessing import StandardScaler
    from dfa_recommender.evaluate import evaluate_regressor
    from dfa_recommender.dataset import SubsetDataset
    from torch.utils.data import DataLoader
    
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cpu')
    
    model = GatedNetwork(nin=58*3+7, n_out=10, n_hidden=96,
                         n_layers=4, droprate=0,
                         elements=list(range(10)))
    X = torch.stack([torch.zeros((20, 58*3+7+1)), torch.ones((20, 58*3+7+1))])
    y = torch.Tensor([-0.041750696, -0.06381429])
    scaler = StandardScaler()
    scaler.fit(np.array([[1], [0], [-1]]))
    dataset = torch.utils.data.TensorDataset(X, y)
    sub = SubsetDataset(dataset, list(range(len(dataset))))
    loader = DataLoader(sub, len(sub), num_workers=0)
    mae, scaled_mae, rval = evaluate_regressor(model, loader, device, scaler)
    
    assert np.abs(mae) < 1e-6
    assert np.abs(scaled_mae) < 1e-6
    assert np.abs(1 - rval) < 1e-6
    
    
def test_nn_workflow():
    '''
    Test the whole workflow of using trained NN for predictions.
    '''
    import torch
    from torch.utils.data import DataLoader
    from dfa_recommender.ml_utils import numpy_to_dataset
    from dfa_recommender.dataset import SubsetDataset
    from dfa_recommender.sampler import InfiniteSampler
    from dfa_recommender.evaluate import evaluate_regressor
    from dfa_recommender.net import GatedNetwork, MySoftplus, TiledMultiLayerNN, MLP, finalMLP, ElementalGate
    from dfa_recommender.vat import regVAT, VAT
    
    if __name__ == '__main__':
        basepath = resource_filename(Requirement.parse("dfa_recommender"), "/dfa_recommender/data/")
        X_org = pickle.load(open(basepath + "/X.pickle", "rb"))
        df_org = pd.read_csv(basepath + "/labeled_res.csv")
        inds = pickle.load(open(basepath + "/final_inds.pkl", "rb"))
        y_scalers = pickle.load(open(basepath + "/abs-reg-y_scalers.pkl", "rb"))
        torch.set_num_threads(4)
        torch.manual_seed(0)
        np.random.seed(0)
        device = torch.device('cpu')
        
        f = 'b3lyp'
        tr_inds, te_inds = inds["train"], inds["test"]
        X_tr, X_te = X_org[tr_inds], X_org[te_inds]
        y_scaler = y_scalers[f]
        y_t = np.abs(df_org["delta.%s.vertsse"%f].values)
        y_t = y_scaler.transform(y_t.reshape(-1, 1)).reshape(-1, )
        y_tr, y_te = y_t[tr_inds], y_t[te_inds]
        _, data_te = numpy_to_dataset(X_tr, y_tr, regression=True), numpy_to_dataset(X_te, y_te, regression=True)
        te_l = SubsetDataset(data_te, list(range(len(data_te))))
        l_te_iter = iter(DataLoader(te_l, 1, num_workers=0,
                                    sampler=InfiniteSampler(len(te_l))))
        te_loader = DataLoader(te_l, len(te_l), num_workers=0)
        best_model = pickle.load(open(basepath + "/models-trends/mergedG10-abs-reg-%s.pkl"%f, "rb"))

        mae, scaled_mae, rval = evaluate_regressor(best_model, te_loader, device, y_scaler)
        assert np.abs(mae - 2.2448943) < 1e-2