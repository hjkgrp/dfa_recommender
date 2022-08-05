'''
Taking the Psi4 wfn and geometry to predict |\Delta \Delta E_{H-L}| and recommend a DFA
'''
from dfa_recommender.df_class import DensityFitting
from dfa_recommender.df_utils import get_spectra, get_subtracted_spectra
import numpy as np
import copy
import torch
import pickle
import operator
from typing import Tuple


all_functionals = [
    "bp86", "blyp", "pbe",
    "tpss", "scan", "m06-l", "mn15-l",
    "b3p86", "b3pw91", "b3lyp",
    "tpssh", "scan0", "m06", "m06-2x",
    "wb97x", "LRC-wPBEh",
    "b2gpplyp", "pbe0-dh", "dsd-blyp-d3bj", "dsd-pbeb95-d3bj", "dsd-pbep86-d3bj",
    "blyp_hfx_10", "blyp_hfx_20", "blyp_hfx_30", "blyp_hfx_40", "blyp_hfx_50",
    "pbe_hfx_10", "pbe_hfx_20", "pbe_hfx_30", "pbe_hfx_40", "pbe_hfx_50",   
    "scan_hfx_10", "scan_hfx_20", "scan_hfx_30", "scan_hfx_40", "scan_hfx_50", 
    "m06-l_hfx_10", "m06-l_hfx_30", "m06-l_hfx_40", "m06-l_hfx_50", 
    "mn15-l_hfx_10", "mn15-l_hfx_20", "mn15-l_hfx_30", "mn15-l_hfx_40", "mn15-l_hfx_50"
]


def get_df_fetaures(wfnpath: str, xyzfile: str, basis: str,
                    charge: int = 0, spin: int = 1,
                    wfnpath2: str = 'NA') -> Tuple[list, list, list]:
    '''
    {wfn, geo, basis, charge, spin, wfn2 (for LS, optional)} -> invariant DF features
    
    Parameters
        ----------
        wfnpath: str,
            path to the wavefunction file of the input molecule.
        xyzfile: str,
            path to the xyz file of the input molecule.
        basis: str,
            name of the auxiliary basis set for density fitting.
        charge: int,
            charge of the input molecule.
        spin: int,
            spin multiplicity (2*S + 1) for the input molecule.
            
    Returns
        ----------
        symbols: list,
            atom type for the molecule.
        ps_alpha: list,
            DF fearures for alpha electron.
        ps_beta: list,
            DF fearures for beta electron.
    '''
    densfit = DensityFitting(
        wfnpath=wfnpath,
        xyzfile=xyzfile,
        charge=charge,
        spin=spin,
        basis=basis,
    )
    ps_alpha = get_spectra(
        densfit=densfit,
        fock=False,
        H=False,
        t="alpha",
    )
    ps_beta = get_spectra(
        densfit=densfit,
        fock=False,
        H=False,
        t="beta",
    )
    symbols = densfit.symbols
    return symbols, ps_alpha, ps_beta


def features2padded_array(symbols, ps_alpha: list, ps_beta: list, 
                          normalize_path: str, max_size: int = 65) -> np.array:
    '''
    Pad, merge, and normalize DF features to a numpy array.
    
    Parameters
        ----------
        symbols: list,
            atom type for the molecule.
        ps_alpha: list,
            DF fearures for alpha electron.
        ps_beta: list,
            DF fearures for beta electron.
        normalize_path: str,
            path to the DF feature standardization dictionary.
        max_size: int, default to 65 (largest in the training set)
            dimension for your feature matrix.
    
    Returns
        ----------
        X: np.array,
            DF features
    '''
    standard_dict = pickle.load(open(normalize_path, "rb"))
    X = np.zeros(shape=(max_size, 58*2))
    atoms  = ["X", "H", "C", "N", "O", "F", "Cr", "Mn", "Fe", "Co"]
    atom_maps = {"X": 0, "H": 1, "C": 2, "N": 3, "O": 4, "F": 5, "P": 3, "S": 4, "Cl": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9}
    ele_group = {"H": "H", "C": "C", "N": "N", "O": "O", "F": "F", "P": "N", "S": "O", "Cl": "F", "Cr": "Cr", "Mn": "Mn", "Fe": "Fe", "Co": "Co"}
    for ii, ele in enumerate(symbols):
        df_alpha = np.pad(np.array(ps_alpha[ii]), (0, 58-len(ps_alpha[ii])))
        df_beta = np.pad(np.array(ps_beta[ii]), (0, 58-len(ps_beta[ii])))
        tot = np.concatenate([df_alpha, df_beta, np.array([atom_maps[ele]])], axis=-1)
        X[ii, :] = (tot - standard_dict[ele_group[ele]]["mean"])/standard_dict[ele_group[ele]]["std"]
    return X

        
def make_predictions(X: np.array, model_basepath: str,
                     y_scaler_path: str, device: str = "cpu",
                     functionals: list = all_functionals,
                    ) -> dict:
    '''
    Make predictions and return a sorted dictionary of results.
    
    Parameters
        ----------
        X: np.array,
            DF features
        model_basepath: str,
            basepath of TL models. Each model should be named as "mergedG10-abs-reg-<DFA>.pkl"
        y_scaler_path: str,
            path to the standard scaler object for |\Delta \Delta E_{H-L}|
        device: str, default as cpu
            torch device
        functionals: list, default as all_functionals
            DFAs to be considered
    
    Returns
        ----------
        sorted_res: dict,
            ordered DFA and predicted |\Delta \Delta E_{H-L}|
    '''
    X = torch.Tensor(X)
    y_scalers = pickle.load(open(y_scaler_path, "rb"))
    res = dict
    for f in functionals:
        model = pickle.load(open(model_basepath + "/mergedG10-abs-reg-%s.pkl"%f, "rb"))
        model.eval()
        with torch.no_grad():
            y_hat = model(X.to(device)).cpu().numpy()
        y_hat = y_scalers[f].inverse_transform(y_hat.reshape(-1, 1)).reshape(-1, )
        res[f] = np.abs(y_hat)
    sorted_res = dict(sorted(res.items(), key=operator.itemgetter(1), reverse=False))
    return sorted_res