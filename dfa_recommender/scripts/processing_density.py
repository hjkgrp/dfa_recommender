'''
The script goes through the process of converting the density fitting features of
electron densities differences to a ready-to-use numpy array with propoer standardization
and zero padding. 
'''
import pickle
import numpy as np
import sys
import os


def standardize_over_all(x: list, padding_len: int = 1) -> dict:
    '''
    Atom-type-based standardization.
    
    Params:
    --------
        x: list,
            a list of density fitting features for a specific atom type
        padding_len: int, default as 1,
            the number of extra padding length; default as 1 for the atom type
    
    Returns:
    --------
        mean_std: dict,
            a dictionary for the mean and std. dev. for the density fitting features
    '''
    _mean = np.concatenate([np.mean(np.array(x), axis=0), np.array([0 for _ in range(padding_len)])], axis=-1)
    _std = np.std(np.array(x), axis=0)
    _std[_std < 1e-6] = 1
    _std = np.concatenate([_std, np.array([1 for _ in range(padding_len)])], axis=-1)
    mean_std = {"mean": _mean, "std": _std} 
    return mean_std

path2density = sys.argv[1]

atoms  = ["X", "H", "C", "N", "O", "F", "Cr", "Mn", "Fe", "Co"] # X represent a "vacuum" atom
atom_maps = {"X": 0, "H": 1, "C": 2, "N": 3, "O": 4, "F": 5, "P": 3, "S": 4, "Cl": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9}
ele_group = {"H": "H", "C": "C", "N": "N", "O": "O", "F": "F", "P": "N", "S": "O", "Cl": "F", 
             "Cr": "Cr", "Mn": "Mn", "Fe": "Fe", "Co": "Co"} # here we use the same local network for 2p/3p pairs.
max_size = 65 # largest number of atoms in the complexes of VSS-452. But the models is not limited by the size of atoms after trained.

if os.path.isfile(path2density):
    res_tot = pickle.load(open(path2density, "rb"))
else:
    raise FileNotFoundError("<%s> not found!" %path2density)
tot_sample = len(res_tot["symbol"])

# ---normalize density fitting features---
standard_dict, arrs = {}, {}
for ele in atoms[1:]:
    arrs[ele] = []
    standard_dict[ele] = {}
for ii in range(tot_sample):
    for jj, ele in enumerate(res_tot["symbol"][ii]):
        _den_alpha = np.pad(np.array(res_tot["spec"][ii]['alpha'][jj]), (0, 58-len(res_tot["spec"][ii]['alpha'][jj])))
        _den_beta = np.pad(np.array(res_tot["spec"][ii]['beta'][jj]), (0, 58-len(res_tot["spec"][ii]['alpha'][jj])))
        _tot = np.concatenate([_den_alpha, _den_beta], axis=-1)
        arrs[ele_group[ele]] += [_tot]
for ele in atoms[1:]:
    standard_dict[ele]  = standardize_over_all(arrs[ele], padding_len=1)
standard_dict["X"] = {"mean": np.zeros(shape=(58*2 + 1, )), "std": np.ones(shape=(58*2 + 1, ))}

# ---get normalized features---
X = np.zeros(shape=(tot_sample, max_size, 58*2 + 1))
c = 0
for ii in range(len(tot_sample)):
    for jj, ele in enumerate(res_tot['symbol'][ii]):
        _den_alpha = np.pad(np.array(res_tot["spec"][ii]['alpha'][jj]), (0, 58-len(res_tot["spec"][ii]['alpha'][jj])), 'constant', constant_values=(0, 0))
        _den_beta = np.pad(np.array(res_tot["spec"][ii]['beta'][jj]), (0, 58-len(res_tot["spec"][ii]['alpha'][jj])), 'constant', constant_values=(0, 0))
        _tot = np.concatenate([_den_alpha, _den_beta, np.array([atom_maps[ele]])], axis=-1)
        X[c, jj, :] = (_tot - standard_dict[ele_group[ele]]["mean"])/standard_dict[ele_group[ele]]["std"]
    c += 1

# ---save files---
with open("X.pkl", "wb") as fo:
    pickle.dump(X, fo)
with open("standard_dict.pkl", "wb") as fo:
    pickle.dump(standard_dict, fo)

