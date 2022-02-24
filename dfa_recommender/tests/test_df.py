'''
test suites for DensityFitting class
'''

from dfa_recommender.df_class import DensityFitting
from dfa_recommender.df_utils import get_spectra, get_subtracted_spectra
import numpy as np
import pytest


def test_df_water():

    ref_dab_P = np.load("./reference/water/dab_p.npy", allow_pickle=True)
    ref_ps = np.load("./reference/water/ps.npy", allow_pickle=True)
    densfit = DensityFitting(
        wfnpath="./inputs/water/wfn.180.npy",
        xyzfile="./inputs/water/geo.xyz",
        charge=0,
        spin=1,
        basis="def2-tzvp"
    )
    ps = get_spectra(
        densfit=densfit,
        fock=False,
        H=False,
        t="alpha",
    )
    assert np.max(np.abs(ref_dab_P - densfit.dab_P)) < 1e-6
    assert np.max([np.max(np.abs(np.array(ref_ps[ii]) - np.array(ps[ii]))) for ii in range(ps.shape[0])]) < 1e-6


def test_df_vert_spin_splitting():

    ref_ps = np.load("./reference/vert_spin_splitting/ps.npy", allow_pickle=True)
    densfit = DensityFitting(
        wfnpath="./inputs/vert_spin_splitting/wfn.180.npy",
        xyzfile="./inputs/vert_spin_splitting/geo.xyz",
        charge=0,
        spin=1,
        basis="def2-tzvp",
        wfnpath2="./inputs/vert_spin_splitting/triplet_wfn.180.npy",
    )
    ps = get_subtracted_spectra(
        densfit=densfit,
        fock=False,
        t="alpha",
    )
    assert np.max([np.max(np.abs(np.array(ref_ps[ii]) - np.array(ps[ii]))) for ii in range(ps.shape[0])]) < 1e-6


def test_no_wfn2():
    try:
        _ = DensityFitting(
            wfnpath="./inputs/vert_spin_splitting/wfn.180.npy",
            xyzfile="./inputs/vert_spin_splitting/geo.xyz",
            charge=0,
            spin=1,
            basis="def2-tzvp",
            wfnpath2="wrong_path",
        )
    except FileNotFoundError as e:
        assert "wrong_path" in str(e) and "Err" in str(e)
