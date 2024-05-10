'''
test suites for DensityFitting class
'''

import torch
from e3nn import o3
from dfa_recommender.df_class import DensityFitting
from dfa_recommender.df_utils import get_spectra, get_subtracted_spectra
import numpy as np
import pytest
from pkg_resources import resource_filename, Requirement


def test_df_water():

    basepath = resource_filename(Requirement.parse("dfa_recommender"), "/dfa_recommender/tests/")
    ref_dab_P = np.load(basepath + "reference/water/dab_p.npy", allow_pickle=True)
    ref_ps = np.load(basepath + "reference/water/ps.npy", allow_pickle=True)
    ref_CP = np.load(basepath + "reference/water/CP_pad_e3nn.npy", allow_pickle=True)
    densfit = DensityFitting(
        wfnpath=basepath + "inputs/water/wfn.180.npy",
        xyzfile=basepath + "inputs/water/geo.xyz",
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
    densfit.compensate_charges()
    densfit.pad_df_coeffs()
    densfit.convert_CP2e3nn()
    
    assert np.max(np.abs(ref_dab_P - densfit.dab_P)) < 1e-6
    assert np.max(np.abs(ref_CP - densfit.C_P_pad_e3nn)) < 1e-6
    assert np.max([np.max(np.abs(np.array(ref_ps[ii]) - np.array(ps[ii]))) for ii in range(ps.shape[0])]) < 1e-6


@pytest.mark.parametrize("name, axis, angle",
    [
        ("rotation1", [1.0, 1.0, 1.0], np.pi / 4),
        ("rotation2", [1.0, 0.5, 0.1], np.pi / 6),
    ]
)
def test_df_rotation_equivariance(name, axis, angle):

    charge = 2
    spin = 2
    basis = "def2-tzvp"

    basepath = resource_filename(Requirement.parse("dfa_recommender"), "/dfa_recommender/tests/")
    densfit = DensityFitting(
        wfnpath=basepath + "inputs/rotational_equivariance/rotation0_LS_HF_wfn.npy",
        xyzfile=basepath + "inputs/rotational_equivariance/rotation0.xyz",
        charge=charge,
        spin=spin,
        basis=basis,
    )
    densfit.get_df_coeffs(densfit.wfn.Da())
    densfit.pad_df_coeffs()
    densfit.convert_CP2e3nn()

    densfit_ref = DensityFitting(
        wfnpath=basepath + f"inputs/rotational_equivariance/{name}_LS_HF_wfn.npy",
        xyzfile=basepath + f"inputs/rotational_equivariance/{name}.xyz",
        charge=charge,
        spin=spin,
        basis=basis,
    )
    densfit_ref.get_df_coeffs(densfit_ref.wfn.Da())
    densfit_ref.pad_df_coeffs()
    densfit_ref.convert_CP2e3nn()

    rot = o3.axis_angle_to_matrix(torch.tensor(axis), torch.tensor(angle))
    # Construct wigner D matrix from rotation matrix
    D = densfit.irreps.D_from_matrix(rot).numpy()

    np.testing.assert_allclose(
        densfit.C_P_pad_e3nn @ D.T,
        densfit_ref.C_P_pad_e3nn,
        atol=1e-6,
        rtol=1e-5,
    )


def test_df_vert_spin_splitting():

    basepath = resource_filename(Requirement.parse("dfa_recommender"), "/dfa_recommender/tests/")
    ref_ps = np.load(basepath + "reference/vert_spin_splitting/ps.npy", allow_pickle=True)
    densfit = DensityFitting(
        wfnpath=basepath + "inputs/vert_spin_splitting/wfn.180.npy",
        xyzfile=basepath + "inputs/vert_spin_splitting/geo.xyz",
        charge=0,
        spin=1,
        basis="def2-tzvp",
        wfnpath2=basepath + "inputs/vert_spin_splitting/triplet_wfn.180.npy",
    )
    ps = get_subtracted_spectra(
        densfit=densfit,
        fock=False,
        t="alpha",
    )
    assert np.max([np.max(np.abs(np.array(ref_ps[ii]) - np.array(ps[ii]))) for ii in range(ps.shape[0])]) < 1e-6


def test_no_wfn2():
    basepath = resource_filename(Requirement.parse("dfa_recommender"), "/dfa_recommender/tests/")
    try:
        _ = DensityFitting(
            wfnpath=basepath + "inputs/vert_spin_splitting/wfn.180.npy",
            xyzfile=basepath + "inputs/vert_spin_splitting/geo.xyz",
            charge=0,
            spin=1,
            basis="def2-tzvp",
            wfnpath2="wrong_path",
        )
    except FileNotFoundError as e:
        assert "wrong_path" in str(e) and "Err" in str(e)
