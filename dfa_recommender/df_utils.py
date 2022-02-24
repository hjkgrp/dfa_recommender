import numpy as np
from dfa_recommender.df_class import DensityFitting


def get_spectra(densfit: DensityFitting, fock: bool = False,
                H: bool = False, t: str = 'alpha') -> np.array:
    '''
    Compute the final power spectrum for the DensityFitting object

    Parameters
    ----------
    densfit: DensityFitting object,
        created from .xyz, .wfn., and basis set
    fock: bool, Optional, default: False
        Fock fitting or not
    H: bool, Optional, default: False
        Hamiltonian (Potential + Kinetics) fitting or not
    t: str, Optional, default: alpha
        alpha or beta spin orbitals

    Returns:
    --------
    powerspec: np.ndarray
       powerspectrum derived from density fitting coefficients
    '''
    if not H:
        if not fock:
            if t == 'alpha':
                D = densfit.wfn.Da()
            elif t == 'beta':
                D = densfit.wfn.Db()
            else:
                raise KeyError("only alpha and beta allowed.")
        else:
            if t == 'alpha':
                D = densfit.wfn.Fa()
            elif t == 'beta':
                D = densfit.wfn.Fb()
            else:
                raise KeyError("only alpha and beta allowed.")
    else:
        D = densfit.wfn.H()
    densfit.get_df_coeffs(D)
    powerspec = densfit.calc_powerspec()
    return powerspec


def get_subtracted_spectra(densfit: DensityFitting,
                           fock=False, t='alpha') -> np.array:
    '''
    Compute the final power spectrum (w.r.t. a second .wfn file) for the DensityFitting object

    Parameters
    ----------
    densfit: DensityFitting object,
        created from .xyz, .wfn., and basis set
    fock: bool, Optional, default: False
        Fock fitting or not
    H: bool, Optional, default: False
        Hamiltonian (Potential + Kinetics) fitting or not
    t: str, Optional, default: alpha
        alpha or beta spin orbitals

    Returns:
    --------
    powerspec: np.ndarray
       powerspectrum derived from density fitting coefficients
    '''

    if not fock:
        if t == 'alpha':
            D1 = densfit.wfn.Da()
        elif t == 'beta':
            D1 = densfit.wfn.Db()
        else:
            raise KeyError("only alpha and beta allowed.")
        if t == 'alpha':
            D2 = densfit.wfn2.Da()
        elif t == 'beta':
            D2 = densfit.wfn2.Db()
        else:
            raise KeyError("only alpha and beta allowed.")
    else:
        if t == 'alpha':
            D1 = densfit.wfn.Fa()
        elif t == 'beta':
            D1 = densfit.wfn.Fb()
        else:
            raise KeyError("only alpha and beta allowed.")
        if t == 'alpha':
            D2 = densfit.wfn2.Fa()
        elif t == 'beta':
            D2 = densfit.wfn2.Fb()
        else:
            raise KeyError("only alpha and beta allowed.")
    delta_D = D1.clone()
    delta_D = delta_D.from_array(D1.to_array() - D2.to_array())
    densfit.get_df_coeffs(delta_D)
    powerspec = densfit.calc_powerspec()
    return powerspec
