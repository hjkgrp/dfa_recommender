'''
A set of utility functions for density fitting.
Credit: https://gitlab.com/jmargraf/kdf
'''

import numpy as np
import psi4
import os

def get_molecule(xyzfile, charge, spin, sym='c1'):
    '''
    Assemble a molecule object from xyzfile, charge and spin.

    Parameters
    ----------
    xyzfile: str,
        path to the xyz file of the input molecule.
    charge: int,
        charge of the input molecule.
    spin: int,
        spin multiplicity (2*S + 1) for the input molecule
    sym: str, Optional, default: c1
        point group symmetry of the input molecule

    Returns
    ----------
    mol: psi4.geometry object
       psi4.geometry object for the input molecule
    symbols: list
        list of atom symbols 
    '''
    wholetext = "%s %s\n" % (charge, spin)
    symbols = []
    if os.path.isfile(xyzfile):
        with open(xyzfile, "r") as fo:
            natoms = int(fo.readline().split()[0])
            fo.readline()
            for ii in range(natoms):
                line = fo.readline()
                wholetext += line
                symbols.append(line.split()[0])
    wholetext += "\nsymmetry %s\nnoreorient\nnocom\n"%sym
    mol = psi4.geometry("""%s""" % wholetext)
    return mol, symbols


def get_dab(orb, aux):
    '''
    Build dab_P tensor

    Parameters
    ----------
    orb: psi4.core.BasisSet object,
        basis set used your obtaining the wfn
    aux: psi4.core.BasisSet object,
        auxiliary basis set built for your molecule for density fitting

    Returns
    ----------
    dab_P: np.ndarray,
        tensor before contracting to aux coeffiecients
    
    '''
    zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
    mints = psi4.core.MintsHelper(orb)
    abQ = mints.ao_eri(orb, orb, aux, zero_bas)
    Jinv = mints.ao_eri(zero_bas, aux, zero_bas, aux)
    Jinv.power(-1.0, 1.e-14)
    abQ = np.squeeze(abQ)
    Jinv = np.squeeze(Jinv)
    dab_P = np.einsum('abQ,QP->abP', abQ, Jinv, optimize=True)
    return dab_P


def get_utils(wfnpath, xyzfile, basis, charge=0, spin=1, cal_dabP=True):
    '''
    Calculate a set of variables used in the final density fitting stage

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
    cal_dab_P: bool, Optional, default: True
        whether to calculate the dab_P tensor

    Returns
    ----------
    orb: psi4.core.BasisSet object,
        basis set used your obtaining the wfn
    aux: psi4.core.BasisSet object,
        auxiliary basis set built for your molecule for density fitting
    dab_P: np.ndarray,
        tensor before contracting to aux coeffiecients
    wfn: psi4.core.Wavefunction object,
        wave function loaded from wfnpath
    symbols: list
        list of atom symbols
    '''

    wfn = psi4.core.Wavefunction.from_file(wfnpath)
    orb = wfn.basisset()
    mol, symbols = get_molecule(xyzfile, charge=charge, spin=spin, sym='c1')
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", basis)
    if not cal_dabP:
        return orb, aux
    dab_P = get_dab(orb,aux)
    return orb, aux, dab_P, wfn, symbols


def get_DF_coeff(orb, aux, D, dab_P=None):
    '''
    Calculates density fitting coefficients in aux basis given original orb basis and density matrix D.

    Parameters
    ----------
    orb: psi4.core.BasisSet object,
        basis set used your obtaining the wfn
    aux: psi4.core.BasisSet object,
        auxiliary basis set built for your molecule for density fitting
    D: psi4.core.Matrix
        density matrix
    dab_P: np.ndarray, Optional, defaul: None
        tensor before contracting to aux coeffiecients 
    
    Returns
    ---------
    C_P: np.ndarray,
        density fitting coefficients
    '''
    if not isinstance(dab_P, np.ndarray):
        zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
        mints = psi4.core.MintsHelper(orb)

        # Build (ab|Q) raw 3-index ERIs, dimension (nbf, nbf, nAux, 1)
        abQ = mints.ao_eri(orb, orb, aux, zero_bas)

        # Build and invert the coulomb metric
        Jinv = mints.ao_eri(zero_bas, aux, zero_bas, aux)
        Jinv.power(-1.0, 1.e-14)

        # Remove the excess dimensions of abQ & metric
        abQ = np.squeeze(abQ)
        Jinv = np.squeeze(Jinv)

        dab_P = np.einsum('abQ,QP->abP', abQ, Jinv, optimize=True)

    C_P = np.einsum('abP,ab->P', dab_P, D, optimize=True)
    return C_P


def calc_powerspec(aux,C_P):
    '''
    Calculates powerspectrum to yeild a invariant representation from density fitting coefficients.

    Parameters
    ----------
    aux: psi4.core.BasisSet object,
        auxiliary basis set built for your molecule for density fitting
    C_P: np.array
        density fitting coefficients
    
    Returns
    ----------
    powerspec: np.array
       powerspectrum derived from density fitting coefficients. 
    '''
    shells = []
    shells_to_at = []
    preshell = -1
    currshell = []
    for i_bf in range(aux.nbf()):
        f2s = aux.function_to_shell(i_bf)
        f2c = int(aux.function_to_center(i_bf))
        if f2s == preshell:
            currshell.append(i_bf)
        else:
            shells.append(currshell)
            currshell = [i_bf]
            preshell  = f2s
            shells_to_at.append(f2c)
    shells.append(currshell)

    powerspec = []
    preat = -1
    currat = []
    for i_s,shell in enumerate(shells[1:]):
        p = np.sum(C_P[shell]**2)
        if shells_to_at[i_s] == preat:
            currat.append(p)
        else:
            powerspec.append(currat)
            currat = [p]
            preat  = shells_to_at[i_s]
    powerspec.append(currat)
    return np.array(powerspec[1:])


def get_spectra(orb, aux, dab_P, wfn, symbols, fock=False, H=False, t='alpha'):
    '''
    Function to call when we have the density fitting variables obtained from get_utils

    Parameters
    ----------
    orb: psi4.core.BasisSet object,
        basis set used your obtaining the wfn
    aux: psi4.core.BasisSet object,
        auxiliary basis set built for your molecule for density fitting
    dab_P: np.ndarray,
        tensor before contracting to aux coeffiecients
    wfn: psi4.core.Wavefunction object,
        wave function loaded from wfnpath
    symbols: list
        list of atom symbols
    fock: bool, Optional, default: False
        Fock fitting or not
    H: bool, Optional, default: False
        Hamiltonian (Potential + Kinetics) fitting or not
    t: str, Optional, default: alpha
        alpha or beta spin orbitals
    
    Returns:
    --------
    powerspec: np.array
       powerspectrum derived from density fitting coefficients.
    symbols: list
        list of atom symbols
    '''
    if not H:
        if not fock:
            if t == 'alpha':
                D = wfn.Da()
            elif t == 'beta':
                D = wfn.Db()
            else:
                raise KeyError("only alpha and beta allowed.")
        else:
            if t == 'alpha':
                D = wfn.Fa()
            elif t == 'beta':
                D = wfn.Fb()
            else:
                raise KeyError("only alpha and beta allowed.")
    else:
        D = wfn.H()
    C_P = get_DF_coeff(orb,aux,D,dab_P=dab_P)
    powerspec = calc_powerspec(aux,C_P)
    return powerspec, symbols


def get_subtracted_spectra(orb, aux, dab_P, wfn1, wfn2, symbols, fock=False, t='alpha'):
    '''
    Similar to get_spectra, but act on the difference of electron density at the *same* geometry
    
    Parameters
    ----------
    orb: psi4.core.BasisSet object,
        basis set used your obtaining the wfn
    aux: psi4.core.BasisSet object,
        auxiliary basis set built for your molecule for density fitting
    dab_P: np.ndarray,
        tensor before contracting to aux coeffiecients
    wfn1: psi4.core.Wavefunction object,
        wave function for the first calculation
    wfn2: psi4.core.Wavefunction object,
        wave function for the second calculation
    symbols: list
        list of atom symbols
    fock: bool, Optional, default: False
        Fock fitting or not
    t: str, Optional, default: alpha
        alpha or beta spin orbitals
    
    Returns:
    --------
    powerspec: np.array
       powerspectrum derived from density fitting coefficients.
    symbols: list
        list of atom symbols
    '''
    if not fock:
        if t == 'alpha':
            D1 = wfn1.Da()
        elif t == 'beta':
            D1 = wfn1.Db()
        else:
            raise KeyError("only alpha and beta allowed.")
        if t == 'alpha':
            D2 = wfn2.Da()
        elif t == 'beta':
            D2 = wfn2.Db()
        else:
            raise KeyError("only alpha and beta allowed.")
    else:
        if t == 'alpha':
            D1 = wfn1.Fa()
        elif t == 'beta':
            D1 = wfn1.Fb()
        else:
            raise KeyError("only alpha and beta allowed.")
        if t == 'alpha':
            D2 = wfn2.Fa()
        elif t == 'beta':
            D2 = wfn2.Fb()
        else:
            raise KeyError("only alpha and beta allowed.")
    delta_D = D1.clone()
    delta_D = delta_D.from_array(D1.to_array() - D2.to_array())
    C_P = get_DF_coeff(orb,aux,delta_D,dab_P=dab_P)
    powerspec = calc_powerspec(aux,C_P)
    return powerspec, symbols