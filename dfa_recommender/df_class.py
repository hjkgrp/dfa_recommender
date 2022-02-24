'''
A set of utility functions for density fitting.
Credit: https://gitlab.com/jmargraf/kdf
'''

import numpy as np
import psi4
import os
from typing import Tuple

def get_molecule(xyzfile: str, charge: int, spin: int, sym: str = 'c1') -> Tuple[psi4.core.Molecule, list]:
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
    else:
        raise FileNotFoundError("No file named : ", xyzfile)
    wholetext += "\nsymmetry %s\nnoreorient\nnocom\n" % sym
    mol = psi4.geometry("""%s""" % wholetext)
    return mol, symbols


class DensityFitting:
    '''
    Density fitting class to project the electron density onto auxiliary basis sets.
    '''

    def __init__(self, wfnpath: str, xyzfile: str, basis: str,
                 charge: int = 0, spin: int = 1, wfnpath2: str = 'NA') -> None:
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
        '''
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.wfnpath = wfnpath
        self.wfnpath2 = wfnpath2
        self.xyzfile = xyzfile
        self.construct_aux()
        self.get_dab()


    def __str__(self) -> None:
        return f'wfnpath: {self.wfnpath}\nxyzfile: {self.xyzfile}\nbasis: {self.basis}'

    @property
    def wfnpath(self) -> None:
        return self._wfnpath
        
    @wfnpath.setter
    def wfnpath(self, wfnpath: str) -> None:
        self._wfnpath = wfnpath
        self.wfn = psi4.core.Wavefunction.from_file(self._wfnpath)
        assert isinstance(self.wfn, psi4.core.Wavefunction)
        self.orb = self.wfn.basisset()

    @property
    def wfnpath2(self) -> None:
        return self._wfnpath2
        
    @wfnpath2.setter
    def wfnpath2(self, wfnpath2: str) -> None:
        self._wfnpath2 = wfnpath2
        if wfnpath2 == "NA":
            pass
        elif os.path.isfile(wfnpath2):
            self.wfn2 = psi4.core.Wavefunction.from_file(self._wfnpath2)
            assert isinstance(self.wfn2, psi4.core.Wavefunction)
        else:
            raise FileNotFoundError("wfn file is not availbale: ", wfnpath2)

    @property
    def xyzfile(self) -> None:
        return self._mol

    @xyzfile.setter
    def xyzfile(self, xyzfile: str) -> None:
        self._xyzfile = xyzfile
        self.mol, self.symbols = get_molecule(self._xyzfile, self.charge, self.spin)
        assert isinstance(self.mol, psi4.core.Molecule)

    def construct_aux(self) -> None:
        '''
        Load files, transform them to utility varibles, and check data types
        '''
        self.aux = psi4.core.BasisSet.build(self.mol, "DF_BASIS_SCF", "", "JKFIT", self.basis)

    def get_dab(self) -> None:
        '''
        Build dab_P tensor as tensor before contracting to aux coeffiecients (np.ndarray) 
        '''
        zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
        mints = psi4.core.MintsHelper(self.orb)
        abQ = mints.ao_eri(self.orb, self.orb, self.aux, zero_bas)
        Jinv = mints.ao_eri(zero_bas, self.aux, zero_bas, self.aux)
        Jinv.power(-1.0, 1.e-14)
        abQ = np.squeeze(abQ)
        Jinv = np.squeeze(Jinv)
        self.dab_P = np.einsum('abQ,QP->abP', abQ, Jinv, optimize=True)
    
    def get_df_coeffs(self, D : psi4.core.Matrix) -> None:
        self.C_P = np.einsum('abP,ab->P', self.dab_P, D, optimize=True)

    def calc_powerspec(self) -> np.array:
        '''
        Calculates powerspectrum to yeild a invariant representation from density fitting coefficients
        
        Returns
        ----------
        powerspec: np.ndarray
            powerspectrum derived from density fitting coefficients. 
        '''
        shells = []
        shells_to_at = []
        preshell = -1
        currshell = []
        for i_bf in range(self.aux.nbf()):
            f2s = self.aux.function_to_shell(i_bf)
            f2c = int(self.aux.function_to_center(i_bf))
            if f2s == preshell:
                currshell.append(i_bf)
            else:
                shells.append(currshell)
                currshell = [i_bf]
                preshell = f2s
                shells_to_at.append(f2c)
        shells.append(currshell)

        powerspec = []
        preat = -1
        currat = []
        for i_s, shell in enumerate(shells[1:]):
            p = np.sum(self.C_P[shell]**2)
            if shells_to_at[i_s] == preat:
                currat.append(p)
            else:
                powerspec.append(currat)
                currat = [p]
                preat = shells_to_at[i_s]
        powerspec.append(currat)
        return np.array(powerspec[1:])
