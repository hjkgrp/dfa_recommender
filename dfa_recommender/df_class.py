'''
A set of utility functions for density fitting.
Some Credits to: https://gitlab.com/jmargraf/kdf
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
        self.calc_utilities()

    def __str__(self) -> None:
        return f'wfnpath: {self.wfnpath}\nxyzfile: {self.xyzfile}\nbasis: {self.basis}'

    @property
    def wfnpath(self) -> None:
        return self._wfnpath

    @wfnpath.setter
    def wfnpath(self, wfnpath: str) -> None:
        '''
        set wfn and orb attribute
        '''
        self._wfnpath = wfnpath
        self.wfn = psi4.core.Wavefunction.from_file(self._wfnpath)
        assert isinstance(self.wfn, psi4.core.Wavefunction)
        self.orb = self.wfn.basisset()

    @property
    def wfnpath2(self) -> None:
        return self._wfnpath2

    @wfnpath2.setter
    def wfnpath2(self, wfnpath2: str) -> None:
        '''
        set wfn2 attribute. wfn2 is the wfn of the other electronic state at the same geometry
        '''
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
        '''
        set psi4 molecule and symbols attribute
        '''
        self._xyzfile = xyzfile
        self.mol, self.symbols = get_molecule(self._xyzfile, self.charge, self.spin)
        assert isinstance(self.mol, psi4.core.Molecule)

    def construct_aux(self) -> None:
        '''
        Load files, transform them to utility varibles, and check data types
        '''
        #psi4.core.set_global_option('df_basis_scf', "def2-universal-JFIT")
        self.aux = psi4.core.BasisSet.build(self.mol, "DF_BASIS_SCF", "", "JKFIT", self.basis)

    def get_dab(self) -> None:
        '''
        Build dab_P tensor as tensor before contracting to aux coeffiecients (np.ndarray)
        '''
        zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
        mints = psi4.core.MintsHelper(self.orb)
        #abQ = mints.ao_eri(self.orb, self.orb, self.aux, zero_bas)
        #Jinv = mints.ao_eri(zero_bas, self.aux, zero_bas, self.aux)
        abQ = mints.ao_eri(self.aux, zero_bas, self.orb, self.orb)
        Jinv = mints.ao_eri(self.aux, zero_bas, self.aux, zero_bas)
        Jinv.power(-1.0, 1.e-14)
        abQ = np.squeeze(abQ)
        self.Jinv = np.squeeze(Jinv)
        self.dab_P = np.einsum('Qab,QP->abP', abQ, self.Jinv, optimize=True)

    def calc_utilities(self) -> None:
        '''
        Calculate the shell numbers and number of basis functions in each shell.
        '''
        self.numfuncatom = np.zeros(self.mol.natom())
        shells = []
        for func in range(0, self.aux.nbf()):
            current = self.aux.function_to_center(func)
            shell = self.aux.function_to_shell(func)
            shells.append(shell)
            self.numfuncatom[current] += 1

        self.shellmap = []
        ii = 0
        tmp, tmp_count = [], 0
        for shell in range(self.aux.nshell()):
            count = shells.count(shell)
            tmp_count += count
            tmp.append((count-1)//2)
            if tmp_count == self.numfuncatom[ii]:
                ii += 1
                self.shellmap.append(tmp)
                tmp = []
                tmp_count = 0

    def get_df_coeffs(self, D : psi4.core.Matrix) -> None:
        '''
        Calculate the raw density fitting coefficients.
        '''
        self.C_P = np.einsum('abP,ab->P', self.dab_P, D, optimize=True)

    def compensate_charges(self):
        '''
        Compensate charges in the density fitting.
        NOTE that currently only work for alpha.
        '''
        q = []
        counter = 0
        shellmap = np.concatenate(self.shellmap)
        for i in range(self.mol.natom()):
            for j in range(counter, counter + int(self.numfuncatom[i])):
                shell_num = self.aux.function_to_shell(j)
                shell = self.aux.shell(shell_num)
                # assumes that each shell only has 1 primitive. true for a2 basis
                normalization = shell.coef(0)
                exponent = shell.exp(0)
                if shellmap[shell_num] == 0:
                    integral = (1/(4*exponent))*np.sqrt(np.pi/exponent)
                    q.append(4*np.pi*normalization*integral)
                else:
                    q.append(0.0)
                counter += 1
        q = np.array(q)*0.5
        bigQ = self.wfn.nalpha()
        # compute lambda
        numer = bigQ - np.dot(q, self.C_P*2)
        # print(bigQ, np.dot(q, self.C_P))
        denom = np.dot(np.dot(q, self.Jinv),q)
        lambchop = numer/denom
        self.C_P += np.dot(self.Jinv, lambchop*q)*0.5
        # print("sum q: ", np.sum(q))
        # print("bigQ: ", bigQ)
        # print(numer, denom, lambchop)

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

    def pad_df_coeffs(self) -> None:
        '''
        Convert self.C_P (a 1D array) to self.self.C_P_pad (N_atoms x M),
        where M corresponds to the largest dim of coeffs of all atoms.
        For example, H2O at def2-universal-jkfit basis has 113 coeffs.
        H -> [0, 0, 1, 1, 2, 2] -> 18 coeffs
        O -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4] -> 77 coeffs
        Then self.self.C_P_pad is a (3, 77) np.array with zero-padding at corresponding irreps
        '''
        self.max_shell = np.array(self.shellmap[np.argmax(self.numfuncatom)])
        self.n_coeff_atom = int(np.max(self.numfuncatom))
        self.max_tmplate = [0]
        self.irreps = []
        for ii in range(np.max(self.max_shell) + 1):
            self.max_tmplate.append((self.max_shell == ii).sum()*(2*ii + 1) + self.max_tmplate[-1])
            evenodd = "e" if ii%2==0 else "o"
            self.irreps += [str((self.max_shell == ii).sum()) + "x" + str(ii) + evenodd]
        self.irreps = "+".join(self.irreps)

        tmplates = []
        for jj in range(len(self.shellmap)):
            tmplate = [0]
            _shell = np.array(self.shellmap[jj])
            for ii in range(np.max(_shell)):
                tmplate.append((_shell == ii).sum()*(2*ii + 1) + tmplate[-1])
            tmplate.append(int(self.numfuncatom[jj]))
            tmplates.append(tmplate)

        self.C_P_pad = np.zeros(shape=(self.numfuncatom.shape[0], self.n_coeff_atom))
        count = 0
        for ii, tmplate in enumerate(tmplates):
            for jj in range(np.max(self.shellmap[ii])+1):
                _end = min(self.max_tmplate[jj] + tmplate[jj+1] - tmplate[jj], self.max_tmplate[jj+1])
                self.C_P_pad[ii, self.max_tmplate[jj]: _end] = self.C_P[(count + tmplate[jj]): (count +tmplate[jj+1])]
            count += int(self.numfuncatom[ii])

    def convert_CP2e3nn(self) -> None:
        '''
        match m between psi4 and e3nn convension within the same l.
        For example, for l=1,
        psi4 m: [0, 1, -1]
        e3nn m: [-1, 0, 1]
        For example, for l=2,
        psi4 m: [0, 1, -1, 2, -2]
        e3nn m: [-2, -1, 0, 1, 2]
        '''

        psi4_2_e3nn = [[0],[2,0,1],[4,2,0,1,3],[6,4,2,0,1,3,5],[8,6,4,2,0,1,3,5,7]]
        self.C_P_pad_e3nn = np.zeros(shape=self.C_P_pad.shape)
        for ii in range(self.C_P_pad.shape[0]):
            for jj, ele in enumerate(self.irreps.split("+")):
                num, l = int(ele.split("x")[0]),  int(ele.split("x")[1][0])
                coeffs = self.C_P_pad[ii][self.max_tmplate[jj]: self.max_tmplate[jj+1]]
                coeffs = np.array_split(coeffs, num)
                coeffs_trans = []
                for coeff in coeffs:
                    for k in psi4_2_e3nn[l]:
                        coeffs_trans.append(coeff[k])
                self.C_P_pad_e3nn[ii][self.max_tmplate[jj]: self.max_tmplate[jj+1]] = coeffs_trans
        mat = np.where(np.abs(self.C_P_pad_e3nn)< 1e-8)
        self.C_P_pad_e3nn[mat[0], mat[1]] = 0


