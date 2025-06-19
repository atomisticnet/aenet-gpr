"""
Collection of utilities for the ASE Tools package.
"""

from subprocess import run, PIPE
import numpy as np
from copy import deepcopy


__author__ = "Alexander Urban; Jianzhou Qu"
__email__ = "aurban@atomistic.net"
__date__ = "2021-03-29"
__version__ = "0.1"


def csv2list(csvlist):
    """
    Convert list of comma-separated integers to an actual list.

    Ranges are defined by ":".  For example:

       [1, "2", "3:7", "8,9:11"]

    will be converted to

       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    """

    def expand_range(r):
        if ":" not in r:
            return [int(r)]
        i0, i1 = [int(s) for s in r.split(":")]
        return list(range(i0, i1 + 1))

    lst = []
    for item in csvlist:
        try:
            lst.append([int(item)])
        except ValueError:
            sublst = []
            for item2 in item.split(","):
                sublst += expand_range(item2)
            lst.append(sublst)
    return lst


def pbc_align(atoms, reference, species=None):
    """
    Align an atomic structure with a reference by ensuring that the same
    periodic images of all atoms are used.

    Arguments:
      atoms (Atoms): Structure to be aligned
      reference (Atoms): Reference structure to be aligned with
      species (list): list of chemical symbols of the species that should
        be considered for alignment

    """
    if not all(atoms.pbc):
        return
    for i, coo in enumerate(reference.positions):
        if species is not None and atoms.symbols[i] not in species:
            continue
        vec = coo - atoms.positions[i]
        vec = np.round(np.dot(vec, atoms.cell.reciprocal().T))
        vec = np.dot(vec, np.array(atoms.cell))
        atoms.positions[i] += vec
    if atoms._calc is not None:
        atoms._calc.atoms.positions[:] = atoms.positions[:]
    return atoms


def pbc_group(atoms, atom_groups):
    """
    Select periodic images of grouped atoms such that all grouped atoms
    are closest to each other.

    Arguments:
      atoms (Atoms): Structure to be aligned
      atom_groups (list of lists): Lists with grouped atoms
    """
    if not all(atoms.pbc):
        return

    def group(at, ref):
        """ Select periodic image of 'at' closest to 'ref'. """
        coo = atoms.positions[at]
        coo_ref = atoms.positions[ref]
        vec = coo_ref - coo
        vec = np.round(np.dot(vec, atoms.cell.reciprocal().T))
        vec = np.dot(vec, np.array(atoms.cell))
        return coo + vec

    for grp in atom_groups:
        for i in grp[1:]:
            atoms.positions[i] = group(i, grp[0])
    if atoms._calc is not None:
        atoms._calc.atoms.positions[:] = atoms.positions[:]
    return atoms


def pbc_wrap(atoms, species=None, translate=False, eps=1.0e-6):
    """
    Wrap coordinates back into the unit cell.

    Arguments:
      atoms (Atoms): Structure to be wrapped
      species (list): List of chemical symbols that should be considered
        for wrapping
      translate (bool): If True, translate the structure such that the
        geometric center of all atoms (or all selected atoms if species is
        set) lies exactly in the center of the unit cell.  After wrapping,
        the reverse shift will be applied so that the returned structure
        is compatible with the input structure.
      eps (float): Numerical precision to be used for comparison

    """
    if not all(atoms.pbc):
        return
    # deepcopy is needed to preserve calculators
    new_atoms = deepcopy(atoms)
    if species is None:
        species = set(new_atoms.symbols)
    frac_coords = new_atoms.get_scaled_positions()
    if translate:
        # translate geometric center to the center of the unit cell
        idx = [i for i, s in enumerate(new_atoms.symbols) if s in species]
        C = np.sum(frac_coords[idx], axis=0) / len(idx)
        shift = np.array([0.5, 0.5, 0.5]) - C
    else:
        shift = np.array([0.0, 0.0, 0.0])
    frac_coords += shift
    for iatom, coo in enumerate(frac_coords):
        if new_atoms.symbols[iatom] in species:
            for i in range(3):
                while (coo[i] < 0.0):
                    coo[i] += 1.0
                while (coo[i] >= (1.0 - eps)):
                    coo[i] -= 1.0
    frac_coords -= shift
    new_atoms.set_scaled_positions(frac_coords)
    if new_atoms._calc is not None:
        new_atoms._calc.atoms.positions[:] = new_atoms.positions[:]
    return new_atoms


def runcmd(command, maxtime=300):
    """
    Run cmd command based on `bash`.
    Args:
        command: String
            Can be the same way in command line.
        maxtime: Int
            The maximum time (second) for running this command.
    """
    runit = run(
        command, stdout=PIPE, stderr=PIPE, shell=True, encoding="utf-8",
        executable="/bin/bash", timeout=maxtime
    )
    if runit.returncode == 0:
        print("Succeed to run command", command)
    else:
        print("Error:", runit)
    return None
