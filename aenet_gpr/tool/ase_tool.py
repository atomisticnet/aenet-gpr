"""
Collection of utilities for the ASE Tools package.
"""

from subprocess import run, PIPE
import numpy as np
from copy import deepcopy


__author__ = "Alexander Urban; Jianzhou Qu; In Won Yeu"
__email__ = "aurban@atomistic.net"
__date__ = "2025-07-05"
__version__ = "1.0"


def get_structure_uncertainty(unc_e: np.array, unc_f: np.array, method="combined"):

    if method == "energy":
        return unc_e
    elif method == "force_max":
        return unc_f.max()
    elif method == "force_mean":
        return unc_f.mean()
    elif method == "combined":
        return np.sqrt(unc_e**2 + unc_f.mean()**2)


def prepare_neb_images(atoms_init, atoms_final, species=None, pair1=(0, 1), pair2=(0, 2)):
    """
    Prepares two PBC structures for NEB by aligning periodic images and wrapping into the unit cell.

    Parameters
    ----------
    atoms_init : Atoms
        initial structure.
    atoms_final : Atoms
        final structure.
    species : list or set, optional
        Subset of species to use for alignment/wrapping.
    pair1 : tuple, list or set, optional
        (a, b): main direction vector to align with z-axis
    pair2 : tuple, list or set, optional
        (a, c): secondary vector to define the xz-plane
    """

    is_init_pbc = all(atoms_init.pbc)
    is_final_pbc = all(atoms_final.pbc)

    if is_init_pbc and is_final_pbc:
        # Step 1: wrap both structures back into the unit cell
        aligned_init = pbc_wrap(atoms_init, species=species)
        aligned_final = pbc_wrap(atoms_final, species=species)

        # Step 2: Match periodic images
        aligned_final = pbc_match_images(aligned_final, aligned_init)

    elif not is_init_pbc and not is_final_pbc:
        # Non-periodic molecule: align by center
        aligned_init = align_molecule_by_two_vectors(atoms_init, pair1=pair1, pair2=pair2)
        aligned_final = align_molecule_by_two_vectors(atoms_final, pair1=pair1, pair2=pair2)

    else:
        raise ValueError("Both structures must be either periodic or non-periodic.")

    return aligned_init, aligned_final


def pbc_wrap(atoms, species=None, translate=False, eps=1.0e-6):
    """
    Wrap coordinates back into the unit cell.

    Arguments:
      atoms (Atoms): Structure to be wrapped
      species (list or set): Optional list of atomic symbols (e.g., ['O', 'H']) to use for center calculation.
                           If None, use all atoms.
      translate (bool): If True, translate the structure such that the geometric center of all atoms
      (or all selected atoms if species is set) lies exactly in the center of the unit cell.
      After wrapping, the reverse shift will be applied so that the returned structure is compatible with the input structure.
      eps (float): Numerical precision to be used for comparison

    """
    # if not all(atoms.pbc):
    #     return
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

    if hasattr(new_atoms, "calc") and new_atoms._calc is not None:
        new_atoms._calc.atoms.positions[:] = new_atoms.positions[:]
    return new_atoms


def pbc_align(atoms, reference, species=None):
    """
    Return a new Atoms object where each atom is shifted into the same periodic image
    as its counterpart in the reference structure.

    Parameters
    ----------
    atoms : The structure to be aligned.
    reference : The reference structure. Must be the same size and atom order as `atoms`.
    species : list or set, optional
        A list or set of atomic symbols to limit which atoms are aligned.
        If None, all atoms are considered.

    Returns
    -------
    aligned_atoms : Atoms
        A new Atoms object with adjusted atomic positions such that atoms are in
        the same periodic image as in the reference structure.
    """
    # if not all(atoms.pbc):
    #     return
    if len(atoms) != len(reference):
        raise ValueError("Input atoms and reference must contain the same number of atoms.")

    aligned_atoms = deepcopy(atoms)

    if species is not None:
        species = set(species)

    recip = aligned_atoms.cell.reciprocal().T
    cell = np.array(aligned_atoms.cell)

    for i, (sym, pos_ref, pos) in enumerate(zip(reference.symbols, reference.positions, aligned_atoms.positions)):
        if species is not None and sym not in species:
            continue
        # Vector from atoms to reference
        delta = pos_ref - pos
        # Compute how many unit cells to shift
        shift_frac = np.round(np.dot(delta, recip))
        # Convert back to cartesian and apply shift
        shift_cart = np.dot(shift_frac, cell)
        aligned_atoms.positions[i] += shift_cart

    # Update calculator if present
    if aligned_atoms._calc is not None:
        aligned_atoms._calc.atoms.positions[:] = aligned_atoms.positions[:]

    return aligned_atoms


def pbc_match_images(atoms, reference, species=None):
    """
    Adjust atoms' positions such that each selected atom is moved to the periodic image
    closest to its counterpart in the reference structure.

    Parameters
    ----------
    atoms : Atoms
        Structure to be shifted.
    reference : Atoms
        Reference structure (same length and atomic order).
    species : list or set, optional
        Only atoms with matching chemical symbols will be shifted.
        If None, all atoms are considered.

    Returns
    -------
    aligned_atoms : Atoms
        Atoms object with positions adjusted to match reference images.
    """
    if not all(atoms.pbc):
        raise ValueError("pbc_match_images requires full periodic boundary conditions.")

    if len(atoms) != len(reference):
        raise ValueError("atoms and reference must contain the same number of atoms.")

    if species is not None:
        species = set(species)

    aligned_atoms = deepcopy(atoms)
    cell = np.array(atoms.cell)
    recip = atoms.cell.reciprocal().T  # reciprocal lattice vectors

    for i in range(len(atoms)):
        if species is not None and atoms.symbols[i] not in species:
            continue  # skip atoms not in species list

        delta = reference.positions[i] - aligned_atoms.positions[i]
        shift_frac = np.round(np.dot(delta, recip))  # periodic shift in fractional coordinates
        shift_cart = np.dot(shift_frac, cell)  # convert back to Cartesian
        aligned_atoms.positions[i] += shift_cart

    # Update attached calculator if necessary
    if aligned_atoms._calc is not None:
        aligned_atoms._calc.atoms.positions[:] = aligned_atoms.positions[:]

    return aligned_atoms


def align_molecule_by_two_vectors(atoms_target, pair1, pair2):
    """
    Align atoms_target using two direction vectors:
    - pair1 = (a, b): main direction vector to align with z-axis
    - pair2 = (a, c): secondary vector to define the xz-plane
    Also moves atom a to origin.
    """
    a, b = pair1
    _, c = pair2

    new_atoms = deepcopy(atoms_target)
    pos = new_atoms.get_positions()

    # Compute local frame in target
    v1 = pos[b] - pos[a]  # main direction (to z)
    v2 = pos[c] - pos[a]  # secondary direction

    z_axis = v1 / np.linalg.norm(v1)
    x_temp = v2 - np.dot(v2, z_axis) * z_axis  # project v2 to plane orthogonal to z
    x_axis = x_temp / np.linalg.norm(x_temp)
    y_axis = np.cross(z_axis, x_axis)

    R = np.vstack([x_axis, y_axis, z_axis]).T  # rotation matrix from local to global frame

    # Apply inverse rotation to align local frame to global axes
    rotated = (pos - pos[a]) @ R
    new_atoms.set_positions(rotated)

    return new_atoms


def align_molecule_centers(atoms_ref, atoms_target, species=None):
    """
    Translate `atoms_target` such that its center of geometry matches that of `atoms_ref`.

    Parameters:
    -----------
    atoms_ref (Atoms): Reference structure
    atoms_target (Atoms): Structure to be shifted
    species (list or set): Optional list of atomic symbols (e.g., ['O', 'H']) to use for center calculation.
                           If None, use all atoms.

    Returns:
    --------
    new_atoms (Atoms): Shifted copy of `atoms_target` aligned to `atoms_ref`
    """

    new_atoms = deepcopy(atoms_target)

    if species is None:
        idx_ref = np.arange(len(atoms_ref))
        idx_target = np.arange(len(new_atoms))
    else:
        idx_ref = [i for i, s in enumerate(atoms_ref.get_chemical_symbols()) if s in species]
        idx_target = [i for i, s in enumerate(new_atoms.get_chemical_symbols()) if s in species]

    # Compute centers
    center_ref = atoms_ref.get_positions()[idx_ref].mean(axis=0)
    center_target = new_atoms.get_positions()[idx_target].mean(axis=0)

    shift = center_ref - center_target
    new_atoms.translate(shift)

    if hasattr(new_atoms, "calc") and new_atoms._calc is not None:
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
