#!/usr/bin/env python

"""
Convert ASE's 'traj' format to aenet's XSF format.

"""

import argparse
import sys

try:
    import ase.io
    import ase.calculators
    import ase.io.trajectory
    import ase.constraints
    has_ase = True
except ImportError:
    print("This converter requires ASE to be installed.")
    print("See https://wiki.fysik.dtu.dk/ase/")
    sys.exit()

__author__ = "Alexander Urban, Nongnuch Artrith"
__email__ = "aurban@atomistic.net, nartrith@atomistic.net"
__date__ = "2020-06-17"
__version__ = "0.1"


def write_xsf(outfile, types, coords, avec=None, energy=None,
              forces=None):
    """
    Write structure information in aenet's extended XSF format.

    Arguments:
      outfile (str): name of the output XSF file
      types (list): list of atomic symbols for each atom
      coords (ndarray): atomic coordinates
      energy (float): total energy of the structure
      ave (ndarray): lattice vectors
      forces (ndarray): interatomic forces

    """
    with open(outfile, 'w') as fp:
        natoms = len(coords)
        pbc = True if avec is not None else False
        if energy is not None:
            fp.write("# total energy = {:.8f} eV\n".format(energy))
            fp.write("\n")
        if pbc:
            fp.write("CRYSTAL\n")
            fp.write("PRIMVEC\n")
            # avec = aux.standard_cell(struc.avec[frame])
            for v in avec:
                fp.write("    {:14.8f} {:14.8f} {:14.8f}\n".format(*v))
            fp.write("PRIMCOORD\n")
            fp.write("{} 1\n".format(natoms))
        else:
            fp.write("ATOMS\n")

        for i in range(natoms):
            fp.write("{:2s} ".format(types[i]))
            fp.write((3*" {:14.8f}").format(coords[i, 0],
                                            coords[i, 1],
                                            coords[i, 2]))
            if forces is not None:
                fp.write((3*" {:14.8f}").format(forces[i, 0],
                                                forces[i, 1],
                                                forces[i, 2]))
            fp.write("\n")


def convert_ase_file(infile, outfile):
    """
    Read atomic structure file in the ASE 'traj' format and convert it
    to aenet's XSF format.  This routine relies on ASE's own parser, so
    ASE has to be installed.

    Arguments:
      infile   name of the input file

    """
    trajec = ase.io.read(infile, index=":")
    types = list(trajec[0].symbols)
    pbc = all(trajec[0].pbc)
    if not pbc:
        avec = None
    w = len(str(len(trajec)))
    if outfile is None:
        outfile_base = "frame_"
    else:
        outfile_base = "".join(outfile.split(".")[:-1])
        if outfile_base == "":
            outfile_base = outfile
    for i, atoms in enumerate(trajec):
        coords = atoms.arrays["positions"]
        if pbc:
            avec = atoms.cell.array
        try:
            energy = atoms.get_potential_energy()
        except RuntimeError:
            energy = None
        try:
            forces = atoms.get_forces()
        except (RuntimeError, ValueError):
            forces = None
        if len(trajec) > 1:
            outfile = outfile_base + "{}.xsf".format(str(i).zfill(w))
        else:
            outfile = outfile_base + ".xsf"
        write_xsf(outfile, types, coords, avec=avec, energy=energy,
                  forces=forces)


if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(
        description=__doc__+"\n{} {}".format(__date__, __author__),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "structure_file",
        help="Input file in ASE-compatible format format.")

    parser.add_argument(
        "output_file",
        help="Name of the output file in XSF format.")

    args = parser.parse_args()

    convert_ase_file(args.structure_file, args.output_file)
