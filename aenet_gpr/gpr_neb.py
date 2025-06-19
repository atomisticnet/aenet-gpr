import copy
import os
import glob

from ase import io
from ase.calculators.emt import EMT

from aenet_gpr.tool.aidneb import AIDNEB
# from aenet_gpr.tool import pbc_align, pbc_group, pbc_wrap


# NEB parameters
neb_F_max = 0.05  # criteria of the force for the convergence.
neb_interpol = "idpp"  # "linear", "idpp", path to trajectory file
neb_n_images = 10  # total images including initial and final
use_mic = False


def main():
    initial = io.read("../00-initial/initial.traj")
    final = io.read("../00-final/final.traj")

    # if wrap_species is not None:
    #     initial = pbc_wrap(initial, species=wrap_species, translate=True)
    #     final = pbc_wrap(final, species=wrap_species, translate=True)
    #
    # if atom_groups is not None:
    #     # make atom index start with 0 instead of 1
    #     for grp in atom_groups:
    #         for i in range(len(grp)):
    #             grp[i] -= 1
    #     initial = pbc_group(initial, atom_groups)
    #     final = pbc_group(final, atom_groups)
    #
    # if align_species is not None:
    #     final = pbc_align(final, initial, species=align_species)

    calc = EMT()

    neb = AIDNEB(start=initial,
                 end=final,
                 interpolation=neb_interpol,
                 n_images=neb_n_images,
                 mic=use_mic,
                 max_train_data=100,  # max_train_data
                 calculator=copy.deepcopy(calc),
                 use_previous_observations=True)

    # if use_mic:
    #     neb.images[-1] = pbc_align(
    #         neb.images[-1], neb.images[-2], species=align_species)
    #     neb.initial_interpolation = neb.images[:]

    io.write("initial_images.traj", neb.images)
    neb.run(fmax=neb_F_max, unc_convergence=0.1, dt=0.05, ml_steps=200)


if __name__ == "__main__":
    main()
