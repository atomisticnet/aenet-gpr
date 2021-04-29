## Create geometry.in file for Li|EC system

from ase import Atoms
import numpy as np
from ase.data.pubchem import pubchem_atoms_search, pubchem_atoms_conformer_search
from ase.visualize import view
from ase.io import write, read
import os

def create_geometry(da, db, dalpha, dbeta, dgamma):

    ########## Set parameters
    nx = 6# supercell size
    ny = 6
    z = 4


    # Structure of lithium slab
    li_slab = read("input_geometries/geometry_li.in")

    # Original structure of EC molecule
    ec = read("input_geometries/ec_default.xyz")

    # Move lithium slab so that upper sheet is at z=0
    # highest Li atom is z=1.947644508604
    li_slab.translate((0, 0, -li_slab.get_positions()[-1,2]))
    print(li_slab.get_positions())

    # Rotate molecule 
    ec.rotate(dalpha, 'x', center=(0, 0, 0))
    ec.rotate(dbeta, 'y', center=(0, 0, 0))
    ec.rotate(dgamma, 'z', center=(0, 0, 0))

    ## translate center of mass of the molecule to vector (0,0,z)
    initial_position = ec.get_center_of_mass()
    target_position = [0, 0, z]
    translate_vector = np.subtract(target_position, initial_position)
    ec.translate(translate_vector)

    #print("Center of mass after translation:", ec.get_center_of_mass())
    #print("Atomic positions after translation:", ec.get_positions())

    ## Calculate translation (dx, dy) of EC molecule
    ## Supercell lattice vectors
    va = li_slab.get_cell()[0]
    vb = li_slab.get_cell()[1]

    print("supercell lattice vectors:")
    print(va)
    print(vb)

    ## Unit cell vectors
    ua = va/nx
    ub = vb/ny
    print("unit cell vectors:")
    print(ua)
    print(ub)

    dx = da * ua[0]
    dy = db * ub[1]


    # Translate molecule
    ec.translate((dx, dy, 0.0))

    
    # Combine geometries of Li slab and EC
    li_slab.extend(ec)
    write('geometry_tmp.in', li_slab)
