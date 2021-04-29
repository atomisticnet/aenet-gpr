from ase import Atoms
import numpy as np
from ase.io import write
import os
from create_geometry import create_geometry
import subprocess
from subprocess import *

def f(X):

    ####################### SET JOB PARAMETERS HERE ####################
    wrkdir = '/burg/ccce/users/as6394/aenet_gpr/' # Work directory
    ####################################################################


    ## input variables from BOSS
    a = X[0][0]
    b = X[0][1]
    #z = X[0][2]
    alpha = X[0][2]
    beta = X[0][3]
    gamma = X[0][4]

    a_float = "%.6f" % a
    b_float = "%.6f" % b
    alpha_float = "%.6f" % alpha
    beta_float = "%.6f" % beta
    gamma_float = "%.6f" % gamma

    # Define calculation directory for current input variables
    calcdir = wrkdir + 'energy_calc/energy_' + a_float + '_' + b_float  + '_' + alpha_float + '_' + beta_float + '_' + gamma_float + '/'
    print(calcdir)

    # Create calculation directory and copy DFT input files in there
    os.system('mkdir ' + calcdir)
    os.system('cp ' + wrkdir + 'dft_input_light/control.in ' + calcdir)
    os.system('cp ' + wrkdir + 'dft_input_light/submit.sh ' + calcdir)
    os.system('cp ' + wrkdir + 'dft_input_light/run_aims.sh ' + calcdir)

    # Create geometry.in file calling python script
    create_geometry(a, b, alpha, beta, gamma)
    # Move geometry.in file to calculation directory
    os.system('mv ' + wrkdir + 'input_geometries/geometry_tmp.in ' + calcdir)

    # go into calc_dir
    os.system('cd ' + calcdir)# + '; submit.sh ')

    ## call bash script for DFT simulation
    command_file = os.path.join(calcdir, "submit.sh")
    subprocess.call(command_file, cwd=calcdir)


    # Get total energy from AIMS output
    with open(calcdir + 'aims.out') as f:
        for line in f:
            if line.startswith('  | Total energy corrected'):
                E = float(line.split()[5])


    return E
