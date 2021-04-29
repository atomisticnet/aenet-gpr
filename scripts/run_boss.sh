#!/bin/sh
#SBATCH --account=ccce
#SBATCH --job-name=boss_6d
#SBATCH --tasks-per-node=18
#SBATCH --time=18:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --nodes=1
#SBATCH --threads-per-core=1

# BO sampling and postprocessing
boss op boss.in

# Pure BO sampling
#boss o boss.in

# Pure postprocessing
#boss p modified.rst boss.out

# Restart from previous BOSS run: the latest *.rst file is employed as input (initpts should be augmented to the total number of points already acquired)
#boss op modified.rst
