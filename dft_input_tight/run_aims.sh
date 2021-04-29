#!/bin/sh
#SBATCH --account=ccce
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=6gb
#SBATCH --threads-per-core=1
#SBATCH --output=aims.out
#SBATCH --error=errors.out



## paste geometry of EC molecule into geometry.in file
## remove previous positions of EC molecule
#cat "geometry_li.in" > geometry.in

## put new position into geometry file
#grep -v "initial" ec.aims > tmpfile ## perform inverted match to grep only those lines that don't match "initial"
#cat "tmpfile" | tail -10 >> geometry.in

module load intel-parallel-studio/2020
ulimit -s unlimited

mpiexec -bootstrap slurm /burg/ccce/users/as6394/Programs/fhi-aims.171221_1/bin/aims.171221_1.mpi.x
#mpiexec -bootstrap slurm /burg/home/na2782/bin/aims.171221_1.mpi.x
