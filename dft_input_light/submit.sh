#!/bin/sh

## cut out initial charges from geometry file
grep -v "initial" geometry_tmp.in > geometry.in ## perform inverted match to grep only those lines that don't match "initial"


date
# run DFT
sbatch --wait run_aims.sh #&
date
