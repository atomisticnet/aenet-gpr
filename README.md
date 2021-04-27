# aenet-gpr

Automated Workflow for Data Generation: Gaussian Process Regression (GPR) Surrogate Models for machine-learning potential (MLP) training.

## Overall workflow

Starting with an initial DFT database in step 1.  This could be a small database consisting of only some DFT calculations.

1. Initial DFT database
2. Construction of a surrugate model based on GPR; refine by extending the DFT database of step 1
3. Sampling of the potential energy surface; generation of reference data for the MLP training
4. Training of the MLP
5. Validation of the MLP

## GPR surrogate model

A GPR surrogate model is used for the local approximation of the potential energy surface (PES).  This approach can be implemented based on different GPR frameworks, and one example is the [Bayesian Optimization Structure Search (BOSS)](https://gitlab.com/cest-group/boss). Once the GPR has learned the local PES, it is used to generate reference data (atomic structures and energies) for the training of the MLP.  For [ænet](http://ann.atomistic.net) artificial neural network (ANN) potentials, the reference data needs to be generated in a modified version of the [XCrysDen Structure Format (XSF)](http://ann.atomistic.net/documentation/#structural-energy-reference-data).

- Approximate local PES (for example, Bayesian sampling with BOSS)
- Generate XSF files with structures and energies

BOSS implements GPR for the energies only.  The next step will therefor be an extension to the GPR model implemented in [CatLearn](https://github.com/SUNCAT-Center/CatLearn), which also supports fitting gradient (i.e., the atomic forces).  There is also a version of the [atomic structure environment (ASE)](https://wiki.fysik.dtu.dk/ase/) that implements the gradient GPR and could be used as an alternative to CatLearn. 

