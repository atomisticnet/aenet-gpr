# ænet-gpr
**Efficient Data Augmentation for ANN Potential Training Using GPR Surrogate Models**

`aenet-gpr` is a Python package that enables scalable and cost-efficient training of artificial neural network (ANN) potentials by leveraging Gaussian Process Regression (GPR) as a surrogate model.  
It automates data augmentation to:

- Reduce the number of expensive DFT calculations  
- Lower ANN training overhead particularly critical for complex and heterogeneous interface systems  
- Maintain high accuracy comparable to the demanding direct force training

📄 Reference:  
[In Won Yeu, Alexander Urban, Nongnuch Artrith et al., “Scalable Training of Neural Network Potentials for Complex Interfaces Through Data Augmentation”, *npj Computational Materials* 11, 156 (2025)](https://doi.org/10.1038/s41524-025-01651-0)

📬 Contact:  
- In Won Yeu (iy2185@columbia.edu)  
- Nongnuch Artrith (n.artrith@uu.nl)

## 🔁 Workflow Overview

<p align="center">
<img src="doc/source/images/0_flowchart.png" width="700">
</p>

1. **Data Grouping**  
   - Split the initial DFT database into homogeneous subsets (same composition and number of atoms)

2. **Train**  
   - Construct local GPR models using structure, energy, and atomic force data of each subset

3. **Test**  
   - Predict and evaluate target properties with the trained GPR models

4. **Augment**  
   - Perturb reference structures and generate new data  
   - Tag with GPR-predicted energies to expand the training dataset

✅ Outputs are saved in [XCrysDen Structure Format (XSF)](http://ann.atomistic.net/documentation/#structural-energy-reference-data), fully compatible with the [ænet package](https://github.com/atomisticnet/aenet-PyTorch) for indirect force training (**GPR-ANN**).

## 🔑 Key Features

- GPR-based prediction of energies and atomic forces with uncertainty estimates  
- Supports various descriptors including Cartesian and SOAP  
- Applicable to periodic and non-periodic systems  
- Batch-based kernel computation for speed and memory efficiency  
- Accepts multiple input formats (e.g., XSF, VASP OUTCAR, etc.)  
- Fully controlled through a single input file (`train.in`)
- Compatible with various GPR applications such as GPR-NEB, GPR-ANN, and ASE-Calculator

## 📦 Installation

**Requirements:**

- Python with PyTorch (to be installed separately, see below)
- Other dependencies (`numpy`, `ASE`) are automatically installed when installing `aenet-gpr`

### 1. Install PyTorch

Refer to [official guide](https://pytorch.org/get-started/locally) and install compatible versions depending on availablity of GPU and CUDA:

   - With CUDA (optional for GPU support):

     `$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

   - CPU-only:
 
     `$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`


### 2. Install ænet-gpr
   
   - Installation using pip

     `$ pip install aenet-gpr`

## 📘 Tutorial

Find interactive notebooks `*.ipynb` in the `./tutorial/` folder, or run directly on Google Colab:

### GPR tutorials for various systems

- [GPR Tutorial: H₂](https://colab.research.google.com/github/atomisticnet/aenet-gpr/blob/main/tutorial/tutorial_1_H2.ipynb)  
- [GPR Tutorial: EC–EC](https://colab.research.google.com/github/atomisticnet/aenet-gpr/blob/main/tutorial/tutorial_2_EC-EC.ipynb)  
- [GPR Tutorial: Li–EC](https://colab.research.google.com/github/atomisticnet/aenet-gpr/blob/main/tutorial/tutorial_3_Li-EC.ipynb)

### GPR applications for accelerating atomistic simulations

- [GPR-ANN: accelerating ANN potential training through GPR data augmentation](https://colab.research.google.com/github/atomisticnet/aenet-gpr/blob/main/tutorial/tutorial_python_GPR-ANN.ipynb)
- [GPR-NEB: accelerating NEB through GPR surrogate model](https://colab.research.google.com/github/atomisticnet/aenet-gpr/blob/main/tutorial/tutorial_python_GPR-NEB.ipynb)
- [GPR-ASE: ASE GPRCalculator](https://colab.research.google.com/github/atomisticnet/aenet-gpr/blob/main/tutorial/tutorial_python_ASE-GPRCalculator.ipynb)

The `./example/` directory includes example input and output data files.

## 📂 Input Files

### 1. Structure–Energy–Force Data

By default, input data is provided in `.xsf` format. 

#### Example: aenet XSF format (non-periodic)
The first comment line should specify **total energy** of a structure. Each line following the keyword `ATOMS` contains **atomic symbol**, **three Cartesian coordinates**, and the three components of **atomic forces**. The length, energy, and force units are Å, eV, and eV/Å.
```
# total energy =  -0.0970905812353288 eV

ATOMS
H    -0.91666666666667    0.00000000000000    0.00000000000000    0.32660398877491    0.00000000000000    0.00000000000000
H    0.91666666666667    0.00000000000000    0.00000000000000    -0.32660398877491    0.00000000000000    0.00000000000000
```

#### Example: aenet XSF format (periodic)
```
# total energy = -16688.9969866290994105 eV

CRYSTAL
PRIMVEC
 10.31700000000000 0.00000000000000 0.00000000000000
 0.00000000000000 10.31700000000000 0.00000000000000
 0.00000000000000 0.00000000000000 32.00000000000000
PRIMCOORD
 46 1
Li     -0.02691046000000     0.02680527000000     10.32468480000000     -0.01367780493112     -0.01466501222916     0.08701630310868
Li     -0.04431013000000     3.46713645000000     10.25290534000000     0.06865473174602     -0.00786890285541     0.15426435842600
Li     0.02355300000000     6.82569825000000     10.31803445000000     0.00877419275000     0.03943267659765     0.14805797440506
...
```

Other formats such as **VASP OUTCAR** (with a line of `File_format vasp-out` in `train.in` below) are also supported as long as they can be read through [ASE](https://wiki.fysik.dtu.dk/ase/ase/io/io.html).

### 2. Configuration File

#### Example: `train.in` (comments are provided to guide the keyword meanings)
```
# File path
Train_file ./example/3_Li-EC/train_set/file_*.xsf
Test_file ./example/3_Li-EC/test_set/file_*.xsf

# File format (default: xsf)
File_format xsf  # Other DFT output files, which can be read via ASE such as "vasp-out" "aims-output" "espresso-out", are also supported

# Uncertainty estimation (default: True)
Get_variance True  # False -> only energy and forces are evaluated without uncertainty estimate

# Descriptor (default: cartesian coordinates)
Descriptor cart  # cart or soap

# Kernel parameter (default: Squared exponential)
scale 0.4  # default: 0.4
weight 1.0  # default: 1.0

# Data process (default: batch, 25)
data_process batch  # batch (memory cost up, time cost down) or iterative (no-batch: memory down, time up)
batch_size 25

# Flags for xsf file writing (default: False)
Train_write False  # True -> xsf files for reference training set are written under "./train_xsf/" directory
Test_write False  # True -> xsf files for reference test set are written under "./test_xsf/" directory
Additional_write False  # True -> additional xsf files are written under "./additional_xsf/" directory; False -> Augmentation step is not executed

# Data augmentation parameter (default: 0.055, 25)
Disp_length 0.05
Num_copy 20  # [num_copy] multiples of reference training data are augmented
```

## 🚀 Usage Example

With the `train.in` file and datasets prepared, simply run:

`$ python -m aenet_gpr ./train.in > train.out`

The **Train–Test–Augment** steps will be executed sequentially. Augmented data will be saved in the `./additional_xsf/` directory.

## 🖥️ Running on an HPC system (SLURM)

To run `aenet_gpr` on an HPC cluster using SLURM, use a batch script like the following:

```
#!/bin/bash
#SBATCH --job-name=aenet-job
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

module load anaconda3
source activate aenet-env

ulimit -s unlimited
python -m aenet_gpr ./train.in > train.out
```

## ⚙️ Tuning Tips

### 1. Accuracy – Descriptor and Kernel Scale Parameter

- Descriptor: **Cartesian**, **SOAP**, and others supported by [DScribe](https://singroup.github.io/dscribe/latest/index.html)
- Default kernel: **Squared Exponential (sqexp)**
- Kernel parameters: **scale** and **weight**

<p align="center">
<img src="doc/source/images/0_kernel.png" width="300">
</p>

Following figure shows energy prediction errors of the `./example/3_Li-EC/` example with different kernel parameters and descriptors.

<p align="center">
<img src="doc/source/images/3_Li-EC_accuracy.png" width="1000">
</p>

When using the **Cartesian descriptor** (gray circles), the error decreases as the `scale` parameter increases, and it converges at `scale = 3.0`. When using the **periodic SOAP descriptor** (for details, see [DScribe documentation](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html)), the error is significantly reduced by one order of magnitude compared to the **Cartesian descriptor**.  

As demonstrated in the examples for the `./example/2_EC-EC/` (results available in the `example` directory), non-periodic systems can be well-represented using **non-periodic Cartesian descriptors**, while periodic systems are expected to yield better accuracy when using **periodic SOAP descriptors**.  

For the example of **SOAP descriptor** here, eight uniformly distributed points in the Li slab Rectangular cuboid were used as `centers` argument for **SOAP**. 

The corresponding `train.in` input arguments are
```
Descriptor soap
soap_r_cut 5.0
soap_n_max 6
soap_l_max 4
soap_centers [[2.20113706670393, 2.328998192856251, 6.952547732109352], [2.20113706670393, 2.328998192856251, 11.895790642109352], [2.20113706670393, 6.760484232856251, 6.952547732109352], [2.20113706670393, 6.760484232856251, 11.895790642109352], [6.63924050670393, 2.328998192856251, 6.952547732109352], [6.63924050670393, 2.328998192856251, 11.895790642109352], [6.63924050670393, 6.760484232856251, 6.952547732109352], [6.63924050670393, 6.760484232856251, 11.895790642109352]]
soap_n_jobs 4  
  
scale 2.0  
weight 1.0
```

### 2. Efficiency – Data Processing Mode

- `data_process iterative`: Computing kernels data-by-data involves `n_data × n_data` sequential kernel evaluations, minimizing the memory overhead but significantly increasing computational time.  

- `data_process batch`: **aenet-gpr** supports batch processing by grouping the data process into a specific size (`batch_size 25`), which significantly reduces train and evaluation time while keeping memory usage efficient.

Below, we provide a benchmark comparing the required time and memory for different batch sizes on the `./example/3_Li-EC/` example.

<p align="center">
<img src="doc/source/images/3_Li-EC_cost.png" width="1000">
</p>
