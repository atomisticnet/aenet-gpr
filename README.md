# ænet-GPR
Automated Workflow for Data Augmentation: Gaussian Process Regression (GPR) Surrogate Models for machine-learning potential (MLP) training.

This Python package is designed to support Artificial Neural Network (ANN) force training by using Gaussian Process Regression (GPR) as a surrogate model, aiming to reduce the computational cost for the training, as benchmarked in the reference: [In Won Yeu, Annika Stuke, Jon L.pez-Zorrilla, James M. Stevenson, David R. Reichman, Richard A. Friesner, Alexander Urban, Nongnuch Artrith, "Scalable Training of Neural Network Potentials for Complex Interfaces Through Data Augmentation", arXiv:2412.05773](
https://doi.org/10.48550/arXiv.2412.05773)  

*Contact: In Won Yeu (iy2185@columbia.edu) or Nongnuch Artrith (n.artrith@uu.nl)  

## Overall workflow of GPR surrogate model
<p align="center">
<img src="doc/source/images/0_flowchart.png" width="700">
</p>

Starting with an initial DFT database, training begins with dividing an existing database into homogeneous subsets (structures with the same composition and number of atoms). Then, following three main steps are automatically performed:  

**1. Train** – Construct local GPR models from reference training subset data, **structure-energy-atomic forces**  
**2. Test** – Predict and evaluate target properties for test structures using the trained GPR model  
**3. Augmentation** – Generate additional structures by perturbing reference training structures and augment the ANN training dataset with GPR-predicted energy tags  

The augmented data are saved in the [XCrysDen Structure Format (XSF)](http://ann.atomistic.net/documentation/#structural-energy-reference-data) file format, compatible with the [aenet package](https://github.com/atomisticnet/aenet-PyTorch), so that the output can be readily integrated as input for ANN potential training enabling indirect force training (**GPR-ANN training**).

Here, the local GPR models are used for the local approximation of the potential energy surface (PES) and the finer PES sampling. The surrogate models can also be used for active learning based on the GPR uncertanty estimate. 

# Table of Contents
* [Installation](#installation)
* [Input files](#input-files)
* [Usage example](#usage-example)
* [Key train.in input keywords](#key-keyword)


<a name="installation"></a>
# Installation
(Requirements) In a working Python environment, following packages should be pre-installed:
* `numpy`: `pip install numpy`
* `torch`: `pip install torch`
* `dscribe`: `pip install dscribe`
* `ase`: `pip install ase`

```
$ pip install aenet-gpr
```

<a name="input-files"></a>
# Input files
## 1. Data files of structure-energy-atomic forces
First, data files containing **structure-energy-atomic forces** must be prepared as training data. By default, the package uses the XSF format, but it also supports other output files—such as **VASP OUTCAR** (`File_format vasp-out` in `train.in` below)—that can be read via [ASE package](https://wiki.fysik.dtu.dk/ase/ase/io/io.html), as long as they contain the **structure-energy-atomic forces** information.

### Example of aenet XSF file of a non-periodic structure:
The first comment line should specify **total energy** of a structure. Each line following the keyword `ATOMS` contains **atomic symbol**, **three Cartesian coordinates**, and the three components of **atomic forces**. The length, energy, and force units are Å, eV, and eV/Å.
```
# total energy =  -0.0970905812353288 eV

ATOMS
H    -0.91666666666667    0.00000000000000    0.00000000000000    0.32660398877491    0.00000000000000    0.00000000000000
H    0.91666666666667    0.00000000000000    0.00000000000000    -0.32660398877491    0.00000000000000    0.00000000000000
```

### Example of aenet XSF file of a periodic structure:
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
Li     3.45933501000000     0.11713693000000     10.39313682000000     -0.05428281161189     -0.13370451564945     0.01502154141400
Li     3.48234757000000     3.45628769000000     10.28938796000000     -0.15390178334724     0.03988563746333     0.15264739590144
Li     3.38451113000000     6.89570814000000     10.32283587000000     -0.16931117144381     0.03419338883993     -0.13537461156890
Li     6.87070897000000     -0.03542177000000     10.35291072000000     0.02788426697501     0.08085039956890     -0.02733266500243
Li     6.91731097000000     3.52549517000000     10.32404846000000     -0.07702924505736     0.00182342017871     0.49539990490405
Li     6.84983915000000     6.86523283000000     10.29295423000000     0.12351312205521     0.04041307532165     0.04432539187174
Li     1.68182068000000     1.78211962000000     8.57642412000000     0.02194527288864     -0.11735081008363     -0.04078838313415
Li     1.71278006000000     5.13703886000000     8.55659408000000     0.04500163183234     0.00602246184502     -0.02201071625761
Li     1.69015722000000     8.57793150000000     8.63699138000000     0.02739032376838     0.08268801966662     -0.12082853518471
Li     5.08608831000000     1.75532529000000     8.66801601000000     0.15244644345286     -0.04679590945894     -0.19233057346870
Li     5.12715894000000     5.20935278000000     8.63087543000000     0.01424255377127     -0.04568006293527     -0.30977966877767
Li     5.20505840000000     8.53732110000000     8.57702008000000     -0.10904619701776     0.13404000740742     0.00602741640574
Li     8.55449106000000     1.77946655000000     8.55255233000000     0.06555974834028     -0.11399016093657     -0.01997425494877
Li     8.61255442000000     5.20357553000000     8.59499606000000     -0.03500676278100     -0.01245628342065     -0.21087556567684
Li     8.59049917000000     8.53920638000000     8.63762633000000     0.00927208947906     0.04639568260529     -0.06979613167385
Li     -0.03558816000000     -0.06090360000000     6.81386356000000     0.08474154059200     0.12844028186333     0.29304027482122
Li     0.04260970000000     3.40077760000000     6.85199787000000     -0.09356260516690     0.08898508923662     0.07447905449118
Li     -0.05316644000000     6.84706025000000     6.87308006000000     0.12035796367987     0.00336721855450     0.10536652362077
Li     3.41863780000000     0.06956215000000     6.85408454000000     0.01770852636794     -0.11041021416635     0.16746244855722
Li     3.42635758000000     3.37380964000000     6.85388333000000     0.08240133306353     0.12306435572541     0.14101105369523
Li     3.41696311000000     6.88798005000000     6.92497904000000     0.01001231303297     0.03780980044033     0.01773695238250
Li     6.90483862000000     0.00617009000000     6.90983269000000     -0.08371059222725     -0.00935865638536     0.09592531048756
Li     6.92619171000000     3.41782081000000     6.93310774000000     -0.13185578517054     0.10104161477236     0.10718300807265
Li     6.87886445000000     6.95216093000000     6.81295995000000     -0.04033062139472     -0.04544260197815     0.23187761212689
Li     1.76866752000000     1.75663386000000     5.18414952000000     -0.05291629349347     -0.08017270176088     -0.20077865390125
Li     1.78395317000000     5.13642683000000     5.16275394000000     -0.04832040347782     0.03422119701945     -0.09190432520406
Li     1.65384589000000     8.58462454000000     5.17668674000000     0.06966039979574     -0.01595299637772     -0.12492763436392
Li     5.19623604000000     1.74353238000000     5.17862204000000     0.01906603203220     0.01456313361113     -0.11371062240387
Li     5.16993902000000     5.23412474000000     5.15128053000000     -0.03678297965232     -0.06786483673801     -0.05828104221775
Li     5.12675933000000     8.65721743000000     5.10601807000000     0.00450720018044     -0.00898253240536     -0.05285136941407
Li     8.59619628000000     1.66987261000000     5.19955251000000     0.01232378428582     0.04206027458414     -0.16787078739431
Li     8.50714416000000     5.22252071000000     5.19930580000000     0.08661871805998     -0.08081203650577     -0.15711607493901
Li     8.66380165000000     8.62905013000000     5.25811808000000     -0.10906753836554     -0.04101019900582     -0.25199734902797
C     5.82067597000000     7.51680180000000     12.99180227000000     -1.49781578054140     -2.36842290754468     5.45815652564390
C     4.59713822000000     6.67699265000000     13.70676619000000     -1.14167326226764     3.88098998187132     0.53147057270491
O     6.26206075000000     6.52853024000000     12.18555160000000     1.64764877673398     -3.15549152166039     -1.31518510377666
H     5.47617490000000     8.29331793000000     12.46445014000000     -1.36661445539386     2.88122399291250     -2.13192195698546
H     6.57622218000000     7.86285331000000     13.72684633000000     -0.22024739267517     -0.38130945039949     -0.45407085119078
O     4.92235787000000     5.35319460000000     13.61347149000000     3.15286983449669     -0.86694838776146     -2.46978359740753
H     3.62016175000000     6.92314193000000     13.17506670000000     1.08930914584153     -0.49365717402185     0.30699432883803
H     4.41443226000000     6.93885315000000     14.77088909000000     0.41980029289196     -0.05588314983771     -0.63295020560789
C     5.96527716000000     5.28827675000000     12.66196442000000     -0.94905207614540     -3.52120754795978     -0.76875635941493
O     6.36114737000000     4.20557578000000     12.18179519000000     -0.99750467595151     3.95392686199225     1.50773308706430
```

## 2. aenet-GPR input file
A principal input file, named `train.in`, consists of lines in the format: 

`keyword 1` `argument 1`  
`keyword 2` `argument 2`  
...

In the example below, comments are provided to explain the meaning of keywords.

### Example of train.in
```
# File path
Train_file ./example/3_Li-EC/train_set/file_*.xsf
Test_file ./example/3_Li-EC/test_set/file_*.xsf

# Train model save (default: False)
Train_model_save False  # True-> train data and trained GPR model are saved in "data_dict.pt" and "calc_dict.pt"

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

<a name="usage-example"></a>
# Usage example
For example, once the `train.in` file above is prepared along with 100 training data files, named `file_{0000..0099}.xsf`, in the directory `./example/3_Li-EC/train_set/` and 300 test data files, named `file_{0000..0299}.xsf` in the directory `./example/3_Li-EC/test_set/`, **aenet-GPR** is executed by the following command:

```
$ python [path of aenet_GPR]/aenet_gpr.py ./train.in > train.out
```

Then, the **Train–Test–Augmentation** steps will be carried out sequentially. The progress details can be monitored in the `train.out` file, and the final augmented data files will be saved in XSF format under `./additional_xsf/` directory.  

The `./example/` directory of this repository includes example input and output files.


<a name="key-keyword"></a>
## Key `train.in` input keywords that affect performance
### 1. Accuracy: `Descriptor` and kernel parameter (`scale` and `weight`)
**aenet-GPR** uses the following `squared exponential (sqexp)` as default kernel function with `scale` and `weight` parameters:

<p align="center">
<img src="doc/source/images/0_kernel.png" width="300">
</p>

To achieve more accurate data augmentation, it is recommended to tune the `scale` and `weight` parameters for each specific system.

Perform a series of **Train–Test** runs using a small subset of training and test data, while varying the `scale` and `weight`. This process helps identify the optimal kernel parameters. Following figure shows energy prediction errors of the `./example/3_Li-EC/` example with different kernel parameters and descriptors:

<p align="center">
<img src="doc/source/images/3_Li-EC_accuracy.png" width="1000">
</p>

`train.in` input file of default arguments
```
Descriptor cart  
scale 0.4  
weight 1.0
```
are shown above. When using the **Cartesian descriptor** (gray circles), the error decreases as the `scale` parameter increases, and it converges at `scale = 3.0`. In contrast, when using the **periodic SOAP descriptor** (for details, see the [DScribe documentation](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html)), the error is reduced by approximately one order of magnitude compared to the default **Cartesian descriptor**.  

As demonstrated in the examples for the `./example/2_EC-EC/` non-periodic system (results available in the `example` directory) and the `./example/3_Li-EC/` periodic system, non-periodic systems can be well-represented using **non-periodic Cartesian descriptors**, while periodic systems are expected to yield better accuracy when using **SOAP descriptors** with periodic setting.  

For the example of **SOAP descriptor** here, eight uniformly distributed points in the Li slab Rectangular cuboid were used as `centers` argument for **SOAP**. The corresponding `train.in` input arguments are
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

### 2. Cost: `data_process` and `batch_size`
The GPR model fitted to both energy and force data requires computing covariances between the fingerprint tensors of shape `[n_data, n_center, n_feature]` and the [fingerprint derivative](https://singroup.github.io/dscribe/latest/tutorials/derivatives.html) tensors of shape `[n_data, n_center, n_atom, 3, n_feature]`. This leads to high memory demands.  

On the other hand, computing kernels data-by-data (`data_process iterative`) involves `n_data × n_data` sequential kernel evaluations, minimizing the memory overhead but significantly increasing computational time.  

To address this, **aenet-GPR** supports batch processing (`data_process batch`) by grouping data process into a specific size (`batch_size 25`), which significantly reduces train and evaluation time while keeping memory usage efficient.

Below, we provide a benchmark comparing the required time and memory for each **Train–Test–Augmentation** step using different batch sizes on the `./example/3_Li-EC/` example.

<p align="center">
<img src="doc/source/images/3_Li-EC_cost.png" width="1000">
</p>
