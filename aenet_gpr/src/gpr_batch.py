import torch
import numpy as np
import os

from copy import deepcopy
from joblib import Parallel, delayed

from aenet_gpr.src.prior import ConstantPrior
from aenet_gpr.src.pytorch_kernel import FPKernel
from aenet_gpr.util.prepare_data import get_N_batch, get_batch_indexes_N_batch, DescriptorStandardizer


def chebyshev_descriptor_gradient(atoms, model, atoms_mask, delta=1e-3, dtype=torch.float64):
    """
    Args:
        atoms (ase.Atoms):
        model: Chebyshev calculator
        atoms_mask: Atom index to keep
        delta (float): displacement (Ang)
        dtype (dtype):

    Returns:
        desc (torch.Tensor): (n_reduced_atoms, descriptor_dim)
        grad (torch.Tensor): (n_reduced_atoms, n_reduced_atoms, 3, descriptor_dim)
    """
    # Get original descriptor
    batch_positions = [(torch.tensor(atoms.positions, dtype=dtype, device=model.featurizer.device))]
    original_positions = atoms.get_positions()

    for i in atoms_mask:
        for j in range(3):
            # Forward perturbation
            pos_f = original_positions.copy()
            pos_f[i, j] += delta
            batch_positions.append(torch.tensor(pos_f, dtype=dtype, device=model.featurizer.device))

            # Backward perturbation
            pos_b = original_positions.copy()
            pos_b[i, j] -= delta
            batch_positions.append(torch.tensor(pos_b, dtype=dtype, device=model.featurizer.device))

    batch_species = [atoms.get_chemical_symbols() for i in range(len(batch_positions))]

    # (Natoms * (6 * Nmaks + 1), Ndescriptor)
    features_batch, batch_indices = model(batch_positions,
                                          batch_species)

    n_atoms = len(atoms)
    n_mask = len(atoms_mask)
    desc = features_batch[:n_atoms]  # (Natoms, Ndescriptor)

    n_atoms, n_features = desc.shape

    # Pre-allocate gradient array
    grad = torch.empty((n_atoms, n_atoms, 3, n_features), dtype=dtype, device=model.featurizer.device)

    for atom_idx, i in enumerate(atoms_mask):
        # 6 perturbations: +x, -x, +y, -y, +z, -z
        desc_f_x = features_batch[n_atoms + n_atoms * (atom_idx * 6 + 0): n_atoms + n_atoms * (atom_idx * 6 + 1)]
        desc_b_x = features_batch[n_atoms + n_atoms * (atom_idx * 6 + 1): n_atoms + n_atoms * (atom_idx * 6 + 2)]
        desc_f_y = features_batch[n_atoms + n_atoms * (atom_idx * 6 + 2): n_atoms + n_atoms * (atom_idx * 6 + 3)]
        desc_b_y = features_batch[n_atoms + n_atoms * (atom_idx * 6 + 3): n_atoms + n_atoms * (atom_idx * 6 + 4)]
        desc_f_z = features_batch[n_atoms + n_atoms * (atom_idx * 6 + 4): n_atoms + n_atoms * (atom_idx * 6 + 5)]
        desc_b_z = features_batch[n_atoms + n_atoms * (atom_idx * 6 + 5): n_atoms + n_atoms * (atom_idx * 6 + 6)]

        # Central difference: shape = (Natoms, Ndescriptor)
        d_dx = (desc_f_x - desc_b_x) / (2 * delta)
        d_dy = (desc_f_y - desc_b_y) / (2 * delta)
        d_dz = (desc_f_z - desc_b_z) / (2 * delta)

        # (Natoms, 3, Ndescriptor) → assign to dfp_dr[idx, :, i, :, :]
        grad[:, i, 0, :] = d_dx
        grad[:, i, 1, :] = d_dy
        grad[:, i, 2, :] = d_dz

    # n_atoms -> n_reduced_atoms
    desc = desc[atoms_mask, :]
    grad = grad[atoms_mask, :, :, :]
    grad = grad[:, atoms_mask, :, :]

    return desc, grad


def chebyshev_descriptor_gradient_periodic(atoms, model, atoms_mask, delta=1e-3, dtype=torch.float64):
    """
    Args:
        atoms (ase.Atoms):
        model: Chebyshev calculator
        atoms_mask: Atom index to keep
        delta (float): displacement (Ang)
        dtype (dtype):

    Returns:
        desc (torch.Tensor): (n_reduced_atoms, descriptor_dim)
        grad (torch.Tensor): (n_reduced_atoms, n_reduced_atoms, 3, descriptor_dim)
    """
    # Get original descriptor
    batch_positions = [(torch.tensor(atoms.positions, dtype=dtype, device=model.featurizer.device))]
    original_positions = atoms.get_positions()

    for i in atoms_mask:
        for j in range(3):
            # Forward perturbation
            pos_f = original_positions.copy()
            pos_f[i, j] += delta
            batch_positions.append(torch.tensor(pos_f, dtype=dtype, device=model.featurizer.device))

            # Backward perturbation
            pos_b = original_positions.copy()
            pos_b[i, j] -= delta
            batch_positions.append(torch.tensor(pos_b, dtype=dtype, device=model.featurizer.device))

    batch_species = [atoms.get_chemical_symbols() for i in range(len(batch_positions))]
    batch_cells = [torch.tensor(np.array([atoms.cell[i] for i in range(3)]), device=model.featurizer.device) for i in
                   range(len(batch_positions))]
    batch_pbc = [torch.tensor(atoms.pbc, dtype=torch.bool, device=model.featurizer.device) for i in range(len(batch_positions))]

    # (Natoms * (6 * Nmaks + 1), Ndescriptor)
    features_batch, batch_indices = model(batch_positions,
                                          batch_species,
                                          batch_cells=batch_cells,
                                          batch_pbc=batch_pbc)

    n_atoms = len(atoms)
    n_mask = len(atoms_mask)
    desc = features_batch[:n_atoms]

    n_atoms, n_features = desc.shape

    # Pre-allocate gradient array
    grad = torch.empty((n_atoms, n_atoms, 3, n_features), dtype=dtype, device=model.featurizer.device)

    for atom_idx, i in enumerate(atoms_mask):
        # 6 perturbations: +x, -x, +y, -y, +z, -z
        desc_f_x = features_batch[n_atoms + n_atoms * (atom_idx * 6 + 0): n_atoms + n_atoms * (atom_idx * 6 + 1)]
        desc_b_x = features_batch[n_atoms + n_atoms * (atom_idx * 6 + 1): n_atoms + n_atoms * (atom_idx * 6 + 2)]
        desc_f_y = features_batch[n_atoms + n_atoms * (atom_idx * 6 + 2): n_atoms + n_atoms * (atom_idx * 6 + 3)]
        desc_b_y = features_batch[n_atoms + n_atoms * (atom_idx * 6 + 3): n_atoms + n_atoms * (atom_idx * 6 + 4)]
        desc_f_z = features_batch[n_atoms + n_atoms * (atom_idx * 6 + 4): n_atoms + n_atoms * (atom_idx * 6 + 5)]
        desc_b_z = features_batch[n_atoms + n_atoms * (atom_idx * 6 + 5): n_atoms + n_atoms * (atom_idx * 6 + 6)]

        # Central difference: shape = (Natoms, Ndescriptor)
        d_dx = (desc_f_x - desc_b_x) / (2 * delta)
        d_dy = (desc_f_y - desc_b_y) / (2 * delta)
        d_dz = (desc_f_z - desc_b_z) / (2 * delta)

        # (Natoms, 3, Ndescriptor) → assign to dfp_dr[idx, :, i, :, :]
        grad[:, i, 0, :] = d_dx
        grad[:, i, 1, :] = d_dy
        grad[:, i, 2, :] = d_dz

    # n_atoms -> n_reduced_atoms
    desc = desc[atoms_mask, :]
    grad = grad[atoms_mask, :, :, :]
    grad = grad[:, atoms_mask, :, :]

    return desc, grad


def mace_descriptor_gradient(atoms, model, atoms_mask, delta=1e-5, invariants=True, num_layers=-1, n_jobs=-1,
                             dtype='float32'):
    """
    Optimized batch version - pre-generates all perturbed positions and processes them efficiently

    Args:
        atoms (ase.Atoms):
        model: MACE calculator
        atoms_mask: Atom index to keep
        delta (float): displacement (Ang)
        num_layers (int): MACE interaction layers number
        n_jobs (int): number of processes
        dtype (dtype):

    Returns:
        desc (torch.Tensor): (n_reduced_atoms, descriptor_dim)
        grad (torch.Tensor): (n_reduced_atoms, n_reduced_atoms, 3, descriptor_dim)
    """
    # Get original descriptor
    atoms_mask = atoms_mask.cpu().numpy()
    desc = model.get_descriptors(atoms, invariants_only=invariants, num_layers=num_layers)
    n_atoms, D = desc.shape

    # Pre-allocate gradient array
    grad = np.empty((n_atoms, n_atoms, 3, D), dtype=dtype)

    # Get original positions once
    original_positions = atoms.get_positions()

    # Pre-create all perturbations
    perturbations = []
    for i in atoms_mask:
        for j in range(3):  # x, y, z
            # Forward perturbation
            pos_f = original_positions.copy()
            pos_f[i, j] += delta
            perturbations.append(('f', i, j, pos_f))

            # Backward perturbation
            pos_b = original_positions.copy()
            pos_b[i, j] -= delta
            perturbations.append(('b', i, j, pos_b))

    if n_jobs == 1:
        # Compute all descriptors
        all_descriptors = []
        for direction, i, j, positions in perturbations:
            atoms_temp = deepcopy(atoms)
            atoms_temp.set_positions(positions)
            desc_temp = model.get_descriptors(atoms_temp, invariants_only=invariants, num_layers=num_layers)
            all_descriptors.append((direction, i, j, desc_temp))
    else:
        # --- Parallelize descriptor computation ---
        def compute_descriptor(direction, i, j, positions):
            atoms_temp = deepcopy(atoms)
            atoms_temp.set_positions(positions)
            desc_temp = model.get_descriptors(atoms_temp, invariants_only=invariants, num_layers=num_layers)
            return direction, i, j, desc_temp

        all_descriptors = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(compute_descriptor)(direction, i, j, positions)
            for (direction, i, j, positions) in perturbations
        )

    # Reorganize and compute gradients
    desc_dict = {}
    for direction, i, j, desc_temp in all_descriptors:
        key = (i, j)
        if key not in desc_dict:
            desc_dict[key] = {}
        desc_dict[key][direction] = desc_temp[:, :]

    # Compute central differences
    for (i, j), descs in desc_dict.items():
        grad[:, i, j, :] = (descs['f'] - descs['b']) / (2 * delta)

    # n_atoms -> n_reduced_atoms
    desc = desc[atoms_mask, :]
    grad = grad[atoms_mask, :, :, :]
    grad = grad[:, atoms_mask, :, :]

    return desc, grad


def apply_force_mask(F, atoms_xyz_mask):
    """
    Args:
        F: (Ntest, 3*Natoms) force tensor
        atoms_xyz_mask: tensor([...]) flattened xyz indices to keep

    Returns:
        F_masked: same shape as F, masked with zeros outside atoms_mask
    """

    mask = torch.zeros(F.shape[1], dtype=torch.bool, device=F.device)
    mask[atoms_xyz_mask] = True

    F_masked = torch.zeros_like(F)
    F_masked[:, mask] = F[:, mask]
    return F_masked


class GaussianProcess(object):
    '''
    Gaussian Process Regression
    Parameters:

    prior: Defaults to ConstantPrior with zero as constant

    kernel: Defaults to the Squared Exponential kernel with derivatives
    '''

    def __init__(self, hp=None, prior=None, prior_update=True, kerneltype='sqexp',
                 scale=0.4, weight=1.0, noise=1e-6, noisefactor=0.5,
                 use_forces=True, images=None, function=None, derivative=None,
                 sparse=None, sparse_derivative=None, autograd=False,
                 train_batch_size=25, eval_batch_size=25,
                 data_type='float64', device='cpu',
                 soap_param=None, mace_param=None, cheb_param=None,
                 descriptor='cartesian coordinates', descriptor_standardization=False,
                 atoms_mask=None):

        if data_type == 'float32':
            self.data_type = 'float32'
            self.torch_data_type = torch.float32
        else:
            self.data_type = 'float64'
            self.torch_data_type = torch.float64

        self.device = device

        self.soap_param = soap_param
        self.mace_param = mace_param
        self.cheb_param = cheb_param

        self.descriptor = descriptor
        self.descriptor_standardization = descriptor_standardization
        self.kerneltype = kerneltype

        self.scale = torch.tensor(scale, dtype=self.torch_data_type, device=self.device)
        self.weight = torch.tensor(weight, dtype=self.torch_data_type, device=self.device)

        self.noise = torch.tensor(noise, dtype=self.torch_data_type, device=self.device)
        self.noisefactor = torch.tensor(noisefactor, dtype=self.torch_data_type, device=self.device)

        self.use_forces = use_forces
        self.images = images
        self.Ntrain = len(self.images)
        self.species = self.images[0].get_chemical_symbols()
        self.pbc = np.all(self.images[0].get_pbc())
        self.Natom = len(self.species)

        if self.descriptor == 'soap':
            try:
                from dscribe.descriptors import SOAP
                print("You are using SOAP descriptor through DScribe:")
                print("[1] A. P. Bartók, R. Kondor and G. Csányi, Phys. Rev. B 87 (2013) 184115.")
                print("[2] L. Himanen, A. S Foster et al., Comput. Phys. Commun. 247 (2020) 106949. \n")
                print("SOAP parameter:")
                print(self.soap_param)
                print("\n")
            except ImportError:
                raise ImportError(
                    "The 'dscribe' package is required for using SOAP descriptors.\n"
                    "Please install it by running:\n\n"
                    "    pip install dscribe\n")

            self.soap = SOAP(species=set(self.species),
                             periodic=self.pbc,
                             r_cut=self.soap_param.get('r_cut'),
                             n_max=self.soap_param.get('n_max'),
                             l_max=self.soap_param.get('l_max'),
                             sigma=self.soap_param.get('sigma'),
                             rbf=self.soap_param.get('rbf'),
                             dtype=self.data_type,
                             sparse=self.soap_param.get('sparse'))

        elif self.descriptor == 'mace':
            if os.path.isfile(self.mace_param.get('model')):
                try:
                    from mace.calculators import MACECalculator
                    print("You are using pre-trained MACE descriptor:")
                    print(
                        "[1] I. Batatia, D. P Kovacs, G. Simm, C. Ortner, and G. Csányi, Adv. Neural Inf. Process. Syst. 35 (2022) 11423.")
                    print("[2] I. Batatia, G. Csányi et al., arXiv:2401.00096 (2023). \n")
                    print("MACE parameter:")
                    print(self.mace_param)
                    print("\n")
                except ImportError:
                    raise ImportError(
                        "The 'mace' package is required for using pre-trained MACE descriptors.\n"
                        "Please install it by running:\n\n"
                        "    pip install mace-torch\n"
                    )

                # Check and set device
                if torch.cuda.is_available():
                    print(
                        "[Note] There is available CUDA device, and it will be used for MACE descriptor computation.\n")
                    mace_device = "cuda:0"
                else:
                    mace_device = "cpu"
                    print("[Warning] CUDA device not available. MACE descriptor computation may be slow on CPU.\n")

                self.mace = MACECalculator(model_paths=[self.mace_param.get('model')],
                                           device=mace_device)
                print("\nUsing MACE for MACECalculator with ", self.mace_param.get('model'))

            else:
                if self.mace_param.get('system') == "materials":
                    try:
                        from mace.calculators import mace_mp
                        print("You are using pre-trained MACE-MP descriptor:")
                        print(
                            "[1] I. Batatia, D. P Kovacs, G. Simm, C. Ortner, and G. Csányi, Adv. Neural Inf. Process. Syst. 35 (2022) 11423.")
                        print("[2] I. Batatia, G. Csányi et al., arXiv:2401.00096 (2023). \n")
                        print("MACE parameter:")
                        print(self.mace_param)
                        print("\n")
                    except ImportError:
                        raise ImportError(
                            "The 'mace' package is required for using pre-trained MACE descriptors.\n"
                            "Please install it by running:\n\n"
                            "    pip install mace-torch\n"
                        )

                    # Check and set device
                    if torch.cuda.is_available():
                        print(
                            "[Note] There is available CUDA device, and it will be used for MACE descriptor computation.\n")
                        mace_device = "cuda:0"
                    else:
                        mace_device = "cpu"
                        print("[Warning] CUDA device not available. MACE descriptor computation may be slow on CPU.\n")

                    self.mace = mace_mp(model=self.mace_param.get('model'),
                                        device=mace_device)

                else:
                    try:
                        from mace.calculators import mace_off
                        print("You are using pre-trained MACE-OFF descriptor:")
                        print(
                            "[1] I. Batatia, D. P Kovacs, G. Simm, C. Ortner, and G. Csányi, Adv. Neural Inf. Process. Syst. 35 (2022) 11423.")
                        print("[2] I. Batatia, G. Csányi et al., arXiv:2401.00096 (2023). \n")
                        print("MACE parameter:")
                        print(self.mace_param)
                        print("\n")
                    except ImportError:
                        raise ImportError(
                            "The 'mace' package is required for using pre-trained MACE descriptors.\n"
                            "Please install it by running:\n\n"
                            "    pip install mace-torch\n"
                        )

                    # Check and set device
                    if torch.cuda.is_available():
                        print(
                            "[Note] There is available CUDA device, and it will be used for MACE descriptor computation.\n")
                        mace_device = "cuda:0"
                    else:
                        mace_device = "cpu"
                        print("[Warning] CUDA device not available. MACE descriptor computation may be slow on CPU.\n")

                    self.mace = mace_off(model=self.mace_param.get('model'),
                                         device=mace_device)

        elif self.descriptor == 'chebyshev':
            try:
                from aenet.torch_featurize import ChebyshevDescriptor, BatchedFeaturizer
                print("You are using Chebyshev descriptor:")
                print("N. Artrith, A. Urban, and G. Ceder, Phys. Rev. B 96 (2017) 014112. \n")
                print("Chebshev parameter:")
                print(self.cheb_param)
                print("\n")
            except ImportError:
                raise ImportError(
                    "The 'aenet-python' package is required for using Chebyshev descriptors.\n"
                    "Please install it by running:\n\n"
                    "    git clone https://github.com/atomisticnet/aenet-python.git\n"
                    "    cd ./aenet-python/\n"
                    "    pip install . --user\n"
                )

            # # Check and set device
            # if torch.cuda.is_available():
            #     print("[Note] There is available CUDA device, and it will be used for Chebyshev descriptor computation.\n")
            #     cheb_device = "cuda:0"
            # else:
            #     cheb_device = "cpu"
            #     # print("[Warning] CUDA device not available. Chebyshev descriptor computation may be slow on CPU.")

            self.chebyshev = ChebyshevDescriptor(species=set(self.species),
                                                 rad_order=self.cheb_param.get("rad_order"),  # Radial polynomial order
                                                 rad_cutoff=self.cheb_param.get("rad_cutoff"),  # Radial cutoff (Ang)
                                                 ang_order=self.cheb_param.get("ang_order"),  # Angular polynomial order
                                                 ang_cutoff=self.cheb_param.get("ang_cutoff"),  # Angular cutoff (Ang)
                                                 device=self.device)
            self.chebyshev_batch = BatchedFeaturizer(self.chebyshev)

        self.atoms_xyz_mask = atoms_mask.to(self.device)
        self.Nmask_xyz = self.atoms_xyz_mask.shape[0]  # 3 * Natoms or 3 * Nreduced_atoms
        self.atoms_mask = (self.atoms_xyz_mask[self.atoms_xyz_mask % 3 == 0] // 3).to(
            self.device)  # Natoms or Nreduced_atoms

        if self.descriptor_standardization and ((self.descriptor == 'soap' and self.soap_param.get("centers") is None) or (self.descriptor == 'mace') or (
                self.descriptor == 'chebyshev')):
            self.standardizer = DescriptorStandardizer()
        else:
            self.standardizer = None

        # train_fp: (Ndata, Ncenter, Ndescriptor)
        # train_dfp_dr: (Ndata, Ncenter, Nreduced_atoms, 3, Ndescriptor)
        self.train_fp, self.train_dfp_dr = self.generate_descriptor(self.images)
        if self.descriptor_standardization and self.standardizer is not None:
            batch_atomic_number = torch.tensor(np.array([image.get_atomic_numbers() for image in self.images]))
            batch_atomic_number = batch_atomic_number[:, self.atoms_mask]  # (Ndata, Nreduced_atoms)
            self.train_fp = self.standardizer.standardize_per_species(self.train_fp, batch_atomic_number,
                                                                      dtype=self.torch_data_type)
            self.train_dfp_dr = self.standardizer.apply_grad_standardization(self.train_dfp_dr, batch_atomic_number)

        self.Y = function  # Y = [Ntrain]
        self.dY = derivative  # dY = [Ntrain, Natom, 3]

        self.dY = self.dY.contiguous().view(self.Ntrain, self.Natom * 3)
        self.dY = self.dY[:, self.atoms_xyz_mask]  # shape: (Ntrain, Nselected)

        # Reshape back to (Ntrain, Nreduced_atoms, 3)
        assert self.dY.shape[1] % 3 == 0, "Selected size must be divisible by 3"
        Nreduced_atoms = self.atoms_mask.shape[0]
        self.dY = self.dY.view(self.Ntrain, Nreduced_atoms, 3)

        self.model_vector = torch.empty((self.Ntrain * (1 + self.Nmask_xyz),), dtype=self.torch_data_type,
                                        device=self.device)

        if prior is None:
            self.prior = ConstantPrior(0.0, dtype=self.torch_data_type, device=self.device,
                                       atoms_xyz_mask=self.atoms_xyz_mask)
        else:
            self.prior = torch.tensor(prior, dtype=self.torch_data_type, device=self.device)
        self.prior_update = prior_update

        self.sparse = sparse
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        if sparse is not None:
            self.sX = sparse  # sX = [Nsparse, Nscenter, Nfeature]
            self.sparse = True

            if sparse_derivative is not None:
                self.sdX = sparse_derivative  # sdX = [Nsparse, Nscenter, Natom, 3, Nfeature]
            else:
                self.sdX = None

        else:
            self.sX = None
            self.sparse = False

        self.kernel = FPKernel(species=self.species,
                               pbc=self.pbc,
                               Natom=self.Natom,
                               Nmask=self.Nmask_xyz,
                               kerneltype=self.kerneltype,
                               data_type=self.data_type,
                               device=self.device)

        hyper_params = dict(kerneltype=self.kerneltype,
                            scale=self.scale,
                            weight=self.weight,
                            noise=self.noise,
                            noisefactor=self.noisefactor,
                            prior=self.prior.constant)

        self.hyper_params = hyper_params
        self.kernel.set_params(self.hyper_params)

        if self.Y is not None:
            if self.dY is not None:
                # [Ntrain] -> [Ntrain, 1]
                # Y_reshaped = self.Y.flatten().unsqueeze(1)
                Y_reshaped = self.Y.contiguous().view(-1, 1)

                # [Ntrain, Natom, 3] -> [Ntrain * 3 * Natom, 1]
                # dY_reshaped = self.dY.flatten().unsqueeze(1)
                dY_reshaped = self.dY.contiguous().view(-1, 1)

                # [Ntrain * (1 + 3 * Natom), 1]
                # [[e1, e2, ..., eN, f11x, f11y, f11z, f12x, f12y, ..., fNzNz]],
                self.YdY = torch.cat((Y_reshaped, dY_reshaped), dim=0)

                del Y_reshaped, dY_reshaped

            else:
                self.YdY = self.Y.flatten().unsqueeze(1)  # no dY [e1, e2, ..., eN]
        else:
            self.YdY = None

    def generate_descriptor(self, images):

        if self.descriptor == 'soap':
            dfp_dr, fp = self.soap.derivatives(images,
                                               centers=[self.soap_param.get('centers')] * len(images),
                                               method=self.soap_param.get('method'),
                                               return_descriptor=True,
                                               n_jobs=self.soap_param.get('n_jobs'))

            fp = torch.as_tensor(fp, dtype=self.torch_data_type, device=self.device)  # (Ndata, Ncenters, Ndescriptor)
            dfp_dr = torch.as_tensor(dfp_dr, dtype=self.torch_data_type,
                                     device=self.device)  # (Ndata, Ncenters, Natom, 3, Ndescriptor)
            dfp_dr = dfp_dr[:, :, self.atoms_mask, :, :]  # (Ndata, Ncenters, Nreduced_atoms, 3, Ndescriptor)

        elif self.descriptor == 'mace':

            fp = []
            dfp_dr = []
            for image in images:
                fp__, dfp_dr__ = mace_descriptor_gradient(image,
                                                          self.mace,
                                                          atoms_mask=self.atoms_mask,
                                                          delta=self.mace_param.get("delta"),
                                                          invariants=self.mace_param.get("invariants"),
                                                          num_layers=self.mace_param.get("num_layers"),
                                                          n_jobs=self.mace_param.get("mace_n_jobs"),
                                                          dtype=self.data_type)
                fp.append(torch.tensor(fp__, dtype=self.torch_data_type, device=self.device))
                dfp_dr.append(torch.tensor(dfp_dr__, dtype=self.torch_data_type, device=self.device))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            fp = torch.stack(fp).to(dtype=self.torch_data_type,
                                    device=self.device)  # (Ndata, Nreduced_atoms, Ndescriptor)
            dfp_dr = torch.stack(dfp_dr).to(dtype=self.torch_data_type,
                                            device=self.device)  # (Ndata, Nreduced_atoms, Nreduced_atoms, 3, Ndescriptor)

        elif self.descriptor == 'chebyshev':

            fp = []
            dfp_dr = []
            for image in images:
                if np.any(image.pbc):
                    fp__, dfp_dr__ = chebyshev_descriptor_gradient_periodic(image,
                                                                            self.chebyshev_batch,
                                                                            atoms_mask=self.atoms_mask,
                                                                            delta=self.cheb_param.get("delta"),
                                                                            dtype=self.torch_data_type)
                    fp.append(fp__)
                    dfp_dr.append(dfp_dr__)

                else:
                    fp__, dfp_dr__ = chebyshev_descriptor_gradient(image,
                                                                   self.chebyshev_batch,
                                                                   atoms_mask=self.atoms_mask,
                                                                   delta=self.cheb_param.get("delta"),
                                                                   dtype=self.torch_data_type)
                    fp.append(fp__)
                    dfp_dr.append(dfp_dr__)

            fp = torch.stack(fp).to(dtype=self.torch_data_type,
                                    device=self.device)  # (Ndata, Nreduced_atoms, Ndescriptor)
            dfp_dr = torch.stack(dfp_dr).to(dtype=self.torch_data_type,
                                            device=self.device)  # (Ndata, Nreduced_atoms, Nreduced_atoms, 3, Ndescriptor)

        else:
            fp = []
            dfp_dr = []
            for image in images:
                fp.append(torch.as_tensor(image.get_positions(wrap=False).reshape(-1), dtype=self.torch_data_type).to(
                    self.device))

                dfp_dr.append(torch.as_tensor(np.eye(self.Natom * 3).reshape(self.Natom, -1, 3, order='F'),
                                              dtype=self.torch_data_type).to(self.device))

            fp = torch.stack(fp).to(self.device)  # (Ndata, Natom*3)
            dfp_dr = torch.stack(dfp_dr).to(self.device)  # (Ndata, Natom, Natom*3, 3)

            fp = fp.unsqueeze(1)  # (Ndata, 1, Natom*3)
            dfp_dr = dfp_dr.transpose(2, 3).unsqueeze(1)  # (Ndata, 1, Natom, 3, Natom*3)
            dfp_dr = dfp_dr[:, :, self.atoms_mask, :, :]  # (Ndata, 1, Nreduced_atoms, 3, Natom*3)

        return fp, dfp_dr

    def generate_descriptor_per_data(self, image):

        if self.descriptor == 'soap':
            dfp_dr, fp = self.soap.derivatives(image,
                                               centers=self.soap_param.get('centers'),
                                               method=self.soap_param.get('method'),
                                               return_descriptor=True,
                                               n_jobs=self.soap_param.get('n_jobs'))

            fp = torch.as_tensor(fp, dtype=self.torch_data_type, device=self.device)  # (Ncenters, Natom*3)
            dfp_dr = torch.as_tensor(dfp_dr, dtype=self.torch_data_type,
                                     device=self.device)  # (Ncenters, Natom, 3, Natom*3)
            dfp_dr = dfp_dr[:, self.atoms_mask, :, :]  # (Ncenters, Nreduced_atoms, 3, Ndescriptor)

        elif self.descriptor == 'mace':
            fp, dfp_dr = mace_descriptor_gradient(image,
                                                  self.mace,
                                                  atoms_mask=self.atoms_mask,
                                                  delta=self.mace_param.get("delta"),
                                                  invariants=self.mace_param.get("invariants"),
                                                  num_layers=self.mace_param.get("num_layers"),
                                                  n_jobs=self.mace_param.get("mace_n_jobs"),
                                                  dtype=self.data_type)

            fp = torch.tensor(fp, dtype=self.torch_data_type, device=self.device)  # (Nreduced_atoms, Ndescriptor)
            dfp_dr = torch.tensor(dfp_dr, dtype=self.torch_data_type,
                                  device=self.device)  # (Nreduced_atoms, Nreduced_atoms, 3, Ndescriptor)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elif self.descriptor == 'chebyshev':
            if np.any(image.pbc):
                fp, dfp_dr = chebyshev_descriptor_gradient_periodic(image,
                                                                    self.chebyshev_batch,
                                                                    atoms_mask=self.atoms_mask,
                                                                    delta=self.cheb_param.get("delta"),
                                                                    dtype=self.torch_data_type)

            else:
                fp, dfp_dr = chebyshev_descriptor_gradient(image,
                                                           self.chebyshev_batch,
                                                           atoms_mask=self.atoms_mask,
                                                           delta=self.cheb_param.get("delta"),
                                                           dtype=self.torch_data_type)

        else:
            fp = torch.as_tensor(image.get_positions(wrap=False).reshape(-1), dtype=self.torch_data_type).to(
                self.device)
            dfp_dr = torch.as_tensor(np.eye(self.Natom * 3).reshape(self.Natom, -1, 3, order='F'),
                                     dtype=self.torch_data_type).to(self.device)

            fp = fp.unsqueeze(0)
            dfp_dr = dfp_dr.transpose(1, 2).unsqueeze(0)
            dfp_dr = dfp_dr[:, self.atoms_mask, :, :]  # (1, Nreduced_atoms, 3, Natom*3)

        return fp, dfp_dr

    def train_model(self):

        # covariance matrix between the training points X
        self.K_XX_L = self.kernel.kernel_matrix_batch(fp=self.train_fp,
                                                      dfp_dr=self.train_dfp_dr,
                                                      batch_size=self.train_batch_size)

        a = torch.full((self.Ntrain, 1), self.hyper_params['noise'] * self.hyper_params['noisefactor'],
                       dtype=self.torch_data_type, device=self.device)
        noise_val = self.hyper_params['noise']
        b = noise_val.expand(self.Ntrain, self.Nmask_xyz)

        # reg = torch.diag(torch.cat((a, b), 1).flatten() ** 2)
        diagonal_values = torch.cat((a, b), 1).flatten() ** 2

        self.K_XX_L.diagonal().add_(diagonal_values)

        try:
            self.K_XX_L = torch.linalg.cholesky(self.K_XX_L, upper=False)

        except torch.linalg.LinAlgError:
            # Diagonal sum (trace)
            diag_sum = torch.sum(torch.diag(self.K_XX_L))

            # epsilon value
            eps = torch.finfo(self.torch_data_type).eps

            # scaling factor
            scaling_factor = 1 / (1 / (4.0 * eps) - 1)

            # adjust K_XX
            adjustment = diag_sum * scaling_factor * torch.ones(self.K_XX_L.shape[0],
                                                                dtype=self.torch_data_type,
                                                                device=self.device)
            self.K_XX_L.diagonal().add_(adjustment)

            # Step 1: Cholesky decomposition for K_XX after adjusting
            self.K_XX_L = torch.linalg.cholesky(self.K_XX_L, upper=False)

        if self.prior_update:
            self.prior.update(len(self.images), len(self.images[0]), self.YdY, self.K_XX_L)
            self.hyper_params.update(dict(prior=self.prior.constant))
            self.kernel.set_params(self.hyper_params)

        _prior_array = self.prior.potential_batch(len(self.images), len(self.images[0]))
        self.model_vector = torch.cholesky_solve(self.YdY.contiguous().view(-1, 1) - _prior_array.view(-1, 1),
                                                 self.K_XX_L, upper=False)

        return

    def eval_batch(self, eval_images, get_variance=False):

        Ntest = len(eval_images)
        eval_x_N_batch = get_N_batch(Ntest, self.eval_batch_size)
        eval_x_indexes = get_batch_indexes_N_batch(Ntest, eval_x_N_batch)

        E_hat = torch.empty((Ntest,), dtype=self.torch_data_type, device=self.device)
        F_hat = torch.zeros((Ntest, self.Natom * 3), dtype=self.torch_data_type, device=self.device)

        if not get_variance:
            for i in range(0, eval_x_N_batch):
                data_per_batch = eval_x_indexes[i][1] - eval_x_indexes[i][0]
                batch_eval_images = eval_images[eval_x_indexes[i][0]:eval_x_indexes[i][1]]
                eval_fp, eval_dfp_dr = self.generate_descriptor(batch_eval_images)
                if self.descriptor_standardization and self.standardizer is not None:
                    batch_atomic_number = torch.tensor(
                        np.array([image.get_atomic_numbers() for image in batch_eval_images]))
                    batch_atomic_number = batch_atomic_number[:, self.atoms_mask]  # (Ndata, Nreduced_atoms)
                    eval_fp = self.standardizer.apply_desc_standardization(eval_fp, batch_atomic_number)
                    eval_dfp_dr = self.standardizer.apply_grad_standardization(eval_dfp_dr,
                                                                               batch_atomic_number)

                pred, kernel = self.eval_data_batch(eval_fp, eval_dfp_dr)
                E_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1]] = pred[0:data_per_batch]
                F_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1], self.atoms_xyz_mask] = pred[data_per_batch:].view(
                    data_per_batch, -1)
                # F_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1], :] = apply_force_mask(
                #     F=pred[data_per_batch:].view(data_per_batch, -1),
                #     atoms_xyz_mask=self.atoms_xyz_mask)

            return E_hat, F_hat.view((Ntest, self.Natom, 3)), None, None

        else:
            unc_e = torch.empty((Ntest,), dtype=self.torch_data_type, device=self.device)
            unc_f = torch.zeros((Ntest, self.Natom * 3), dtype=self.torch_data_type, device=self.device)

            for i in range(0, eval_x_N_batch):
                data_per_batch = eval_x_indexes[i][1] - eval_x_indexes[i][0]
                batch_eval_images = eval_images[eval_x_indexes[i][0]:eval_x_indexes[i][1]]
                eval_fp, eval_dfp_dr = self.generate_descriptor(batch_eval_images)
                if self.descriptor_standardization and self.standardizer is not None:
                    batch_atomic_number = torch.tensor(
                        np.array([image.get_atomic_numbers() for image in batch_eval_images]))
                    batch_atomic_number = batch_atomic_number[:, self.atoms_mask]  # (Ndata, Nreduced_atoms)
                    eval_fp = self.standardizer.apply_desc_standardization(eval_fp, batch_atomic_number)
                    eval_dfp_dr = self.standardizer.apply_grad_standardization(eval_dfp_dr,
                                                                               batch_atomic_number)

                pred, kernel = self.eval_data_batch(eval_fp, eval_dfp_dr)
                E_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1]] = pred[0:data_per_batch]
                F_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1], self.atoms_xyz_mask] = pred[data_per_batch:].view(
                    data_per_batch, -1)
                # F_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1], :] = apply_force_mask(
                #     F=pred[data_per_batch:].view(data_per_batch, -1),
                #     atoms_xyz_mask=self.atoms_xyz_mask)

                var = self.eval_variance_batch(get_variance=get_variance,
                                               eval_fp=eval_fp,
                                               eval_dfp_dr=eval_dfp_dr,
                                               k=kernel)
                std = torch.sqrt(torch.diagonal(var))

                unc_e[eval_x_indexes[i][0]:eval_x_indexes[i][1]] = std[0:data_per_batch] / self.weight
                unc_f[eval_x_indexes[i][0]:eval_x_indexes[i][1], self.atoms_xyz_mask] = std[data_per_batch:].view(
                    data_per_batch, -1)
                # unc_f[eval_x_indexes[i][0]:eval_x_indexes[i][1], :] = apply_force_mask(
                #     F=std[data_per_batch:].view(data_per_batch, -1),
                #     atoms_xyz_mask=self.atoms_xyz_mask)

            return E_hat, F_hat.view((Ntest, self.Natom, 3)), unc_e, unc_f.view((Ntest, self.Natom, 3))

    def eval_data_batch(self, eval_fp, eval_dfp_dr):
        # kernel between test point x and training points X
        kernel = self.kernel.kernel_vector_batch(fp_1=eval_fp,
                                                 dfp_dr_1=eval_dfp_dr,
                                                 fp_2=self.train_fp,
                                                 dfp_dr_2=self.train_dfp_dr,
                                                 batch_size=self.eval_batch_size)

        pred = torch.matmul(kernel, self.model_vector.view(-1)) + self.prior.potential_batch(eval_dfp_dr.shape[0],
                                                                                             eval_dfp_dr.shape[2])

        return pred, kernel

    def eval_variance_batch(self, get_variance, eval_fp, eval_dfp_dr, k):
        """
        variance.shape  # [Ntest * (1 + 3 * Natom), Ntest * (1 + 3 * Natom)]
        self.K_XX_L.shape  # [Ntrain * (1 + 3 * Natom), Ntrain * (1 + 3 * Natom)]
        k.T.clone().shape  # [Ntrain * (1 + 3 * Natom), Ntest * (1 + 3 * Natom)]
        self.Ck.shape  # [Ntrain * (1 + 3 * Natom), Ntest * (1 + 3 * Natom)]
        covariance.shape  # [Ntest * (1 + 3 * Natom), Ntest * (1 + 3 * Natom)]
        """

        if get_variance:
            # Kx=k -> x = K^(-1)k
            covariance = torch.matmul(k, torch.cholesky_solve(k.T.clone(), self.K_XX_L, upper=False))

            # Adjust variance by subtracting covariance
            return self.kernel.kernel_matrix_batch(fp=eval_fp,
                                                   dfp_dr=eval_dfp_dr,
                                                   batch_size=self.eval_batch_size) - covariance

        else:
            return None

    def eval_per_data(self, eval_image, get_variance=False):
        eval_fp_i, eval_dfp_dr_i = self.generate_descriptor_per_data(eval_image)
        if self.descriptor_standardization and self.standardizer is not None:
            atomic_number = torch.tensor(eval_image.get_atomic_numbers()).unsqueeze(dim=0)
            atomic_number = atomic_number[:, self.atoms_mask]  # (Ndata, Nreduced_atoms)
            eval_fp_i = self.standardizer.apply_desc_standardization(eval_fp_i.unsqueeze(dim=0), atomic_number)
            eval_fp_i = eval_fp_i.squeeze(dim=0)
            eval_dfp_dr_i = self.standardizer.apply_grad_standardization(eval_dfp_dr_i.unsqueeze(dim=0), atomic_number)
            eval_dfp_dr_i = eval_dfp_dr_i.squeeze(dim=0)

        pred, kernel = self.eval_data_per_data(eval_fp_i, eval_dfp_dr_i)
        E_hat = pred[0]
        F_hat = torch.zeros((self.Natom * 3,), dtype=self.torch_data_type, device=self.device)
        F_hat[self.atoms_xyz_mask] = pred[1:]
        # F_hat = apply_force_mask(F=pred[1:].view(1, -1), atoms_xyz_mask=self.atoms_xyz_mask)

        if not get_variance:
            return E_hat, F_hat.view((self.Natom, 3)), None

        else:
            unc_f = torch.zeros((self.Natom * 3,), dtype=self.torch_data_type, device=self.device)

            var = self.eval_variance_per_data(get_variance=True,
                                              eval_fp_i=eval_fp_i,
                                              eval_dfp_dr_i=eval_dfp_dr_i,
                                              k=kernel)

            std = torch.sqrt(torch.diagonal(var))
            unc_e = std[0] / self.weight
            unc_f[self.atoms_xyz_mask] = std[1:]
            # unc_f = apply_force_mask(F=std[1:].view(1, -1), atoms_xyz_mask=self.atoms_xyz_mask)

            return E_hat, F_hat.view((self.Natom, 3)), unc_e, unc_f.view((self.Natom, 3))

    def eval_data_per_data(self, eval_fp_i, eval_dfp_dr_i):
        # kernel between test point x and training points X
        kernel = self.kernel.kernel_vector_per_data(fp_1_i=eval_fp_i,
                                                    dfp_dr_1_i=eval_dfp_dr_i,
                                                    fp_2=self.train_fp,
                                                    dfp_dr_2=self.train_dfp_dr,
                                                    batch_size=self.train_batch_size)

        pred = torch.matmul(kernel, self.model_vector.view(-1)) + self.prior.potential_per_data(eval_dfp_dr_i.shape[1])

        return pred, kernel

    def eval_variance_per_data(self, get_variance, eval_fp_i, eval_dfp_dr_i, k):

        if get_variance:
            covariance = torch.matmul(k, torch.cholesky_solve(k.T.clone(), self.K_XX_L, upper=False))

            # Adjust variance by subtracting covariance
            return self.kernel.kernel_matrix_per_data(fp_i=eval_fp_i,
                                                      dfp_dr_i=eval_dfp_dr_i) - covariance

        else:
            return None

    def save_data(self, file="calc_dict.pt"):
        """
        self.data_type
        self.torch_data_type

        self.device = device
        self.noise
        self.noisefactor
        self.scale
        self.weight
        self.use_forces
        self.sparse

        (self.train_batch_size)
        (self.eval_batch_size)

        self.images
        self.Y
        self.dY
        self.YdY

        self.Ntrain
        self.Natom

        self.K_XX_L
        self.model_vector
        """

        state = {
            'kerneltype': self.kerneltype,
            'noise': self.noise,
            'noisefactor': self.noisefactor,
            'scale': self.scale,
            'weight': self.weight,
            'use_forces': self.use_forces,
            'sparse': self.sparse,
            'train_batch_size': self.train_batch_size,
            'eval_batch_size': self.eval_batch_size,
            'Y': self.Y,
            'dY': self.dY,
            'YdY': self.YdY,
            'K_XX_L': self.K_XX_L,
            'model_vector': self.model_vector,
        }
        torch.save(state, file)

    def load_data(self, file="calc_dict.pt"):
        state = torch.load(file)

        self.kerneltype = state.get('kerneltype')
        self.noise = state.get('noise')
        self.noisefactor = state.get('noisefactor')
        self.scale = state.get('scale')
        self.weight = state.get('weight')

        self.use_forces = state.get('use_forces')
        self.sparse = state.get('sparse')

        self.train_batch_size = state.get('train_batch_size')
        self.eval_batch_size = state.get('eval_batch_size')

        self.Y = state.get('Y')
        self.dY = state.get('dY')
        self.YdY = state.get('YdY')

        self.K_XX_L = state.get('K_XX_L')
        self.model_vector = state.get('model_vector')

        hyper_params = dict(kerneltype=self.kerneltype,
                            scale=self.scale,
                            weight=self.weight,
                            noise=self.noise,
                            noisefactor=self.noisefactor)

        self.hyper_params = hyper_params
        self.kernel.set_params(self.hyper_params)
