import torch
import numpy as np

from copy import deepcopy
from joblib import Parallel, delayed

from aenet_gpr.src.prior import ConstantPrior
from aenet_gpr.src.pytorch_kernel import FPKernel
from aenet_gpr.util.prepare_data import get_N_batch, get_batch_indexes_N_batch


def _central_diff_task(atoms, i, j, delta, model, num_layers):
    """
    Desciptor gradient for each atom i, and cartexian coordinate j(x/y/z)
    """
    atoms_f = deepcopy(atoms)
    atoms_f.positions[i, j] += delta
    desc_f = model.get_descriptors(atoms_f, num_layers=num_layers)

    atoms_b = deepcopy(atoms)
    atoms_b.positions[i, j] -= delta
    desc_b = model.get_descriptors(atoms_b, num_layers=num_layers)

    grad_ij = (desc_f - desc_b) / (2 * delta)
    return i, j, grad_ij


def numerical_descriptor_gradient_parallel(atoms, model, delta=1e-4, num_layers=-1, n_jobs=1, dtype=torch.float32):
    """
    Args:
        atoms (ase.Atoms):
        model: MACE calculator
        delta (float): displacement (Ang)
        num_layers (int): MACE interaction layers number
        n_jobs (int): pararell job number
        dtype (torch.dtype):

    Returns:
        desc (torch.Tensor): (n_atoms, descriptor_dim)
        grad (torch.Tensor): (n_atoms, n_atoms, 3, descriptor_dim)
    """

    desc = model.get_descriptors(atoms, num_layers=num_layers)
    n_atoms, D = desc.shape

    # Job list
    jobs = [(atoms, i, j, delta, model, num_layers) for i in range(n_atoms) for j in range(3)]

    # Pararell execution
    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_central_diff_task)(*job) for job in jobs)
    results.sort(key=lambda x: (x[0], x[1]))

    grad = torch.empty((n_atoms, n_atoms, 3, D), dtype=dtype)
    for i, j, grad_ij in results:
        grad[:, i, j, :] = torch.tensor(grad_ij, dtype=dtype)

    return torch.tensor(desc, dtype=dtype), grad


def apply_force_mask(F, atoms_mask):
    """
    Args:
        F: (Ntest, 3*Natoms) force tensor
        atoms_mask: tensor([...]) flattened xyz indices to keep

    Returns:
        F_masked: same shape as F, masked with zeros outside atoms_mask
    """
    if atoms_mask is None:
        # keep all forces
        return F.clone()
    else:
        mask = torch.zeros(F.shape[1], dtype=torch.bool, device=F.device)
        mask[atoms_mask] = True

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
                 soap_param=None, mace_param=None, descriptor='cartesian coordinates',
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
        self.descriptor = descriptor
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
        self.atoms_mask = atoms_mask

        if self.descriptor == 'soap':
            try:
                from dscribe.descriptors import SOAP
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
            if self.mace_param.get('system') == "materials":
                try:
                    from mace.calculators import mace_mp
                except ImportError:
                    raise ImportError(
                        "The 'joblib' and 'mace' packages are required for using pre-trained MACE descriptors.\n"
                        "Please install it by running:\n\n"
                        "    pip install joblib\n"
                        "    pip install mace-torch\n"
                    )

                self.mace = mace_mp(model=self.mace_param.get('model'),
                                    device=self.device)

            else:
                try:
                    from mace.calculators import mace_off
                except ImportError:
                    raise ImportError(
                        "The 'joblib' and 'mace' packages are required for using pre-trained MACE descriptors.\n"
                        "Please install it by running:\n\n"
                        "    pip install joblib\n"
                        "    pip install mace-torch\n"
                    )

                self.mace = mace_off(model=self.mace_param.get('model'),
                                     device=self.device)

        self.train_fp, self.train_dfp_dr = self.generate_descriptor(self.images)
        self.Y = function  # Y = [Ntrain]
        self.dY = derivative  # dY = [Ntrain, Natom, 3]
        self.model_vector = torch.empty((self.Ntrain * (1 + 3 * self.Natom),), dtype=self.torch_data_type,
                                        device=self.device)

        if prior is None:
            self.prior = ConstantPrior(0.0, dtype=self.torch_data_type, device=self.device, atoms_mask=self.atoms_mask)
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

            dfp_dr = torch.as_tensor(dfp_dr, dtype=self.torch_data_type).to(self.device)  # (Ndata, Ncenters, Natom, 3, Natom*3)
            fp = torch.as_tensor(fp, dtype=self.torch_data_type).to(self.device)  # (Ndata, Ncenters, Natom*3)

        elif self.descriptor == 'mace':
            fp = []
            dfp_dr = []
            for image in images:
                # fp__, dfp_dr__ = self.mace.get_descriptors_with_jacobian(image)
                # fp.append(fp__)
                # dfp_dr.append(dfp_dr__)

                # if self.device == 'cpu':
                #     fp__, dfp_dr__ = numerical_descriptor_gradient_parallel(image, self.mace, n_jobs=self.n_jobs, dtype=self.torch_data_type)
                # else:
                fp__, dfp_dr__ = numerical_descriptor_gradient_parallel(image,
                                                                        self.mace,
                                                                        delta=self.mace_param.get("delta"),
                                                                        num_layers=self.mace_param.get("num_layers"),
                                                                        n_jobs=self.mace_param.get("n_jobs"),
                                                                        dtype=self.torch_data_type)
                fp.append(fp__)
                dfp_dr.append(dfp_dr__)

            fp = torch.stack(fp).to(dtype=self.torch_data_type, device=self.device)  # (Ndata, Natom, Ndescriptor)
            dfp_dr = torch.stack(dfp_dr).to(dtype=self.torch_data_type, device=self.device)  # (Ndata, Natom, Natom, 3, Ndescriptor)

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

        return fp, dfp_dr

    def generate_descriptor_per_data(self, image):

        if self.descriptor == 'soap':
            dfp_dr, fp = self.soap.derivatives(image,
                                               centers=self.soap_param.get('centers'),
                                               method=self.soap_param.get('method'),
                                               return_descriptor=True,
                                               n_jobs=self.soap_param.get('n_jobs'))

            dfp_dr = torch.as_tensor(dfp_dr, dtype=self.torch_data_type).to(
                self.device)  # (Ncenters, Natom, 3, Natom*3)
            fp = torch.as_tensor(fp, dtype=self.torch_data_type).to(self.device)  # (Ncenters, Natom*3)

        elif self.descriptor == 'mace':
            # if self.device == 'cpu':
            #     fp, dfp_dr = numerical_descriptor_gradient_parallel(image, self.mace, n_jobs=self.n_jobs, dtype=self.torch_data_type)
            # else:
            fp, dfp_dr = numerical_descriptor_gradient_parallel(image,
                                                                self.mace,
                                                                delta=self.mace_param.get("delta"),
                                                                num_layers=self.mace_param.get("num_layers"),
                                                                n_jobs=self.mace_param.get("n_jobs"),
                                                                dtype=self.torch_data_type)
            fp = fp.to(dtype=self.torch_data_type, device=self.device)  # (Natom, Ndescriptor)
            dfp_dr = dfp_dr.to(dtype=self.torch_data_type, device=self.device)  # (Natom, Natom, 3, Ndescriptor)

        else:
            fp = torch.as_tensor(image.get_positions(wrap=False).reshape(-1), dtype=self.torch_data_type).to(
                self.device)
            dfp_dr = torch.as_tensor(np.eye(self.Natom * 3).reshape(self.Natom, -1, 3, order='F'),
                                     dtype=self.torch_data_type).to(self.device)

            fp = fp.unsqueeze(0)
            dfp_dr = dfp_dr.transpose(1, 2).unsqueeze(0)

        return fp, dfp_dr

    def train_model(self):

        # covariance matrix between the training points X
        self.K_XX_L = self.kernel.kernel_matrix_batch(fp=self.train_fp,
                                                      dfp_dr=self.train_dfp_dr,
                                                      batch_size=self.train_batch_size)

        a = torch.full((self.Ntrain, 1), self.hyper_params['noise'] * self.hyper_params['noisefactor'],
                       dtype=self.torch_data_type, device=self.device)
        noise_val = self.hyper_params['noise']
        b = noise_val.expand(self.Ntrain, 3 * self.Natom)

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
                eval_fp, eval_dfp_dr = self.generate_descriptor(eval_images[eval_x_indexes[i][0]:eval_x_indexes[i][1]])

                pred, kernel = self.eval_data_batch(eval_fp, eval_dfp_dr)
                E_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1]] = pred[0:data_per_batch]
                F_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1], :] = apply_force_mask(F=pred[data_per_batch:].view(data_per_batch, -1),
                                                                                       atoms_mask=self.atoms_mask)

            return E_hat, F_hat.view((Ntest, self.Natom, 3)), None, None

        else:
            unc_e = torch.empty((Ntest,), dtype=self.torch_data_type, device=self.device)
            unc_f = torch.zeros((Ntest, self.Natom * 3), dtype=self.torch_data_type, device=self.device)

            for i in range(0, eval_x_N_batch):
                data_per_batch = eval_x_indexes[i][1] - eval_x_indexes[i][0]
                eval_fp, eval_dfp_dr = self.generate_descriptor(eval_images[eval_x_indexes[i][0]:eval_x_indexes[i][1]])

                pred, kernel = self.eval_data_batch(eval_fp, eval_dfp_dr)
                E_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1]] = pred[0:data_per_batch]
                F_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1], :] = apply_force_mask(F=pred[data_per_batch:].view(data_per_batch, -1),
                                                                                       atoms_mask=self.atoms_mask)

                var = self.eval_variance_batch(get_variance=get_variance,
                                               eval_fp=eval_fp,
                                               eval_dfp_dr=eval_dfp_dr,
                                               k=kernel)
                std = torch.sqrt(torch.diagonal(var))

                unc_e[eval_x_indexes[i][0]:eval_x_indexes[i][1]] = std[0:data_per_batch] / self.weight
                unc_f[eval_x_indexes[i][0]:eval_x_indexes[i][1], :] = apply_force_mask(F=std[data_per_batch:].view(data_per_batch, -1),
                                                                                       atoms_mask=self.atoms_mask)

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

        pred, kernel = self.eval_data_per_data(eval_fp_i, eval_dfp_dr_i)
        E_hat = pred[0]
        F_hat = apply_force_mask(F=pred[1:].view(1, -1), atoms_mask=self.atoms_mask)

        if not get_variance:
            return E_hat, F_hat.view((self.Natom, 3)), None

        else:
            var = self.eval_variance_per_data(get_variance=True,
                                              eval_fp_i=eval_fp_i,
                                              eval_dfp_dr_i=eval_dfp_dr_i,
                                              k=kernel)

            std = torch.sqrt(torch.diagonal(var))
            unc_e = std[0] / self.weight
            unc_f = apply_force_mask(F=std[1:].view(1, -1), atoms_mask=self.atoms_mask)

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
