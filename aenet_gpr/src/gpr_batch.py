import torch
import torch.nn as nn
import numpy as np

from aenet_gpr.src.prior import ConstantPrior
from aenet_gpr.src.pytorch_kernel import FPKernel, FPKernelNoforces
from aenet_gpr.util.prepare_data import get_N_batch, get_batch_indexes_N_batch


class GaussianProcess(nn.Module):
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
        super().__init__()

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

        if autograd:
            self.scale = nn.Parameter(torch.tensor(scale, dtype=self.torch_data_type), requires_grad=True).to(
                self.device)
            self.weight = nn.Parameter(torch.tensor(weight, dtype=self.torch_data_type), requires_grad=True).to(
                self.device)
        else:
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

        if self.atoms_mask is not None:
            self.Nmask = self.atoms_mask.shape[0]
        else:
            self.Nmask = 3 * self.Natom

        self.Y = function  # Y = [Ntrain]
        self.dY = derivative  # dY = [Ntrain, Natom, 3]
        self.model_vector = torch.empty((self.Ntrain * (1 + self.Nmask),), dtype=self.torch_data_type, device=self.device)

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

        if self.use_forces:
            self.kernel = FPKernel(species=self.species,
                                   pbc=self.pbc,
                                   Natom=self.Natom,
                                   kerneltype=self.kerneltype,
                                   data_type=self.data_type,
                                   soap_param=self.soap_param,
                                   mace_param=self.mace_param,
                                   descriptor=self.descriptor,
                                   device=self.device,
                                   atoms_mask=self.atoms_mask)
        else:
            self.kernel = FPKernelNoforces(species=self.species,
                                           pbc=self.pbc,
                                           Natom=self.Natom,
                                           kerneltype=self.kerneltype,
                                           data_type=self.data_type,
                                           soap_param=self.soap_param,
                                           mace_param=self.mace_param,
                                           descriptor=self.descriptor,
                                           device=self.device,
                                           atoms_mask=self.atoms_mask)

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
                if self.atoms_mask is not None:
                    dY_reshaped = self.dY.contiguous().view(self.dY.shape[0], -1)[:, self.atoms_mask]
                    dY_reshaped = dY_reshaped.view(-1, 1)
                else:
                    dY_reshaped = self.dY.contiguous().view(-1, 1)

                # [Ntrain * (1 + 3 * Natom), 1]
                # [[e1, e2, ..., eN, f11x, f11y, f11z, f12x, f12y, ..., fNzNz]],
                self.YdY = torch.cat((Y_reshaped, dY_reshaped), dim=0)

                del Y_reshaped, dY_reshaped

            else:
                self.YdY = self.Y.flatten().unsqueeze(1)  # no dY [e1, e2, ..., eN]
        else:
            self.YdY = None

    def train_model(self):
        if self.sparse:
            if self.use_forces:  # self.kernel = FPKernel
                # covariance matrix between the training points X
                K_XX = self.kernel.kernel_matrix_batch(images=self.images, batch_size=self.train_batch_size)

                # reg = [Ntrain * (1 + 3 * Natom), Ntrain * (1 + 3 * Natom)]
                a = torch.tensor(self.Ntrain * [self.hyper_params['noise'] * self.hyper_params['noisefactor']], dtype=self.torch_data_type, device=self.device).reshape(
                    self.Ntrain, 1)
                b = torch.tensor(self.Ntrain * self.Nmask * [self.hyper_params['noise']],
                                 dtype=self.torch_data_type, device=self.device).reshape(self.Ntrain, -1)
                reg = torch.diag(torch.cat((a, b), 1).flatten() ** 2)
                self.inv_reg = torch.linalg.inv(reg)

                K_XX.add_(reg)
                try:
                    self.K_XX_L = torch.linalg.cholesky(K_XX, upper=False)
                except torch.linalg.LinAlgError:
                    with torch.no_grad():
                        # Diagonal sum (trace)
                        diag_sum = torch.sum(torch.diag(K_XX))

                        # epsilon value
                        eps = torch.finfo(self.torch_data_type).eps

                        # scaling factor
                        scaling_factor = 1 / (1 / (4.0 * eps) - 1)

                        # adjust K_XX
                        adjustment = diag_sum * scaling_factor * torch.ones(K_XX.shape[0],
                                                                            dtype=self.torch_data_type)
                        K_XX.diagonal().add_(adjustment)

                        # Step 1: Cholesky decomposition for K_XX after adjusting
                        self.K_XX_L = torch.linalg.cholesky(K_XX, upper=False)

                # covariance matrix between the inducing points S
                K_ss = self.kernel.kernel_matrix_batch(images=self.images, batch_size=self.train_batch_size)

                try:
                    # Step 1: Cholesky decomposition for K_ss
                    self.K_ss_L = torch.linalg.cholesky(K_ss, upper=False)
                except torch.linalg.LinAlgError:
                    with torch.no_grad():
                        # Diagonal sum (trace)
                        diag_sum = torch.sum(torch.diag(K_ss))

                        # epsilon value
                        eps = torch.finfo(self.torch_data_type).eps

                        # scaling factor
                        scaling_factor = 1 / (1 / (4.0 * eps) - 1)

                        # adjust K_XX
                        adjustment = diag_sum * scaling_factor * torch.ones(K_ss.shape[0],
                                                                            dtype=self.torch_data_type)
                        K_ss.diagonal().add_(adjustment)

                        # Step 1: Cholesky decomposition for K_ss after adjusting
                        self.K_ss_L = torch.linalg.cholesky(K_ss, upper=False)

                # covariance between inducing points S and training points X
                self.K_sX = self.kernel.kernel_vector_batch(x=self.sX, dx=self.sdX, X=self.X, dX=self.dX)

                Qs = K_ss + torch.einsum('pi,ij,jq->pq', self.K_sX, self.inv_reg, self.K_sX.T)

                CKK_XX_L = torch.cholesky_solve(self.K_sX.T.clone(), self.K_XX_L, upper=False)
                Q_ss = torch.einsum('ij,jk->ik', self.K_sX, CKK_XX_L)
                self.CQ_ss = torch.cholesky_solve(Q_ss, self.K_ss_L, upper=False)

            else:  # self.kernel = FPKernelNoforces
                # covariance matrix between the training points X
                K_XX = self.kernel.kernel_without_deriv(X1=self.X, X2=self.X)

                a = torch.tensor(self.Ntrain * [self.hyper_params['noise']], dtype=self.torch_data_type, device=self.device)
                reg = torch.diag(a ** 2)
                self.inv_reg = torch.linalg.inv(reg)

                K_XX.add_(reg)
                try:
                    self.K_XX_L = torch.linalg.cholesky(K_XX, upper=False)
                except torch.linalg.LinAlgError:
                    with torch.no_grad():
                        # Diagonal sum (trace)
                        diag_sum = torch.sum(torch.diag(K_XX))

                        # epsilon value
                        eps = torch.finfo(self.torch_data_type).eps

                        # scaling factor
                        scaling_factor = 1 / (1 / (4.0 * eps) - 1)

                        # adjust K_XX
                        adjustment = diag_sum * scaling_factor * torch.ones(K_XX.shape[0],
                                                                            dtype=self.torch_data_type)
                        K_XX.diagonal().add_(adjustment)

                        # Step 1: Cholesky decomposition for K_XX after adjusting
                        self.K_XX_L = torch.linalg.cholesky(K_XX, upper=False)

                # covariance matrix between the inducing points S
                K_ss = self.kernel.kernel_without_deriv(X1=self.sX, X2=self.sX)

                try:
                    # Step 1: Cholesky decomposition for K_ss
                    self.K_ss_L = torch.linalg.cholesky(K_ss, upper=False)
                except torch.linalg.LinAlgError:
                    with torch.no_grad():
                        # Diagonal sum (trace)
                        diag_sum = torch.sum(torch.diag(K_ss))

                        # epsilon value
                        eps = torch.finfo(self.torch_data_type).eps

                        # scaling factor
                        scaling_factor = 1 / (1 / (4.0 * eps) - 1)

                        # adjust K_XX
                        adjustment = diag_sum * scaling_factor * torch.ones(K_ss.shape[0],
                                                                            dtype=self.torch_data_type)
                        K_ss.diagonal().add_(adjustment)

                        # Step 1: Cholesky decomposition for K_ss after adjusting
                        self.K_ss_L = torch.linalg.cholesky(K_ss, upper=False)

                # covariance between inducing points S and training points X
                self.K_sX = self.kernel.kernel_without_deriv(X1=self.sX, X2=self.X)

                # KK = [Nsparse, Nsparse]
                Qs = K_ss + torch.einsum('pi,ij,jq->pq', self.K_sX, self.inv_reg, self.K_sX.T)

                CKK_XX_L = torch.cholesky_solve(self.K_sX.T.clone(), self.K_XX_L, upper=False)
                Q_ss = torch.einsum('ij,jk->ik', self.K_sX, CKK_XX_L)
                self.CQ_ss = torch.cholesky_solve(Q_ss, self.K_ss_L, upper=False)

            self.model_vector = self.calculate_model_vector_sparse(matrix=Qs)

        else:
            if self.use_forces:  # self.kernel = FPKernel
                # covariance matrix between the training points X
                self.K_XX_L = self.kernel.kernel_matrix_batch(images=self.images, batch_size=self.train_batch_size)

                a = torch.full((self.Ntrain, 1), self.hyper_params['noise'] * self.hyper_params['noisefactor'],
                               dtype=self.torch_data_type, device=self.device)
                noise_val = self.hyper_params['noise']
                b = noise_val.expand(self.Ntrain, self.Nmask)

                # reg = torch.diag(torch.cat((a, b), 1).flatten() ** 2)
                diagonal_values = torch.cat((a, b), 1).flatten() ** 2

                self.K_XX_L.diagonal().add_(diagonal_values)

                try:
                    self.K_XX_L = torch.linalg.cholesky(self.K_XX_L, upper=False)

                except torch.linalg.LinAlgError:
                    with torch.no_grad():
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

            else:  # self.kernel = FPKernelNoforces
                # KK = [Ntrain, Ntrain]
                __K_XX = self.kernel.kernel_matrix(X=self.X)

                a = torch.tensor(self.Ntrain * [self.hyper_params['noise']],
                                 dtype=self.torch_data_type,
                                 device=self.device)
                reg = torch.diag(a ** 2)

                __K_XX.add_(reg)
                try:
                    self.K_XX_L = torch.linalg.cholesky(__K_XX, upper=False)
                except torch.linalg.LinAlgError:
                    with torch.no_grad():
                        # Diagonal sum (trace)
                        diag_sum = torch.sum(torch.diag(__K_XX))

                        # epsilon value
                        eps = torch.finfo(self.torch_data_type).eps

                        # scaling factor
                        scaling_factor = 1 / (1 / (4.0 * eps) - 1)

                        # adjust K_XX
                        adjustment = diag_sum * scaling_factor * torch.ones(__K_XX.shape[0],
                                                                            dtype=self.torch_data_type,
                                                                            device=self.device)
                        __K_XX.diagonal().add_(adjustment)

                        # Step 1: Cholesky decomposition for K_XX after adjusting
                        self.K_XX_L = torch.linalg.cholesky(__K_XX, upper=False)

            if self.prior_update:
                self.prior.update(self.images, self.YdY, self.K_XX_L, self.use_forces)
                self.hyper_params.update(dict(prior=self.prior.constant))
                self.kernel.set_params(self.hyper_params)

            _prior_array = self.prior.potential_batch(self.images, self.use_forces)
            self.model_vector = torch.cholesky_solve(self.YdY.contiguous().view(-1, 1) - _prior_array.view(-1, 1),
                                                     self.K_XX_L, upper=False)

        return

    def forward(self, eval_images, get_variance=False):

        Ntest = len(eval_images)
        eval_x_N_batch = get_N_batch(Ntest, self.eval_batch_size)
        eval_x_indexes = get_batch_indexes_N_batch(Ntest, eval_x_N_batch)

        if not get_variance:
            if self.use_forces:
                # E_hat = pred[0:x.shape[0]]
                # F_hat = pred[x.shape[0]:].reshape(x.shape[0], self.Natom, -1)
                E_hat = torch.empty((Ntest,), dtype=self.torch_data_type, device=self.device)
                F_hat = torch.zeros((Ntest, self.Natom * 3), dtype=self.torch_data_type, device=self.device)

                for i in range(0, eval_x_N_batch):
                    pred, kernel = self.eval_data_batch(eval_images=eval_images[eval_x_indexes[i][0]:eval_x_indexes[i][1]])

                    data_per_batch = eval_x_indexes[i][1] - eval_x_indexes[i][0]
                    E_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1]] = pred[0:data_per_batch]
                    if self.atoms_mask is not None:
                        F_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1], self.atoms_mask] = pred[data_per_batch:].view(data_per_batch, -1)
                    else:
                        F_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1], :] = pred[data_per_batch:].view(data_per_batch, -1)

                return E_hat, F_hat.view((Ntest, self.Natom, 3)), None

            else:
                pass
                # E_hat = pred
                #
                # return E_hat, None, None

        else:
            if self.use_forces:
                E_hat = torch.empty((Ntest,), dtype=self.torch_data_type, device=self.device)
                F_hat = torch.zeros((Ntest, self.Natom * 3), dtype=self.torch_data_type, device=self.device)
                uncertainty = torch.empty((Ntest,), dtype=self.torch_data_type, device=self.device)

                for i in range(0, eval_x_N_batch):
                    pred, kernel = self.eval_data_batch(eval_images=eval_images[eval_x_indexes[i][0]:eval_x_indexes[i][1]])
                    var = self.eval_variance_batch(get_variance=get_variance,
                                                   eval_images=eval_images[eval_x_indexes[i][0]:eval_x_indexes[i][1]],
                                                   k=kernel)

                    data_per_batch = eval_x_indexes[i][1] - eval_x_indexes[i][0]
                    E_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1]] = pred[0:data_per_batch]
                    if self.atoms_mask is not None:
                        F_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1], self.atoms_mask] = pred[data_per_batch:].view(data_per_batch, -1)
                    else:
                        F_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1], :] = pred[data_per_batch:].view(data_per_batch, -1)

                    uncertainty[eval_x_indexes[i][0]:eval_x_indexes[i][1]] = torch.sqrt(torch.diagonal(var)[0:data_per_batch]) / self.weight

                return E_hat, F_hat.view((Ntest, self.Natom, 3)), uncertainty

            else:
                pass
                # E_hat = pred
                # uncertainty_squared = torch.diagonal(var)
                # uncertainty = torch.sqrt(uncertainty_squared)
                #
                # return E_hat, None, uncertainty

    def eval_data_batch(self, eval_images):

        # kernel between test point x and inducing points S
        if self.sparse:
            if self.use_forces:
                kernel = self.kernel.kernel_vector_batch(eval_images=eval_images,
                                                         train_images=self.images,
                                                         batch_size=self.eval_batch_size)
            # else:
            #     kernel = self.kernel.kernel_vector(x=x, X=self.sX)

            # pred = torch.einsum('hi,i->h', kernel, self.model_vector.flatten())
            pred = torch.matmul(kernel, self.model_vector.view(-1))

        # kernel between test point x and training points X
        else:
            if self.use_forces:
                kernel = self.kernel.kernel_vector_batch(eval_images=eval_images,
                                                         train_images=self.images,
                                                         batch_size=self.eval_batch_size)
            # else:
            #     kernel = self.kernel.kernel_vector(x=x, X=self.X)

            # pred = torch.einsum('hi,i->h', kernel, self.model_vector.flatten())
            pred = torch.matmul(kernel, self.model_vector.view(-1)) + self.prior.potential_batch(eval_images, self.use_forces)

        return pred, kernel

    def eval_data_per_data(self, eval_image):

        # kernel between test point x and inducing points S
        if self.sparse:
            if self.use_forces:
                kernel = self.kernel.kernel_vector_per_data(eval_image=eval_image,
                                                            train_images=self.images)

            pred = torch.matmul(kernel, self.model_vector.view(-1))

        # kernel between test point x and training points X
        else:
            if self.use_forces:
                kernel = self.kernel.kernel_vector_per_data(eval_image=eval_image,
                                                            train_images=self.images)

            pred = torch.matmul(kernel, self.model_vector.view(-1)) + self.prior.potential_per_data(eval_image, self.use_forces)

        return pred, kernel

    def eval_variance_batch(self, get_variance, eval_images, k):
        """

        variance.shape  # [Ntest * (1 + 3 * Natom), Ntest * (1 + 3 * Natom)]
        self.K_XX_L.shape  # [Ntrain * (1 + 3 * Natom), Ntrain * (1 + 3 * Natom)]
        k.T.clone().shape  # [Ntrain * (1 + 3 * Natom), Ntest * (1 + 3 * Natom)]
        self.Ck.shape  # [Ntrain * (1 + 3 * Natom), Ntest * (1 + 3 * Natom)]
        covariance.shape  # [Ntest * (1 + 3 * Natom), Ntest * (1 + 3 * Natom)]
        """
        # var = None
        if get_variance:
            # Compute variance of test points x
            # if self.use_forces:
            #     variance = self.kernel.kernel_matrix_batch(X=x, dX=dx)
            # else:
            #     variance = self.kernel.kernel_matrix(X=x)

            # Perform Cholesky decomposition and solve the system
            if self.sparse:
                # Step 2: Cholesky solve
                CK_ss_L = torch.cholesky_solve(k.T.clone(), self.K_ss_L, upper=False)

                covariance = torch.einsum('pi,ij,jq->pq', k,
                                          torch.eye(self.CQ_ss.shape[0], dtype=self.torch_data_type) - self.CQ_ss,
                                          CK_ss_L)

            else:
                # CK_XX_L = torch.cholesky_solve(k.T.clone(), self.K_XX_L, upper=False)
                # covariance = torch.einsum('ij,jk->ik', k, CK_XX_L)
                covariance = torch.matmul(k, torch.cholesky_solve(k.T.clone(), self.K_XX_L, upper=False))

            # Adjust variance by subtracting covariance
            if self.use_forces:
                return self.kernel.kernel_matrix_batch(images=eval_images, batch_size=self.eval_batch_size) - covariance
            # else:
            #     return self.kernel.kernel_matrix(X=x) - covariance

        else:
            return None

    def eval_variance_per_data(self, get_variance, eval_image, k):

        if get_variance:

            # Perform Cholesky decomposition and solve the system
            if self.sparse:
                # Step 2: Cholesky solve
                CK_ss_L = torch.cholesky_solve(k.T.clone(), self.K_ss_L, upper=False)

                covariance = torch.einsum('pi,ij,jq->pq', k,
                                          torch.eye(self.CQ_ss.shape[0], dtype=self.torch_data_type) - self.CQ_ss,
                                          CK_ss_L)

            else:
                # covariance = torch.einsum('ij,jk->ik', k, CK_XX_L)
                covariance = torch.matmul(k, torch.cholesky_solve(k.T.clone(), self.K_XX_L, upper=False))

            # Adjust variance by subtracting covariance
            if self.use_forces:
                return self.kernel.kernel_matrix_per_data(image=eval_image) - covariance
            # else:
            #     return self.kernel.kernel_matrix(X=x) - covariance

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
