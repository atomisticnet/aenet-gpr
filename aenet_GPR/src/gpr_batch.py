import torch
import torch.nn as nn
import numpy as np

from aenet_GPR.src.pytorch_kernel import FPKernel, FPKernelNoforces  # , FPKernelParallel
from aenet_GPR.util.prepare_data import get_N_batch, get_batch_indexes_N_batch


class GaussianProcess(nn.Module):
    '''
    Gaussian Process Regression
    Parameters:

    prior: Prior class, as in ase.optimize.gpmin.prior
        Defaults to ConstantPrior with zero as constant

    kernel: Kernel function for the regression, as
       in ase.optimize.gpmin.kernel
        Defaults to the Squared Exponential kernel with derivatives
    '''

    def __init__(self, hp=None, prior=None, kernel=None, kerneltype='sqexp',
                 scale=0.4, weight=1.0, noise=1e-6, noisefactor=0.5,
                 use_forces=True, images=None, function=None, derivative=None,
                 sparse=None, sparse_derivative=None, autograd=False,
                 train_batch_size=25, eval_batch_size=25,
                 data_type='float64', device='cpu', soap_param=None, descriptor='cartesian coordinates'):
        super().__init__()

        if data_type == 'float32':
            self.data_type = 'float32'
            self.torch_data_type = torch.float32
        else:
            self.data_type = 'float64'
            self.torch_data_type = torch.float64

        self.device = device
        self.soap_param = soap_param
        self.descriptor = descriptor
        self.kerneltype = kerneltype

        if autograd:
            self.scale = nn.Parameter(torch.tensor(scale, dtype=self.torch_data_type), requires_grad=True).to(
                self.device)
            self.weight = nn.Parameter(torch.tensor(weight, dtype=self.torch_data_type), requires_grad=True).to(
                self.device)
        else:
            self.scale = torch.tensor(scale, dtype=self.torch_data_type)
            self.weight = torch.tensor(weight, dtype=self.torch_data_type)

        self.noise = torch.tensor(noise, dtype=self.torch_data_type)
        self.noisefactor = torch.tensor(noisefactor, dtype=self.torch_data_type)

        self.use_forces = use_forces
        self.images = images
        self.Ntrain = len(self.images)
        self.species = self.images[0].get_chemical_symbols()
        self.pbc = np.all(self.images[0].get_pbc())
        self.Natom = len(self.species)

        self.Y = function  # Y = [Ntrain]
        self.dY = derivative  # dY = [Ntrain, Natom, 3]
        self.model_vector = torch.empty((self.Ntrain * (1 + 3 * self.Natom),), dtype=self.torch_data_type)

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
                                   descriptor=self.descriptor,
                                   device=self.device)
        else:
            self.kernel = FPKernelNoforces(species=self.species,
                                           pbc=self.pbc,
                                           Natom=self.Natom,
                                           kerneltype=self.kerneltype,
                                           data_type=self.data_type,
                                           soap_param=self.soap_param,
                                           descriptor=self.descriptor,
                                           device=self.device)

        hyper_params = dict(kerneltype=self.kerneltype,
                            scale=self.scale,
                            weight=self.weight,
                            noise=self.noise,
                            noisefactor=self.noisefactor)

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

    def train_model(self):
        if self.sparse:
            if self.use_forces:  # self.kernel = FPKernel
                # covariance matrix between the training points X
                K_XX = self.kernel.kernel_matrix_batch(images=self.images, batch_size=self.train_batch_size)

                # reg = [Ntrain * (1 + 3 * Natom), Ntrain * (1 + 3 * Natom)]
                a = torch.tensor(self.Ntrain * [self.hyper_params['noise'] * self.hyper_params['noisefactor']],
                                 dtype=self.torch_data_type).reshape(
                    self.Ntrain, 1)
                b = torch.tensor(self.Ntrain * 3 * self.Natom * [self.hyper_params['noise']],
                                 dtype=self.torch_data_type).reshape(self.Ntrain, -1)
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

                a = torch.tensor(self.Ntrain * [self.hyper_params['noise']], dtype=self.torch_data_type)
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

                # a = torch.tensor(self.Ntrain * [self.hyper_params['noise'] * self.hyper_params['noisefactor']], dtype=torch.float64).reshape(self.Ntrain, 1)
                # b = torch.tensor(self.Ntrain * 3 * self.Natom * [self.hyper_params['noise']], dtype=torch.float64).reshape(self.Ntrain, -1)
                a = torch.full((self.Ntrain, 1), self.hyper_params['noise'] * self.hyper_params['noisefactor'],
                               dtype=self.torch_data_type)
                noise_val = self.hyper_params['noise']
                b = noise_val.expand(self.Ntrain, 3 * self.Natom)

                # reg = torch.diag(torch.cat((a, b), 1).flatten() ** 2)
                diagonal_values = torch.cat((a, b), 1).flatten() ** 2

                self.K_XX_L.diagonal().add_(diagonal_values)

                try:
                    self.K_XX_L = torch.linalg.cholesky(self.K_XX_L, upper=False)
                    # torch.linalg.cholesky(self.K_XX_L, upper=False, out=self.K_XX_L)

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
                                                                            dtype=self.torch_data_type)
                        self.K_XX_L.diagonal().add_(adjustment)

                        # Step 1: Cholesky decomposition for K_XX after adjusting
                        self.K_XX_L = torch.linalg.cholesky(self.K_XX_L, upper=False)

            else:  # self.kernel = FPKernelNoforces
                # KK = [Ntrain, Ntrain]
                __K_XX = self.kernel.kernel_matrix(X=self.X)

                a = torch.tensor(self.Ntrain * [self.hyper_params['noise']], dtype=self.torch_data_type)
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
                                                                            dtype=self.torch_data_type)
                        __K_XX.diagonal().add_(adjustment)

                        # Step 1: Cholesky decomposition for K_XX after adjusting
                        self.K_XX_L = torch.linalg.cholesky(__K_XX, upper=False)

            self.model_vector = torch.cholesky_solve(self.YdY.contiguous().view(-1, 1), self.K_XX_L, upper=False)

        return

    def calculate_model_vector(self, matrix):
        """
        What is the role of self.prior_array?
        
        self.YdY.shape  # [Ntrain, 1 + 3 * Natom]
        model_vector.shape  # [Ntrain * (1 + 3 * Natom)]
        model_vector.unsqueeze(1).shape  # [Ntrain * (1 + 3 * Natom), 1]
        self.K_XX_L.shape  # [Ntrain * (1 + 3 * Natom), Ntrain * (1 + 3 * Natom)]
        """

        # Factorize K-matrix (Cholesky decomposition) using torch.linalg.cholesky:
        self.K_XX_L = torch.linalg.cholesky(matrix, upper=False)  # Lower triangular by default

        # Compute the prior array
        # self.prior_array = self.calculate_prior_array(self.X, get_forces=self.use_forces)

        # Flatten Y and compute the model vector
        model_vector = self.YdY.flatten()  # - self.prior_array

        # Solve the system L * L^T * v = model_vector (Cholesky solve)
        model_vector = torch.cholesky_solve(model_vector.unsqueeze(1), self.K_XX_L, upper=False)

        return model_vector

    def calculate_model_vector_sparse(self, matrix):
        # [Nsparse * (1 + 3 * Natom), Ntrain * (1 + 3 * Natom)] * 
        # [Ntrain * (1 + 3 * Natom), Ntrain * (1 + 3 * Natom)] *
        # [Ntrain * (1 + 3 * Natom), 1]
        # -> [Nsparse * (1 + 3 * Natom), 1]
        reduced_target = torch.einsum('pi,ij,jq->pq', self.K_sX, self.inv_reg, self.YdY.reshape(-1, 1))

        try:
            matrix_L = torch.linalg.cholesky(matrix, upper=False)
        except torch.linalg.LinAlgError:
            with torch.no_grad():
                # Diagonal sum (trace)
                diag_sum = torch.sum(torch.diag(matrix))

                # epsilon value
                eps = torch.finfo(self.torch_data_type).eps

                # scaling factor
                scaling_factor = 1 / (1 / (4.0 * eps) - 1)

                # adjust K_XX
                adjustment = diag_sum * scaling_factor * torch.ones(matrix.shape[0],
                                                                    dtype=self.torch_data_type)
                matrix.diagonal().add_(adjustment)

                # Step 1: Cholesky decomposition for K_XX after adjusting
                matrix_L = torch.linalg.cholesky(matrix, upper=False)

        # Solve the system L * L^T * v = model_vector (Cholesky solve)
        model_vector = torch.cholesky_solve(reduced_target, matrix_L, upper=False)

        return model_vector

    def calculate_prior_array(self, list_of_fingerprints, get_forces=True):

        if get_forces:
            return list(torch.hstack([self.prior.potential(x) for x in list_of_fingerprints]))
        else:
            return list(torch.hstack([self.prior.potential(x)[0] for x in list_of_fingerprints]))

    def forward(self, eval_images, get_variance=False):

        Ntest = len(eval_images)
        eval_x_N_batch = get_N_batch(Ntest, self.eval_batch_size)
        eval_x_indexes = get_batch_indexes_N_batch(Ntest, eval_x_N_batch)

        if not get_variance:
            if self.use_forces:
                # E_hat = pred[0:x.shape[0]]
                # F_hat = pred[x.shape[0]:].reshape(x.shape[0], self.Natom, -1)
                E_hat = torch.empty((Ntest,), dtype=self.torch_data_type)
                F_hat = torch.empty((Ntest, self.Natom, 3), dtype=self.torch_data_type)

                for i in range(0, eval_x_N_batch):
                    pred, kernel = self.eval_data_batch(eval_images=eval_images[eval_x_indexes[i][0]:eval_x_indexes[i][1]])

                    data_per_batch = eval_x_indexes[i][1] - eval_x_indexes[i][0]
                    E_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1]] = pred[0:data_per_batch]
                    F_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1], :, :] = pred[data_per_batch:].view(data_per_batch,
                                                                                                        self.Natom, 3)

                return E_hat, F_hat, None

            else:
                pass
                # E_hat = pred
                #
                # return E_hat, None, None

        else:
            if self.use_forces:
                E_hat = torch.empty((Ntest,), dtype=self.torch_data_type)
                F_hat = torch.empty((Ntest, self.Natom, 3), dtype=self.torch_data_type)
                uncertainty = torch.empty((Ntest,), dtype=self.torch_data_type)

                for i in range(0, eval_x_N_batch):
                    pred, kernel = self.eval_data_batch(eval_images=eval_images[eval_x_indexes[i][0]:eval_x_indexes[i][1]])
                    var = self.eval_variance_batch(get_variance=get_variance,
                                                   eval_images=eval_images[eval_x_indexes[i][0]:eval_x_indexes[i][1]],
                                                   k=kernel)

                    data_per_batch = eval_x_indexes[i][1] - eval_x_indexes[i][0]
                    E_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1]] = pred[0:data_per_batch]
                    F_hat[eval_x_indexes[i][0]:eval_x_indexes[i][1], :, :] = pred[data_per_batch:].view(data_per_batch,
                                                                                                        self.Natom, 3)
                    uncertainty[eval_x_indexes[i][0]:eval_x_indexes[i][1]] = torch.sqrt(
                        torch.diagonal(var)[0:data_per_batch])

                return E_hat, F_hat, uncertainty

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
            pred = torch.matmul(kernel, self.model_vector.view(-1))

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
