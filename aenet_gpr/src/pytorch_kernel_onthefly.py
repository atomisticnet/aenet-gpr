import torch
import numpy as np
import gc

from aenet_gpr.src.pytorch_kerneltypes import SquaredExp
from aenet_gpr.util.prepare_data import get_N_batch, get_batch_indexes_N_batch


class BaseKernelType:
    '''
    Base class for all kernel types with common properties,
    attributes and methods.
    '''

    def __init__(self, params=None):
        # Currently, all the kernel types take weight and scale as parameters

        if params is None:
            self.params = {}

    @property
    def scale(self):
        return self.params['scale']

    @property
    def weight(self):
        return self.params['weight']

    def update(self, params):
        '''
        Update the kernel function hyperparameters.
        '''

        self.params.update(params)


class FPKernel(BaseKernelType):

    def __init__(self,
                 species, pbc, Natom,
                 kerneltype='sqexp',
                 params=None,
                 data_type='float64',
                 soap_param=None,
                 mace_param=None,
                 descriptor='cartesian coordinates',
                 device='cpu',
                 atoms_mask=None):
        super().__init__()
        '''
        params: dict
            Hyperparameters for the kernel type
        '''
        kerneltypes = {'sqexp': SquaredExp}
        self.device = device
        self.atoms_mask = atoms_mask

        if data_type == 'float32':
            self.data_type = 'float32'
            self.torch_data_type = torch.float32
        else:
            self.data_type = 'float64'
            self.torch_data_type = torch.float64

        self.species = species
        self.pbc = pbc
        self.Natom = Natom

        if self.atoms_mask is not None:
            self.Nmask = self.atoms_mask.shape[0]
            self.free_atoms = self.atoms_mask // 3
            self.free_atoms = torch.unique(self.free_atoms)
        else:
            self.Nmask = 3 * self.Natom

        if params is None:
            params = {}

        kerneltype = kerneltypes.get(kerneltype)
        self.kerneltype = kerneltype(**params)

        self.soap_param = soap_param
        self.mace_param = mace_param
        self.descriptor = descriptor
        self.soap = None

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
                    from mace_descriptor.calculators import mace_mp
                except ImportError:
                    raise ImportError(
                        "The 'mace-descriptor' package is required for using pre-trained MACE descriptors.\n"
                        "Please install it by running:\n\n"
                        "    pip install mace-descriptor\n\n"
                        "Note: This is a lightweight fork of the original MACE (mace-torch) package, "
                        "designed exclusively for descriptor extraction."
                    )

                self.mace = mace_mp(model=self.mace_param.get('model'), default_dtype=self.data_type, device=self.device)

            else:
                try:
                    from mace_descriptor.calculators import mace_off
                except ImportError:
                    raise ImportError(
                        "The 'mace-descriptor' package is required for using pre-trained MACE descriptors.\n"
                        "Please install it by running:\n\n"
                        "    pip install mace-descriptor\n\n"
                        "Note: This is a lightweight fork of the original MACE (mace-torch) package, "
                        "designed exclusively for descriptor extraction."
                    )

                self.mace = mace_off(model=self.mace_param.get('model'), default_dtype=self.data_type, device=self.device)

    def pairwise_distances(self, fingerprints, sin_transform=False):
        # fingerprints: (Ndata, Ncenter, Nfeature)
        Ndata = fingerprints.shape[0]

        distances = []
        for i in range(Ndata):
            for j in range(i + 1, Ndata):
                diff = fingerprints[i].flatten() - fingerprints[j].flatten()
                if sin_transform:
                    diff = torch.sin(diff / 2)
                distance = torch.linalg.norm(diff)
                distances.append(distance.item())

        distances = torch.tensor(distances)
        mean_distance = torch.mean(distances)
        std_distance = torch.std(distances)
        max_distance = torch.max(distances)
        min_distance = torch.min(distances)

        return mean_distance, std_distance, max_distance, min_distance

    def generate_descriptor(self, images):

        if self.descriptor == 'soap':
            dfp_dr, fp = self.soap.derivatives(images,
                                               centers=[self.soap_param.get('centers')] * len(images),
                                               method=self.soap_param.get('method'),
                                               return_descriptor=True,
                                               n_jobs=self.soap_param.get('n_jobs'))

            if self.atoms_mask is not None:
                dfp_dr = torch.as_tensor(dfp_dr, dtype=self.torch_data_type).to(self.device)[:, :, :, :, self.atoms_mask]  # (Ndata, Ncenters, Natom, 3, Natoms_masked)
                dfp_dr = dfp_dr[:, :, self.free_atoms, :, :]  # (Ndata, Ncenters, Natom_masked, 3, Natoms_masked)
                fp = torch.as_tensor(fp, dtype=self.torch_data_type).to(self.device)[:, :, self.atoms_mask]  # (Ndata, Ncenters, Natoms_masked)
            else:
                dfp_dr = torch.as_tensor(dfp_dr, dtype=self.torch_data_type).to(self.device)  # (Ndata, Ncenters, Natom, 3, Natom*3)
                fp = torch.as_tensor(fp, dtype=self.torch_data_type).to(self.device)  # (Ndata, Ncenters, Natom*3)

        elif self.descriptor == 'mace':
            fp = []
            dfp_dr = []
            for image in images:
                fp.append(self.mace.get_descriptors(image).to(self.device))
                dfp_dr.append(numerical_descriptor_gradient(image).to(self.device))

            fp = torch.stack(fp).to(self.device)  # (Ndata, Natom, Ndescriptor)
            dfp_dr = torch.stack(dfp_dr).to(self.device)  # (Ndata, Natom, Natom, 3, Ndescriptor)

        else:
            fp = []
            dfp_dr = []
            for image in images:
                fp.append(torch.as_tensor(image.get_positions(wrap=False).reshape(-1), dtype=self.torch_data_type).to(self.device))

                dfp_dr.append(torch.as_tensor(np.eye(self.Natom * 3).reshape(self.Natom, -1, 3, order='F'),
                                              dtype=self.torch_data_type).to(self.device))

            fp = torch.stack(fp).to(self.device)  # (Ndata, Natom*3)
            dfp_dr = torch.stack(dfp_dr).to(self.device)  # (Ndata, Natom, Natom*3, 3)

            if self.atoms_mask is not None:
                fp = fp.unsqueeze(1)[:, :, self.atoms_mask]  # (Ndata, 1, Natoms_masked)
                dfp_dr = dfp_dr.transpose(2, 3).unsqueeze(1)[:, :, :, :, self.atoms_mask]  # (Ndata, 1, Natom, 3, Natoms_masked)
                dfp_dr = dfp_dr[:, :, self.free_atoms, :, :]  # (Ndata, Ncenters, Natom_masked, 3, Natoms_masked)
            else:
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

            if self.atoms_mask is not None:
                dfp_dr = torch.as_tensor(dfp_dr, dtype=self.torch_data_type).to(self.device)[:, :, :, self.atoms_mask]  # (Ncenters, Natom, 3, Natoms_masked)
                dfp_dr = dfp_dr[:, self.free_atoms, :, :]  # (Ncenters, Natom_masked, 3, Natoms_masked)
                fp = torch.as_tensor(fp, dtype=self.torch_data_type).to(self.device)[:, self.atoms_mask]  # (Ncenters, Natoms_masked)
            else:
                dfp_dr = torch.as_tensor(dfp_dr, dtype=self.torch_data_type).to(self.device)  # (Ncenters, Natom, 3, Natom*3)
                fp = torch.as_tensor(fp, dtype=self.torch_data_type).to(self.device)  # (Ncenters, Natom*3)

        elif self.descriptor == 'mace':
            fp = self.mace.get_descriptors(image).to(self.device)  # (Natom, Ndescriptor)
            dfp_dr = numerical_descriptor_gradient(image).to(self.device)  # (Natom, Natom, 3, Ndescriptor)

        else:
            fp = torch.as_tensor(image.get_positions(wrap=False).reshape(-1), dtype=self.torch_data_type).to(self.device)
            dfp_dr = torch.as_tensor(np.eye(self.Natom * 3).reshape(self.Natom, -1, 3, order='F'),
                                     dtype=self.torch_data_type).to(self.device)

            if self.atoms_mask is not None:
                fp = fp.unsqueeze(0)[:, self.atoms_mask]
                dfp_dr = dfp_dr.transpose(1, 2).unsqueeze(0)[:, :, :, self.atoms_mask]
                dfp_dr = dfp_dr[:, self.free_atoms, :, :]
            else:
                fp = fp.unsqueeze(0)
                dfp_dr = dfp_dr.transpose(1, 2).unsqueeze(0)

        return fp, dfp_dr

    def kernel_with_deriv(self, X1=None, dX1=None, X2=None, dX2=None):
        '''
        Data process type: Broadcast, Batch Processing
        Return a full kernel matrix between two structural fingerprints, 'X1' and 'X2'.
        X1, X2 = [Ncenter, Nfeature] -> [Ndata, Ncenter, Nfeature]
        dX1, dX2 = [Ncenter, Natom, 3, Nfeature] -> [Ndata, Ncenter, Natom, 3, Nfeature]
        where Ndata = Ntrain or Nsparse or Ntest
        '''

        # Create empty kernel
        # [[K(X11,X21), K(X11,X22), ..., K(X11,X2N), K(X11,dX21x), K(X11,dX21y), ..., K(X11,dX2Nz)],
        #  [K(X12,X21), K(X12,X22), ..., K(X12,X2N), K(X12,dX21x), K(X12,dX21y), ..., K(X12,dX2Nz)]
        #  ...
        #  [K(X1N,X21), K(X1N,X22), ..., K(X1N,X2N), K(X1N,dX21x), K(X1N,dX21y), ..., K(X1N,dX2Nz)]
        #  [K(dX11x,X21), K(dX11x,X22), ..., K(dX11x,X2N), K(dX11x,dX21x), K(dX11x,dX21y), ..., K(dX11x,dX2Nz)]
        #  ...
        #  [K(dX1Nz,X21), K(dX1Nz,X22), ..., K(dX1Nz,X2N), K(dX1Nz,dX21x), K(dX1Nz,dX21y), ..., K(dX1Nz,dX2Nz)]

        K_X1X2 = torch.empty(((1 + self.Nmask) * X1.shape[0],
                              (1 + self.Nmask) * X2.shape[0]), dtype=self.torch_data_type, device=self.device)

        # Expand value fingerprint
        X1_expanded = X1[:, None, :, None, :]  # [Ndata1, 1, Ncenter, 1, Nfeature]
        X2_expanded = X2[None, :, None, :, :]  # [1, Ndata2, 1, Ncenter, Nfeature]

        X1__outer_minus__X2 = X1_expanded - X2_expanded  # [Ndata1, Ndata2, Ncenter, Ncenter, Nfeature]
        X2__outer_minus__X1 = X2_expanded - X1_expanded  # [Ndata1, Ndata2, Ncenter, Ncenter, Nfeature]

        # [Ndata, Ndata, Ncenter, Ncenter]
        __k_X1X2 = self.weight ** 2 * torch.exp(
            -torch.linalg.norm(X1__outer_minus__X2, dim=4) ** 2 / (2 * self.scale ** 2))
        __k_X2X1 = self.weight ** 2 * torch.exp(
            -torch.linalg.norm(X2__outer_minus__X1, dim=4) ** 2 / (2 * self.scale ** 2))

        """"K_X1X2[:X1.shape[0], :X2.shape[0]] kernel between fp1 and fp2"""
        # kernel = torch.sum(__k_X1X2, dim=(2, 3))
        # K_X1X2[:X1.shape[0], :X2.shape[0]].copy_(kernel)
        K_X1X2[:X1.shape[0], :X2.shape[0]] = torch.sum(__k_X1X2, dim=(2, 3))

        """K_X1X2[X1.shape[0]: X1.shape[0] * (1 + 3 * Natom), :X2.shape[0]] kernel between fp1_deriv and fp2"""
        # [Ndata, Ncenter, Natom, 3, Nfeature] -> [Ndata, Ncenter, Natom * 3, Nfeature]
        dX1_reshaped = dX1.flatten(start_dim=2, end_dim=3)
        # intermediate_result = torch.einsum('xycnf,xcdf->xycnd', X1__outer_minus__X2, dX1_reshaped)
        # intermediate_result = torch.matmul(X1__outer_minus__X2, dX1_reshaped.transpose(-1, -2))

        # [Ndata1, Ndata2, Natom * 3]
        # kernel = torch.einsum('xycn,xycnd->xyd', __k_X1X2 / self.scale**2, intermediate_result)
        # kernel = torch.matmul(__k_X1X2 / self.scale ** 2, intermediate_result).sum(dim=(2, 3))

        # [Ndata1 * Natom * 3, Ndata2]
        # K_X1X2[X1.shape[0]:, :X2.shape[0]] = kernel.permute(0, 2, 1).reshape(X1.shape[0] * 3 * self.Natom, X2.shape[0])
        # K_X1X2[X1.shape[0]:, :X2.shape[0]].copy_(
        #     kernel.permute(0, 2, 1).contiguous().view(X1.shape[0] * 3 * self.Natom, X2.shape[0]))
        K_X1X2[X1.shape[0]:, :X2.shape[0]].copy_(torch.einsum('xycn,xycnd->xyd', __k_X1X2 / self.scale ** 2,
                                                              torch.einsum('xycnf,xcdf->xycnd', X1__outer_minus__X2,
                                                                           dX1_reshaped)).permute(0, 2,
                                                                                                  1).contiguous().view(
            X1.shape[0] * self.Nmask, X2.shape[0]))

        """K_X1X2[:X1.shape[0], X2.shape[0]: X2.shape[0] * (1 + 3 * Natom)] kernel between fp1 and fp2_deriv"""
        # [Ndata, Ncenter, Natom, 3, Nfeature] -> [Ndata, Ncenter, Natom * 3, Nfeature]
        dX2_reshaped = dX2.flatten(start_dim=2, end_dim=3)
        # intermediate_result = torch.einsum('xycnf,yndf->xycnd', X2__outer_minus__X1, dX2_reshaped)
        # intermediate_result = torch.matmul(X2__outer_minus__X1, dX2_reshaped.transpose(-1, -2))

        # kernel = torch.einsum('xycn,xycnd->xyd', __k_X2X1 / self.scale**2, intermediate_result)
        # kernel = torch.matmul(__k_X2X1 / self.scale ** 2, intermediate_result).sum(dim=(2, 3))

        # [Ndata1, Ndata2 * Natom * 3]
        # K_X1X2[:X1.shape[0], X2.shape[0]:] = kernel.reshape(X1.shape[0], X2.shape[0] * 3 * self.Natom)
        # K_X1X2[:X1.shape[0], X2.shape[0]:].copy_(
        #     kernel.view(X1.shape[0], X2.shape[0] * 3 * self.Natom))
        K_X1X2[:X1.shape[0], X2.shape[0]:].copy_(torch.einsum('xycn,xycnd->xyd', __k_X2X1 / self.scale ** 2,
                                                              torch.einsum('xycnf,yndf->xycnd', X2__outer_minus__X1,
                                                                           dX2_reshaped)).view(X1.shape[0], X2.shape[
            0] * self.Nmask))

        """K_X1X2[X1.shape[0]: X1.shape[0] * (1 + 3 * Natom), X2.shape[0]: X2.shape[0] * (1 + 3 * Natom)] 
        kernel between fp1_deriv and fp2_deriv"""
        # size sign in eisum:
        # x, y: Ndata
        # c, n: Ncenter
        # b, d: Natom *3
        # f: Nfeature
        # [Ndata, Ndata, Ncenter, Ncenter, Nfeature] * [Ndata, Ndata, Ncenter, Natom * 3, Nfeature]
        # -> [Ndata, Ndata, Ncenter, Ncenter, Natom * 3]
        DdD_dr1 = torch.einsum('xycnf,x...cbf->xycnb',
                               X1__outer_minus__X2,
                               dX1_reshaped)
        # -> [Ndata, Ndata, Ncenter, Ncenter, Natom * 3]
        # DdD_dr1 = torch.matmul(X1__outer_minus__X2,
        #                        dX1_reshaped.transpose(-1, -2)).sum(dim=-1)

        DdD_dr2 = torch.einsum('xycnf,...yndf->xycnd',
                               X2__outer_minus__X1,
                               dX2_reshaped)
        # -> [Ndata, Ndata, Ncenter, Ncenter, Natom * 3]
        # DdD_dr2 = torch.matmul(X2__outer_minus__X1,
        #                        dX2_reshaped.transpose(-1, -2)).sum(dim=-1)

        # [Ndata, Ndata, Natom * 3, Natom * 3]
        # -> [Ndata, Ndata, Ncenter, Ncenter, Natom * 3, Natom * 3]
        # intermediate_result = torch.einsum('xycnb,xycnd->xycnbd', DdD_dr1, DdD_dr2)
        # intermediate_result = torch.matmul(DdD_dr1.unsqueeze(-1), DdD_dr2.unsqueeze(-2))

        # C0 = torch.einsum('xycn,xycnbd->xybd',
        #                   __k_X1X2 / self.scale**4, intermediate_result)
        # -> [Ndata, Ndata, Natom * 3, Natom * 3]
        # C0 = ((__k_X1X2 / self.scale ** 4).unsqueeze(-1).unsqueeze(-1) * intermediate_result).sum(dim=(2, 3))
        # kernel = ((__k_X1X2 / self.scale ** 4).unsqueeze(-1).unsqueeze(-1)
        #           * torch.matmul(DdD_dr1.unsqueeze(-1), DdD_dr2.unsqueeze(-2))).sum(dim=(2, 3))
        # kernel = torch.einsum('xycn,xycnbd->xybd', __k_X1X2 / self.scale**4,
        #                       torch.einsum('xycnb,xycnd->xycnbd', DdD_dr1, DdD_dr2))

        # [Ndata, Ncenter, Natom * 3, Nfeature] * [Ndata, Ncenter, Natom * 3, Nfeature]
        # -> [Ndata, Ndata, Ncenter, Ncenter, Natom * 3, Natom * 3]
        # intermediate_result = torch.einsum('xcbf,yndf->xycnbd', dX1_reshaped, dX2_reshaped)
        # intermediate_result = torch.matmul(dX1_reshaped.unsqueeze(-1), dX2_reshaped.unsqueeze(-2)).sum(dim=-1)

        # C1 = torch.einsum('xycn,xycnbd->xybd',
        #                   __k_X1X2 / self.scale**2, intermediate_result)
        # C1 = ((__k_X1X2 / self.scale ** 2).unsqueeze(-1).unsqueeze(-1) * intermediate_result).sum(dim=(2, 3))

        # kernel = C0 + C1
        # kernel.add_(((__k_X1X2 / self.scale ** 2).unsqueeze(-1).unsqueeze(-1)
        #              * torch.matmul(dX1_reshaped.unsqueeze(-1), dX2_reshaped.unsqueeze(-2)).sum(dim=-1)).sum(dim=(2, 3)))
        # kernel.add_(torch.einsum('xycn,xycnbd->xybd', __k_X1X2 / self.scale**2,
        #                          torch.einsum('xcbf,yndf->xycnbd', dX1_reshaped, dX2_reshaped)))

        # [Ndata1 * Natom * 3, Ndata2 * Natom * 3]
        # K_X1X2[X1.shape[0]:, X2.shape[0]:] = kernel.permute(0, 2, 1, 3).reshape(X1.shape[0] * 3 * self.Natom,
        #                                                                         X2.shape[0] * 3 * self.Natom)
        # K_X1X2[X1.shape[0]:, X2.shape[0]:].copy_(
        #     kernel.permute(0, 2, 1, 3).contiguous().view(X1.shape[0] * 3 * self.Natom,
        #                                                  X2.shape[0] * 3 * self.Natom))

        # intermediate_result = torch.einsum('xycnb,xycnd->xycnbd', DdD_dr1, DdD_dr2)
        K_X1X2[X1.shape[0]:, X2.shape[0]:].copy_(torch.einsum('xycn,xycnbd->xybd', __k_X1X2 / self.scale ** 4,
                                                              torch.einsum('xycnb,xycnd->xycnbd', DdD_dr1,
                                                                           DdD_dr2)).permute(0, 2, 1,
                                                                                             3).contiguous().view(
            X1.shape[0] * self.Nmask, X2.shape[0] * self.Nmask))

        # intermediate_result = torch.einsum('xcbf,yndf->xycnbd', dX1_reshaped, dX2_reshaped)
        K_X1X2[X1.shape[0]:, X2.shape[0]:].add_(torch.einsum('xycn,xycnbd->xybd', __k_X1X2 / self.scale ** 2,
                                                             torch.einsum('xcbf,yndf->xycnbd', dX1_reshaped,
                                                                          dX2_reshaped)).permute(0, 2, 1,
                                                                                                 3).contiguous().view(
            X1.shape[0] * self.Nmask, X2.shape[0] * self.Nmask))

        del X1_expanded, X2_expanded, __k_X1X2, __k_X2X1, X1__outer_minus__X2, X2__outer_minus__X1, \
            DdD_dr1, DdD_dr2, dX1_reshaped, dX2_reshaped
        gc.collect()

        return K_X1X2

    def kernel_per_data(self, X1=None, dX1=None, X2=None, dX2=None, iter=0):
        '''
        Return a full kernel matrix between two structural fingerprints, 'X1' and 'X2'.
        X1, X2 = [Ncenter, Nfeature]
        dX1, dX2 = [Ncenter, Natom, 3, Nfeature]
        '''

        Nfeature = X1.shape[1]
        X1_expanded = X1[:, None, :]  # [Ncenter, 1, Nfeature]
        X2_expanded = X2[None, :, :]  # [1, Ncenter, Nfeature]

        X1_outer_minus_X2 = X1_expanded - X2_expanded  # [Ncenter, Ncenter, Nfeature]
        X2_outer_minus_X1 = X2_expanded - X1_expanded  # [Ncenter, Ncenter, Nfeature]

        # RBF kernel
        k = self.weight ** 2 * torch.exp(-torch.linalg.norm(X1_outer_minus_X2, dim=2) ** 2 / (2 * self.scale ** 2))

        # Create empty kernel
        kernel = torch.empty((1 + self.Nmask,
                              1 + self.Nmask), dtype=self.torch_data_type, device=self.device)

        # Evaluate the first element of kernel
        kernel[0, 0] = torch.sum(k)

        # Evaluate the first column of kernel using einsum
        intermediate_result = torch.einsum('cnf,c...af->cna',
                                           X1_outer_minus_X2,
                                           dX1.reshape(X1.shape[0],
                                                       self.Nmask,
                                                       Nfeature))
        kernel[1:, 0] = torch.einsum('cn,cna->a',
                                     k / self.scale ** 2,
                                     intermediate_result)

        # Evaluate the first row of kernel using einsum
        intermediate_result = torch.einsum('cnf,...naf->cna',
                                           X2_outer_minus_X1,
                                           dX2.reshape(X2.shape[0],
                                                       self.Nmask,
                                                       Nfeature))
        kernel[0, 1:] = torch.einsum('cn,cna->a',
                                     k / self.scale ** 2,
                                     intermediate_result)

        # [Ncenter, Nfeature] * [Ncenter, Natom * 3, Nfeature]
        # -> [Ncenter, Natom * 3]
        DdD_dr1 = torch.einsum('cnf,c...bf->cnb',
                               X1_outer_minus_X2,
                               dX1.reshape(X1.shape[0],
                                           self.Nmask,
                                           Nfeature))
        DdD_dr2 = torch.einsum('cnf,...ndf->cnd',
                               X2_outer_minus_X1,
                               dX2.reshape(X2.shape[0],
                                           self.Nmask,
                                           Nfeature))

        # [Ncenter, Natom * 3] * [Ncenter, Natom * 3]
        # -> [Ncenter, Natom * 3, Natom * 3]
        C0 = torch.einsum('cn,cnbd->bd',
                          k / self.scale ** 4,
                          torch.einsum('cnb,cnd->cnbd',
                                       DdD_dr1,
                                       DdD_dr2))

        # [Ncenter, Natom * 3, Nfeature] * [Ncenter, Natom * 3, Nfeature]
        # -> [Ncenter, Natom * 3, Natom * 3]
        intermediate_result = torch.einsum('cbf,ndf->cnbd',
                                           dX1.reshape(X1.shape[0],
                                                       self.Nmask,
                                                       Nfeature),
                                           dX2.reshape(X2.shape[0],
                                                       self.Nmask,
                                                       Nfeature))
        C1 = torch.einsum('cn,cnbd->bd',
                          k / self.scale ** 2,
                          intermediate_result)
        kernel[1:, 1:] = C0 + C1

        if iter % 10 == 9:
            del X1_expanded, X2_expanded, X1_outer_minus_X2, X2_outer_minus_X1, k, intermediate_result, \
                DdD_dr1, DdD_dr2, C0, C1
            gc.collect()

        return kernel

    def kernel_matrix_batch(self, images, batch_size=25):
        '''
        Data process type: Broadcast, Batch Processing
        Calculates C(X,X) i.e. full kernel matrix for training data.
        X = [Ntrain or Nsparse or Ntest, Ncenter, Nfeature]
        dX = [Ntrain or Nsparse or Ntest, Ncenter, Natom, 3, Nfeature]
        '''

        Ndata = len(images)
        X_N_batch = get_N_batch(Ndata, batch_size)
        X_indexes = get_batch_indexes_N_batch(Ndata, X_N_batch)

        K_XX = torch.empty((Ndata * (1 + self.Nmask),
                            Ndata * (1 + self.Nmask)), dtype=self.torch_data_type, device=self.device)

        for i in range(0, X_N_batch):
            fp_i, dfp_dr_i = self.generate_descriptor(images=images[X_indexes[i][0]:X_indexes[i][1]])

            row_indexes = torch.arange(X_indexes[i][0], X_indexes[i][1])
            row_deriv_indexes = torch.arange(Ndata + X_indexes[i][0] * self.Nmask,
                                             Ndata + X_indexes[i][1] * self.Nmask)
            selected_rows = torch.cat([row_indexes, row_deriv_indexes])

            K_XX[selected_rows[:, None], selected_rows] = self.kernel_with_deriv(X1=fp_i, dX1=dfp_dr_i,
                                                                                 X2=fp_i, dX2=dfp_dr_i)

            for j in range(i + 1, X_N_batch):
                fp_j, dfp_dr_j = self.generate_descriptor(images=images[X_indexes[j][0]:X_indexes[j][1]])

                col_indexes = torch.arange(X_indexes[j][0], X_indexes[j][1])
                col_deriv_indexes = torch.arange(Ndata + X_indexes[j][0] * self.Nmask,
                                                 Ndata + X_indexes[j][1] * self.Nmask)
                selected_cols = torch.cat([col_indexes, col_deriv_indexes])

                K_XX[selected_rows[:, None], selected_cols] = self.kernel_with_deriv(X1=fp_i, dX1=dfp_dr_i,
                                                                                     X2=fp_j, dX2=dfp_dr_j)
                K_XX[selected_cols[:, None], selected_rows] = K_XX[selected_rows[:, None], selected_cols].T

        return K_XX

    def kernel_matrix_iterative(self, images):

        Ndata = len(images)

        K_XX = torch.empty((Ndata * (1 + self.Nmask),
                            Ndata * (1 + self.Nmask)), dtype=self.torch_data_type, device=self.device)

        for i in range(0, Ndata):
            fp_i, dfp_dr_i = self.generate_descriptor_per_data(image=images[i])

            row_index = torch.tensor([i])
            row_deriv_index = torch.arange(Ndata + i * self.Nmask, Ndata + (i + 1) * self.Nmask)
            selected_rows = torch.cat([row_index, row_deriv_index])

            K_XX[selected_rows[:, None], selected_rows] = self.kernel_per_data(X1=fp_i, dX1=dfp_dr_i,
                                                                               X2=fp_i, dX2=dfp_dr_i,
                                                                               iter=i)

            for j in range(i + 1, Ndata):
                fp_j, dfp_dr_j = self.generate_descriptor_per_data(image=images[j])

                col_index = torch.tensor([j])
                col_deriv_index = torch.arange(Ndata + j * self.Nmask, Ndata + (j + 1) * self.Nmask)
                selected_cols = torch.cat([col_index, col_deriv_index])

                K_XX[selected_rows[:, None], selected_cols] = self.kernel_per_data(X1=fp_i, dX1=dfp_dr_i,
                                                                                   X2=fp_j, dX2=dfp_dr_j,
                                                                                   iter=j)
                K_XX[selected_cols[:, None], selected_rows] = K_XX[selected_rows[:, None], selected_cols].T

        return K_XX

    def kernel_matrix_per_data(self, image):
        fp_i, dfp_dr_i = self.generate_descriptor_per_data(image=image)

        K_X_i_X_i = self.kernel_per_data(X1=fp_i, dX1=dfp_dr_i,
                                         X2=fp_i, dX2=dfp_dr_i)

        return K_X_i_X_i

    def kernel_vector_batch(self, eval_images, train_images, batch_size=25):
        '''
        Data process type: Broadcast, Batch Processing
        Calculates C(x,X) i.e. the kernel matrix between fingerprint "x" and the training data fingerprints in "X".
        x = [Ntest or Nsparse, Ncenter, Nfeature]
        dx = [Ntest or Nsparse, Ncenter, Natom, 3, Nfeature]
        X = [Ntrain, Ncenter, Nfeature]
        dX = [Ntrain, Ncenter, Natom, 3, Nfeature]
        '''

        Ntest = len(eval_images)
        Ntrain = len(train_images)

        x_N_batch = get_N_batch(Ntest, batch_size)
        x_indexes = get_batch_indexes_N_batch(Ntest, x_N_batch)

        X_N_batch = get_N_batch(Ntrain, batch_size)
        X_indexes = get_batch_indexes_N_batch(Ntrain, X_N_batch)

        K_xX = torch.empty((Ntest * (1 + self.Nmask),
                            Ntrain * (1 + self.Nmask)), dtype=self.torch_data_type, device=self.device)

        for i in range(0, x_N_batch):
            fp_i, dfp_dr_i = self.generate_descriptor(images=eval_images[x_indexes[i][0]:x_indexes[i][1]])

            row_indexes = torch.arange(x_indexes[i][0], x_indexes[i][1])
            row_deriv_indexes = torch.arange(Ntest + x_indexes[i][0] * self.Nmask,
                                             Ntest + x_indexes[i][1] * self.Nmask)
            selected_rows = torch.cat([row_indexes, row_deriv_indexes])

            for j in range(0, X_N_batch):
                fp_j, dfp_dr_j = self.generate_descriptor(images=train_images[X_indexes[j][0]:X_indexes[j][1]])

                col_indexes = torch.arange(X_indexes[j][0], X_indexes[j][1])
                col_deriv_indexes = torch.arange(Ntrain + X_indexes[j][0] * self.Nmask,
                                                 Ntrain + X_indexes[j][1] * self.Nmask)
                selected_cols = torch.cat([col_indexes, col_deriv_indexes])

                K_xX[selected_rows[:, None], selected_cols] = self.kernel_with_deriv(X1=fp_i, dX1=dfp_dr_i,
                                                                                     X2=fp_j, dX2=dfp_dr_j)

        return K_xX

    def kernel_vector_iterative(self, eval_images, train_images):
        '''
        Data process type: Sequential, iterative Processing
        Calculates C(x,X) i.e. the kernel matrix between fingerprint "x" and the training data fingerprints in "X".
        x = [Ntest or Nsparse, Ncenter, Nfeature]
        dx = [Ntest or Nsparse, Ncenter, Natom, 3, Nfeature]
        X = [Ntrain, Ncenter, Nfeature]
        dX = [Ntrain, Ncenter, Natom, 3, Nfeature]
        '''

        Ntest = len(eval_images)
        Ntrain = len(train_images)

        # if dx is not None and dX is not None:
        K_xX = torch.empty((Ntest * (1 + self.Nmask),
                            Ntrain * (1 + self.Nmask)), dtype=self.torch_data_type, device=self.device)

        for i in range(0, Ntest):
            fp_i, dfp_dr_i = self.generate_descriptor_per_data(image=eval_images[i])

            row_index = torch.tensor([i])
            row_deriv_index = torch.arange(Ntest + i * self.Nmask, Ntest + (i + 1) * self.Nmask)
            selected_rows = torch.cat([row_index, row_deriv_index])

            for j in range(0, Ntrain):
                fp_j, dfp_dr_j = self.generate_descriptor_per_data(image=train_images[j])

                col_index = torch.tensor([j])
                col_deriv_index = torch.arange(Ntrain + j * self.Nmask, Ntrain + (j + 1) * self.Nmask)
                selected_cols = torch.cat([col_index, col_deriv_index])

                K_xX[selected_rows[:, None], selected_cols] = self.kernel_per_data(X1=fp_i, dX1=dfp_dr_i,
                                                                                   X2=fp_j, dX2=dfp_dr_j,
                                                                                   iter=j)

        return K_xX

    def kernel_vector_per_data(self, eval_image, train_images, batch_size=25):
        '''
        Calculates C(x,X) i.e. the kernel matrix between fingerprint "x" and the training data fingerprints in "X".
        x = [Ncenter, Nfeature]
        dx = [Ncenter, Natom, 3, Nfeature]
        X = [Ntrain, Ncenter, Nfeature]
        dX = [Ntrain, Ncenter, Natom, 3, Nfeature]
        '''

        Ntrain = len(train_images)

        X_N_batch = get_N_batch(Ntrain, batch_size)
        X_indexes = get_batch_indexes_N_batch(Ntrain, X_N_batch)

        fp_i, dfp_dr_i = self.generate_descriptor_per_data(image=eval_image)
        fp_i = fp_i.unsqueeze(0)
        dfp_dr_i = dfp_dr_i.unsqueeze(0)

        # K_xX_i = torch.empty(((1 + self.Nmask),
        # Ntrain * (1 + self.Nmask)), dtype=self.torch_data_type, device=self.device)
        K_xX_i = torch.empty((1 * (1 + self.Nmask),
                              Ntrain * (1 + self.Nmask)), dtype=self.torch_data_type, device=self.device)

        for j in range(0, X_N_batch):
            fp_j, dfp_dr_j = self.generate_descriptor(images=train_images[X_indexes[j][0]:X_indexes[j][1]])

            col_indexes = torch.arange(X_indexes[j][0], X_indexes[j][1])
            col_deriv_indexes = torch.arange(Ntrain + X_indexes[j][0] * self.Nmask,
                                             Ntrain + X_indexes[j][1] * self.Nmask)
            selected_cols = torch.cat([col_indexes, col_deriv_indexes])

            K_xX_i[:, selected_cols] = self.kernel_with_deriv(X1=fp_i, dX1=dfp_dr_i, X2=fp_j, dX2=dfp_dr_j)

        return K_xX_i

    def set_params(self, params):
        '''
        Set new (hyper)parameters for the kernel function.
        '''
        self.update(params)
        self.kerneltype.update(params)


class FPKernelNoforces(BaseKernelType):

    def __init__(self,
                 species, pbc, Natom,
                 kerneltype='sqexp',
                 params=None,
                 data_type='float64',
                 soap_param=None,
                 mace_param=None,
                 descriptor='cartesian coordinates',
                 device='cpu',
                 atoms_mask=None):
        super().__init__()
        '''
        params: dict
            Hyperparameters for the kernel type
        '''
        kerneltypes = {'sqexp': SquaredExp}
        self.device = device
        self.atoms_mask = atoms_mask

        if data_type == 'float32':
            self.data_type = 'float32'
            self.torch_data_type = torch.float32
        else:
            self.data_type = 'float64'
            self.torch_data_type = torch.float64

        self.species = species
        self.pbc = pbc
        self.Natom = Natom

        if params is None:
            params = {}

        kerneltype = kerneltypes.get(kerneltype)
        self.kerneltype = kerneltype(**params)

        self.soap_param = soap_param
        self.mace_param = mace_param
        self.descriptor = descriptor
        self.soap = None

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
                        "The 'MACE' package is required for using pre-trained MACE descriptors.\n"
                        "Please install it by running:\n\n"
                        "    pip install mace-torch\n")

                self.mace = mace_mp(model=self.mace_param.get('model'), device=self.device)

            else:
                try:
                    from mace.calculators import mace_off
                except ImportError:
                    raise ImportError(
                        "The 'MACE' package is required for using pre-trained MACE descriptors.\n"
                        "Please install it by running:\n\n"
                        "    pip install mace-torch\n")

                self.mace = mace_off(model=self.mace_param.get('model'), device=self.device)

    def kernel_without_deriv(self, X1=None, X2=None):
        '''
        Data process type: Broadcast, Batch Processing
        Return a full kernel matrix between two structural fingerprints, 'X1' and 'X2'.
        X1, X2 = [Ndata, Ncenter, Nfeature]
        where Ndata = Ntrain or Nsparse
        '''

        X1_expanded = X1[:, None, :, None, :]  # [Ntrain, 1, Ncenter, 1, Nfeature]
        X2_expanded = X2[None, :, None, :, :]  # [1, Ntrain, 1, Ncenter, Nfeature]
        X1__outer_minus__X2 = X1_expanded - X2_expanded

        k = self.weight ** 2 * torch.exp(-torch.linalg.norm(X1__outer_minus__X2, dim=4) ** 2 / (2 * self.scale ** 2))

        kernel = torch.sum(k, dim=(2, 3))

        return kernel

    def kernel_matrix(self, X=None):
        '''
        Data process type: Broadcast, Batch Processing
        Calculates C(X,X) i.e. full kernel matrix for training data.
        X = [Ntrain or Nsparse or Ntest, Ncenter, Nfeature]
        '''

        K_XX = self.kernel_without_deriv(X, X)

        return K_XX

    def kernel_vector(self, x=None, X=None):
        '''
        Data process type: Broadcast, Batch Processing
        Calculates C(x,X) i.e. the kernel matrix between fingerprint "x" and the training data fingerprints in "X".
        x = [Ntest or Nsparse, Ncenter, Nfeature]
        X = [Ntrain, Ncenter, Nfeature]
        '''

        K_xX = self.kernel_without_deriv(x, X)

        return K_xX

    def set_params(self, params):
        '''
        Set new (hyper)parameters for the kernel function.
        '''
        self.update(params)
        self.kerneltype.update(params)
