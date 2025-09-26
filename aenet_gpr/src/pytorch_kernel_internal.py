import torch
import numpy as np
import os

# import chemcoord as cc
import ase.io

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


class FPKernelInternal(BaseKernelType):

    def __init__(self,
                 species, pbc, Natom,
                 kerneltype='sqexp',
                 params=None,
                 data_type='float64',
                 soap_param=None,
                 descriptor='internal',
                 device='cpu',
                 atoms_mask=None,
                 c_table=None):
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
        self.c_table = c_table

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
        self.descriptor = descriptor
        self.soap = None

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

    def generate_internal_descriptor(self, images):

        Ndata = len(images)
        Ncenter = 1
        Natom = len(images[0])
        Nfeature = Natom

        fp = {'bond': torch.empty((Ndata, Ncenter, Nfeature), dtype=self.torch_data_type, device=self.device),
              'angle': torch.empty((Ndata, Ncenter, Nfeature), dtype=self.torch_data_type, device=self.device),
              'dihedral': torch.empty((Ndata, Ncenter, Nfeature), dtype=self.torch_data_type, device=self.device), }

        dfp_dr = {'bond': torch.empty((Ndata, Ncenter, Natom, 3, Nfeature), dtype=self.torch_data_type, device=self.device),
                  'angle': torch.empty((Ndata, Ncenter, Natom, 3, Nfeature), dtype=self.torch_data_type, device=self.device),
                  'dihedral': torch.empty((Ndata, Ncenter, Natom, 3, Nfeature), dtype=self.torch_data_type, device=self.device), }

        # Set the c_table
        if self.c_table is None:
            ase.io.write("./tmp.xyz", images[0], plain=True)
            cc_atom = cc.Cartesian.read_xyz('./tmp.xyz', start_index=1)
            os.remove("./tmp.xyz")

            z_matrix = cc_atom.get_zmat()
            self.c_table = z_matrix.loc[:, ['b', 'a', 'd']]

            print("c_table has been constructed:")
            print(self.c_table)

        # Set z-matrix and z-matrix-gradient
        for i, image in enumerate(images):
            ase.io.write("./tmp_{0}.xyz".format(i), image, plain=True)
            cc_atom = cc.Cartesian.read_xyz('./tmp_{0}.xyz'.format(i), start_index=1)
            os.remove("./tmp_{0}.xyz".format(i))

            # z-matrix
            zmat = cc_atom.get_zmat(self.c_table)

            fp.get('bond')[i, 0, :] = torch.as_tensor(np.array(zmat.safe_loc[:, 'bond']),
                                                      dtype=self.torch_data_type).to(self.device)
            fp.get('angle')[i, 0, :] = torch.as_tensor(np.array(zmat.safe_loc[:, 'angle']),
                                                       dtype=self.torch_data_type).to(self.device)
            fp.get('dihedral')[i, 0, :] = torch.as_tensor(np.array(zmat.safe_loc[:, 'dihedral']),
                                                          dtype=self.torch_data_type).to(self.device)

            # z-matrix-gradient
            zmat_grad = cc_atom.loc[self.c_table.index].get_grad_zmat(self.c_table).args[0]
            zmat_grad = torch.as_tensor(zmat_grad, dtype=self.torch_data_type).to(self.device)

            dfp_dr.get('bond')[i, 0, :, :, :] = zmat_grad[0].permute(1, 2, 0)  # 0, 2, 1
            dfp_dr.get('angle')[i, 0, :, :, :] = zmat_grad[1].permute(1, 2, 0)
            dfp_dr.get('dihedral')[i, 0, :, :, :] = zmat_grad[2].permute(1, 2, 0)

        return fp, dfp_dr

    def generate_internal_descriptor_per_data(self, image):

        Ncenter = 1
        Natom = len(image)
        Nfeature = Natom

        fp = {'bond': torch.empty((Ncenter, Nfeature), dtype=self.torch_data_type, device=self.device),
              'angle': torch.empty((Ncenter, Nfeature), dtype=self.torch_data_type, device=self.device),
              'dihedral': torch.empty((Ncenter, Nfeature), dtype=self.torch_data_type, device=self.device), }

        dfp_dr = {'bond': torch.empty((Ncenter, Natom, 3, Nfeature), dtype=self.torch_data_type, device=self.device),
                  'angle': torch.empty((Ncenter, Natom, 3, Nfeature), dtype=self.torch_data_type, device=self.device),
                  'dihedral': torch.empty((Ncenter, Natom, 3, Nfeature), dtype=self.torch_data_type, device=self.device), }

        # Set the c_table
        if self.c_table is None:
            ase.io.write("./tmp.xyz", image, plain=True)
            cc_atom = cc.Cartesian.read_xyz('./tmp.xyz', start_index=1)

            z_matrix = cc_atom.get_zmat()
            self.c_table = z_matrix.loc[:, ['b', 'a', 'd']]

        cc_atom = cc.Cartesian.read_xyz('./tmp.xyz', start_index=1)
        os.remove("./tmp.xyz")

        # z-matrix
        zmat = cc_atom.get_zmat(self.c_table)

        fp.get('bond')[0, :] = torch.as_tensor(np.array(zmat.safe_loc[:, 'bond']),
                                               dtype=self.torch_data_type).to(self.device)
        fp.get('angle')[0, :] = torch.as_tensor(np.array(zmat.safe_loc[:, 'angle']),
                                                dtype=self.torch_data_type).to(self.device)
        fp.get('dihedral')[0, :] = torch.as_tensor(np.array(zmat.safe_loc[:, 'dihedral']),
                                                   dtype=self.torch_data_type).to(self.device)

        # z-matrix-gradient
        zmat_grad = cc_atom.loc[self.c_table.index].get_grad_zmat(self.c_table).args[0]
        zmat_grad = torch.as_tensor(zmat_grad, dtype=self.torch_data_type).to(self.device)

        dfp_dr.get('bond')[0, :, :, :] = zmat_grad[0].permute(1, 2, 0)  # 0, 2, 1
        dfp_dr.get('angle')[0, :, :, :] = zmat_grad[1].permute(1, 2, 0)
        dfp_dr.get('dihedral')[0, :, :, :] = zmat_grad[2].permute(1, 2, 0)

        return fp, dfp_dr

    def kernel_with_deriv(self, X1=None, dX1=None, X2=None, dX2=None):
        '''
        Return a full kernel matrix between two structural fingerprints, 'X1' and 'X2'.
        X1, X2 = [Ndata, Ncenter, Nfeature]
        dX1, dX2 = [Ndata, Ncenter, Natom, 3, Nfeature]

        where Ndata = Nbatch, Ncenter=1, Nfeature=Natom
        '''

        X1_b = X1.get('bond')
        X1_a = X1.get('angle') * torch.pi / 180  # unit: degree -> radian
        X1_d = X1.get('dihedral') * torch.pi / 180  # unit: degree -> radian
        dX1_b = dX1.get('bond')
        dX1_a = dX1.get('angle')  # unit: radian
        dX1_d = dX1.get('dihedral')  # unit: radian

        X2_b = X2.get('bond')
        X2_a = X2.get('angle') * torch.pi / 180  # unit: degree -> radian
        X2_d = X2.get('dihedral') * torch.pi / 180  # unit: degree -> radian
        dX2_b = dX2.get('bond')
        dX2_a = dX2.get('angle')  # unit: radian
        dX2_d = dX2.get('dihedral')  # unit: radian

        scale_b = self.scale.get('bond')
        scale_a = self.scale.get('angle')
        scale_d = self.scale.get('dihedral')

        # pairwise_distances example
        # mean_bond, std_bond, max_bond, min_bond = self.pairwise_distances(X1_b)
        # mean_angle_sin, std_angle_sin, max_angle_sin, min_angle_sin = self.pairwise_distances(X1_a, sin_transform=True)
        # mean_dihedral_sin, std_dihedral_sin, max_dihedral_sin, min_dihedral_sin = self.pairwise_distances(X1_d, sin_transform=True)
        #
        # print("Bond distance - mean:", mean_bond, "std:", std_bond, "max:", max_bond, "min:", min_bond)
        # print("Angle distance (sin) - mean:", mean_angle_sin, "std:", std_angle_sin, "max:", max_angle_sin, "min:", min_angle_sin)
        # print("Dihedral distance (sin) - mean:", mean_dihedral_sin, "std:", std_dihedral_sin, "max:", max_dihedral_sin, "min:", min_dihedral_sin)

        # Create empty kernel
        K_X1X2 = torch.empty(((1 + self.Nmask) * X1_b.shape[0],
                              (1 + self.Nmask) * X2_b.shape[0]), dtype=self.torch_data_type, device=self.device)

        # Expand value fingerprint
        X1_b_expanded = X1_b[:, None, :, None, :]  # [Ndata1, 1, Ncenter, 1, Nfeature]
        X2_b_expanded = X2_b[None, :, None, :, :]  # [1, Ndata2, 1, Ncenter, Nfeature]
        X1_b__outer_minus__X2_b = X1_b_expanded - X2_b_expanded  # [Ndata1, Ndata2, Ncenter, Ncenter, Nfeature]
        X2_b__outer_minus__X1_b = X2_b_expanded - X1_b_expanded  # [Ndata1, Ndata2, Ncenter, Ncenter, Nfeature]

        X1_a_expanded = X1_a[:, None, :, None, :]  # [Ndata1, 1, Ncenter, 1, Nfeature]
        X2_a_expanded = X2_a[None, :, None, :, :]  # [1, Ndata2, 1, Ncenter, Nfeature]
        X1_a__outer_minus__X2_a = X1_a_expanded - X2_a_expanded  # [Ndata1, Ndata2, Ncenter, Ncenter, Nfeature]
        X2_a__outer_minus__X1_a = X2_a_expanded - X1_a_expanded  # [Ndata1, Ndata2, Ncenter, Ncenter, Nfeature]

        X1_d_expanded = X1_d[:, None, :, None, :]  # [Ndata1, 1, Ncenter, 1, Nfeature]
        X2_d_expanded = X2_d[None, :, None, :, :]  # [1, Ndata2, 1, Ncenter, Nfeature]
        X1_d__outer_minus__X2_d = X1_d_expanded - X2_d_expanded  # [Ndata1, Ndata2, Ncenter, Ncenter, Nfeature]
        X2_d__outer_minus__X1_d = X2_d_expanded - X1_d_expanded  # [Ndata1, Ndata2, Ncenter, Ncenter, Nfeature]

        # [Ndata, Ndata, Ncenter, Ncenter]
        __k_X1X2_b = torch.exp(-torch.linalg.norm(X1_b__outer_minus__X2_b, dim=4) ** 2 / (2 * scale_b ** 2))
        __k_X2X1_b = torch.exp(-torch.linalg.norm(X2_b__outer_minus__X1_b, dim=4) ** 2 / (2 * scale_b ** 2))

        __k_X1X2_a = torch.exp(-torch.linalg.norm(torch.sin(X1_a__outer_minus__X2_a / 2), dim=4) ** 2 / (2 * scale_a ** 2))
        __k_X2X1_a = torch.exp(-torch.linalg.norm(torch.sin(X2_a__outer_minus__X1_a / 2), dim=4) ** 2 / (2 * scale_a ** 2))

        __k_X1X2_d = torch.exp(-torch.linalg.norm(torch.sin(X1_d__outer_minus__X2_d / 2), dim=4) ** 2 / (2 * scale_d ** 2))
        __k_X2X1_d = torch.exp(-torch.linalg.norm(torch.sin(X2_d__outer_minus__X1_d / 2), dim=4) ** 2 / (2 * scale_d ** 2))

        """"K_X1X2[:X1.shape[0], :X2.shape[0]] kernel between fp1 and fp2"""
        # [Ndata, Ndata, Ncenter, Ncenter] -> [Ndata, Ndata]
        K_X1X2[:X1_b.shape[0], :X2_b.shape[0]] = torch.sum(self.weight ** 2 * __k_X1X2_b * __k_X1X2_a * __k_X1X2_d,
                                                           dim=(2, 3))

        """K_X1X2[X1.shape[0]: X1.shape[0] * (1 + 3 * Natom), :X2.shape[0]] kernel between fp1_deriv and fp2"""
        # [Ndata, Ncenter, Natom, 3, Nfeature] -> [Ndata, Ncenter, Natom * 3, Nfeature]
        dX1_b_reshaped = dX1_b.flatten(start_dim=2, end_dim=3)
        dX1_a_reshaped = dX1_a.flatten(start_dim=2, end_dim=3)
        dX1_d_reshaped = dX1_d.flatten(start_dim=2, end_dim=3)

        intermediate_result_1 = torch.einsum('xycnf,xcbf->xycnb', X1_b__outer_minus__X2_b / scale_b ** 2, dX1_b_reshaped)
        intermediate_result_1 += 1 / 4 * torch.einsum('xycnf,xcbf->xycnb', torch.sin(X1_a__outer_minus__X2_a) / scale_a ** 2, dX1_a_reshaped)
        intermediate_result_1 += 1 / 4 * torch.einsum('xycnf,xcbf->xycnb', torch.sin(X1_d__outer_minus__X2_d) / scale_d ** 2, dX1_d_reshaped)

        K_X1X2[X1_b.shape[0]:, :X2_b.shape[0]].copy_(torch.einsum('xycn,xycnb->xyb', self.weight ** 2 * __k_X1X2_b * __k_X1X2_a * __k_X1X2_d,
                                                                  intermediate_result_1).permute(0, 2, 1).contiguous().view(X1_b.shape[0] * self.Nmask, X2_b.shape[0]))

        """K_X1X2[:X1.shape[0], X2.shape[0]: X2.shape[0] * (1 + 3 * Natom)] kernel between fp1 and fp2_deriv"""
        # [Ndata, Ncenter, Natom, 3, Nfeature] -> [Ndata, Ncenter, Natom * 3, Nfeature]
        dX2_b_reshaped = dX2_b.flatten(start_dim=2, end_dim=3)
        dX2_a_reshaped = dX2_a.flatten(start_dim=2, end_dim=3)
        dX2_d_reshaped = dX2_d.flatten(start_dim=2, end_dim=3)

        intermediate_result_2 = torch.einsum('xycnf,yndf->xycnd', X2_b__outer_minus__X1_b / scale_b ** 2, dX2_b_reshaped)
        intermediate_result_2 += 1 / 4 * torch.einsum('xycnf,yndf->xycnd', torch.sin(X2_a__outer_minus__X1_a) / scale_a ** 2, dX2_a_reshaped)
        intermediate_result_2 += 1 / 4 * torch.einsum('xycnf,yndf->xycnd', torch.sin(X2_d__outer_minus__X1_d) / scale_d ** 2, dX2_d_reshaped)

        K_X1X2[:X1_b.shape[0], X2_b.shape[0]:].copy_(torch.einsum('xycn,xycnd->xyd', self.weight ** 2 * __k_X1X2_b * __k_X1X2_a * __k_X1X2_d,
                                                                  intermediate_result_2).view(X1_b.shape[0], X2_b.shape[0] * self.Nmask))

        """K_X1X2[X1.shape[0]: X1.shape[0] * (1 + 3 * Natom), X2.shape[0]: X2.shape[0] * (1 + 3 * Natom)] 
        kernel between fp1_deriv and fp2_deriv"""
        # size sign in eisum:
        # x, y: Ndata
        # c, n: Ncenter
        # b, d: Natom *3
        # f, f: Nfeature

        K_X1X2[X1_b.shape[0]:, X2_b.shape[0]:].copy_(torch.einsum('xycn,xycnbd->xybd', self.weight ** 2 * __k_X1X2_b * __k_X1X2_a * __k_X1X2_d,
                                                                  torch.einsum('xycnb,xycnd->xycnbd',
                                                                               intermediate_result_1,
                                                                               intermediate_result_2)).permute(0, 2, 1, 3).contiguous().view(X1_b.shape[0] * self.Nmask, X2_b.shape[0] * self.Nmask))

        intermediate_result_3 = torch.einsum('xcbf,yndf->xycnbd', dX1_b_reshaped / scale_b ** 2, dX2_b_reshaped)
        intermediate_result_3 += 1 / 4 * torch.einsum('xcbf,xycnf,yndf->xycnbd', dX1_a_reshaped / scale_a ** 2, torch.cos(X2_a__outer_minus__X1_a), dX2_a_reshaped)
        intermediate_result_3 += 1 / 4 * torch.einsum('xcbf,xycnf,yndf->xycnbd', dX1_d_reshaped / scale_d ** 2, torch.cos(X2_d__outer_minus__X1_d), dX2_d_reshaped)

        K_X1X2[X1_b.shape[0]:, X2_b.shape[0]:].add_(torch.einsum('xycn,xycnbd->xybd', self.weight ** 2 * __k_X1X2_b * __k_X1X2_a * __k_X1X2_d,
                                                                 intermediate_result_3).permute(0, 2, 1, 3).contiguous().view(X1_b.shape[0] * self.Nmask, X2_b.shape[0] * self.Nmask))

        return K_X1X2

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
            fp_i, dfp_dr_i = self.generate_internal_descriptor(images=images[X_indexes[i][0]:X_indexes[i][1]])

            row_indexes = torch.arange(X_indexes[i][0], X_indexes[i][1])
            row_deriv_indexes = torch.arange(Ndata + X_indexes[i][0] * self.Nmask,
                                             Ndata + X_indexes[i][1] * self.Nmask)
            selected_rows = torch.cat([row_indexes, row_deriv_indexes])

            K_XX[selected_rows[:, None], selected_rows] = self.kernel_with_deriv(X1=fp_i, dX1=dfp_dr_i,
                                                                                 X2=fp_i, dX2=dfp_dr_i)

            for j in range(i + 1, X_N_batch):
                fp_j, dfp_dr_j = self.generate_internal_descriptor(images=images[X_indexes[j][0]:X_indexes[j][1]])

                col_indexes = torch.arange(X_indexes[j][0], X_indexes[j][1])
                col_deriv_indexes = torch.arange(Ndata + X_indexes[j][0] * self.Nmask,
                                                 Ndata + X_indexes[j][1] * self.Nmask)
                selected_cols = torch.cat([col_indexes, col_deriv_indexes])

                K_XX[selected_rows[:, None], selected_cols] = self.kernel_with_deriv(X1=fp_i, dX1=dfp_dr_i,
                                                                                     X2=fp_j, dX2=dfp_dr_j)
                K_XX[selected_cols[:, None], selected_rows] = K_XX[selected_rows[:, None], selected_cols].T

        return K_XX

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
            fp_i, dfp_dr_i = self.generate_internal_descriptor(images=eval_images[x_indexes[i][0]:x_indexes[i][1]])

            row_indexes = torch.arange(x_indexes[i][0], x_indexes[i][1])
            row_deriv_indexes = torch.arange(Ntest + x_indexes[i][0] * self.Nmask,
                                             Ntest + x_indexes[i][1] * self.Nmask)
            selected_rows = torch.cat([row_indexes, row_deriv_indexes])

            for j in range(0, X_N_batch):
                fp_j, dfp_dr_j = self.generate_internal_descriptor(images=train_images[X_indexes[j][0]:X_indexes[j][1]])

                col_indexes = torch.arange(X_indexes[j][0], X_indexes[j][1])
                col_deriv_indexes = torch.arange(Ntrain + X_indexes[j][0] * self.Nmask,
                                                 Ntrain + X_indexes[j][1] * self.Nmask)
                selected_cols = torch.cat([col_indexes, col_deriv_indexes])

                K_xX[selected_rows[:, None], selected_cols] = self.kernel_with_deriv(X1=fp_i, dX1=dfp_dr_i,
                                                                                     X2=fp_j, dX2=dfp_dr_j)

        return K_xX

    def set_params(self, params):
        '''
        Set new (hyper)parameters for the kernel function.
        '''
        self.update(params)
        self.kerneltype.update(params)
