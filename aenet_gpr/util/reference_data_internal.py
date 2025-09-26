import os
import glob
import copy

import numpy as np
import torch

# import chemcoord as cc
import ase.io
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from aenet_gpr.src import gpr_iterative
from aenet_gpr.src.gpr_batch_internal import GaussianProcessInternal
from aenet_gpr.util.prepare_data import standard_output, inverse_standard_output


class ReferenceDataInternal(object):
    def __init__(self, structure_files: list = None,
                 file_format: str = 'xsf',
                 device='cpu',
                 descriptor='internal',
                 standardization=False,
                 data_type='float64',
                 data_process='batch',
                 soap_param=None,
                 mask_constraints=False,
                 c_table=None):

        self.data_process = data_process
        if data_type == 'float32':
            self.data_type = 'float32'
            self.numpy_data_type = np.float32
            self.torch_data_type = torch.float32
        else:
            self.data_type = 'float64'
            self.numpy_data_type = np.float64
            self.torch_data_type = torch.float64

        if device == 'cpu':
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.images = []
        self.lattice = np.array([], dtype=self.numpy_data_type)  # [Ndata, 3, 3] if pbc
        self.structure = np.array([], dtype=self.numpy_data_type)  # [Ndata, Natom, 3]
        self.energy = np.array([], dtype=self.numpy_data_type)  # [Ndata]
        self.force = np.array([], dtype=self.numpy_data_type)  # [Ndata, Natom, 3]

        self.descriptor = descriptor
        self.standardization = standardization
        self.calculator = None
        self.c_table = c_table

        self.fix_ind = None
        self.pbc = None
        self.species = None
        self.num_atom = None

        if structure_files is not None:
            self.read_structure_files(structure_files, file_format)
        self.set_data()

        self.energy_scale = np.array([], dtype=self.numpy_data_type)  # [Ndata]
        self.force_scale = np.array([], dtype=self.numpy_data_type)  # [Ndata, Natom, 3]

        if soap_param is None:
            self.soap_param = {'r_cut': 5.0,
                               'n_max': 6,
                               'l_max': 4,
                               'sigma': 0.5,
                               'rbf': 'gto',
                               'sparse': False,
                               'centers': None,
                               'method': 'numerical',
                               'n_jobs': 1}
        else:
            self.soap_param = soap_param

        self.mask_constraints = mask_constraints
        if self.mask_constraints:
            self.atoms_mask = self.create_mask()
        else:
            self.atoms_mask = None

        # self.dfp_dr = np.array([], dtype=self.numpy_data_type)  # [Ndata, Ncenter, Natom, 3, Nfeature]
        # self.fp = np.array([], dtype=self.numpy_data_type)  # [Ndata, Ncenter, Nfeature]

    def read_structure_files(self, structure_files=None, file_format: str = 'xsf'):
        # check 1: force unit
        # check 2: self.images, self.energy, self.force consistency
        if file_format == 'xsf':
            structures = []
            energies = []
            forces = []
            for structure_file in structure_files:
                image, structure, energy, force = self.read_xsf_image(structure_file)
                self.images.extend(image)
                structures.append(structure)
                energies.append(energy)
                forces.append(force)

            self.structure = np.asarray(structures, dtype=self.numpy_data_type)

            if any(energy is None for energy in energies):
                self.energy = None
            else:
                self.energy = np.asarray(energies, dtype=self.numpy_data_type)

            if any(force is None for force in forces):
                self.force = None
            else:
                self.force = np.asarray(forces, dtype=self.numpy_data_type)

        elif file_format == 'ase':
            self.images = structure_files
            self.structure = np.asarray([image.get_positions() for image in self.images], dtype=self.numpy_data_type)
            self.energy = np.asarray([image.get_potential_energy() for image in self.images],
                                     dtype=self.numpy_data_type)
            self.force = np.asarray([image.get_forces() for image in self.images], dtype=self.numpy_data_type)

        else:
            for structure_file in structure_files:
                self.images.extend(ase.io.read(structure_file, index=':', format=file_format))

            self.structure = np.asarray([image.get_positions() for image in self.images], dtype=self.numpy_data_type)
            try:
                self.energy = np.asarray([image.get_potential_energy() for image in self.images],
                                         dtype=self.numpy_data_type)
            except:
                self.energy = None

            try:
                self.force = np.asarray([image.get_forces() for image in self.images], dtype=self.numpy_data_type)
            except:
                self.force = None

    def write_params(self):
        return dict(num_data=len(self.images),
                    calculator=self.calculator.hyper_params,
                    fix_ind=self.fix_ind,
                    pbc=self.pbc,
                    species=self.species,
                    num_atom=self.num_atom)

    def set_data(self):
        self.species = self.images[0].get_chemical_symbols()
        lattice_constants = []
        if np.all(self.images[0].get_pbc()):
            self.pbc = True
            lattice_constants.append(self.images[0].get_cell())

            for i, image in enumerate(self.images[1:]):
                species_ = image.get_chemical_symbols()
                assert species_ == self.species, "Chemical elements are not homogeneous with {0} data".format(i + 1)

                pbc_ = np.all(image.get_pbc())
                assert pbc_ == self.pbc, "Periodic boundary condition error: different pbc files are contained"

                lattice_constants.append(image.get_cell())

            self.lattice = np.asarray(lattice_constants, dtype=self.numpy_data_type)

        else:
            self.pbc = False

            for i, image in enumerate(self.images[1:]):
                species_ = image.get_chemical_symbols()
                assert species_ == self.species, "Chemical elements are not homogeneous with {0} data".format(i + 1)

                pbc_ = np.all(image.get_pbc())
                assert pbc_ == self.pbc, "Periodic boundary condition error: different pbc files are contained"

        self.num_atom = len(self.species)

    def create_mask(self):
        """
        This function mask atoms coordinates that will not participate in the
        model, i.e. the coordinates of the atoms that are kept fixed
        or constraint.
        """
        atoms = self.images[0]
        constraints = atoms.constraints
        mask_constraints = torch.ones_like(torch.tensor(atoms.positions), dtype=torch.bool)
        for i in range(0, len(constraints)):
            try:
                mask_constraints[constraints[i].a] = ~constraints[i].mask
            except Exception:
                pass

            try:
                mask_constraints[constraints[i].index] = False
            except Exception:
                pass

            try:
                mask_constraints[constraints[0].a] = ~constraints[0].mask
            except Exception:
                pass

            try:
                mask_constraints[constraints[-1].a] = ~constraints[-1].mask
            except Exception:
                pass
        return torch.argwhere(mask_constraints.reshape(-1)).reshape(-1)

    def set_fix_index_by_coord(self, max, axis=2):
        pos = self.images[0].get_positions()
        self.fix_ind = np.where(pos[:, axis] < max)

    def set_fix_index(self, fix_ind):
        self.fix_ind = fix_ind

    def generate_internal(self, images):

        Ndata = len(images)
        Ncenter = 3
        Natom = len(images[0])
        Nfeature = Natom

        # (Ndata, 3, Natom)
        fp = torch.empty((Ndata, Ncenter, Nfeature), dtype=self.torch_data_type, device=self.device)

        # Set the c_table
        if self.c_table is None:
            ase.io.write("./tmp.xyz", images[0], plain=True)
            cc_atom = cc.Cartesian.read_xyz('./tmp.xyz', start_index=1)
            os.remove("./tmp.xyz")

            z_matrix = cc_atom.get_zmat()
            self.c_table = z_matrix.loc[:, ['b', 'a', 'd']]

            print("c_table has been constructed:")
            print(self.c_table)

        for i, image in enumerate(images):
            ase.io.write("./tmp_{0}.xyz".format(i), image, plain=True)
            cc_atom = cc.Cartesian.read_xyz('./tmp_{0}.xyz'.format(i), start_index=1)
            os.remove("./tmp_{0}.xyz".format(i))

            # z-matrix
            zmat = cc_atom.get_zmat(self.c_table)
            fp[i, 0, :] = torch.as_tensor(np.array(zmat.safe_loc[:, 'bond']), dtype=self.torch_data_type).to(self.device)
            fp[i, 1, :] = torch.as_tensor(np.array(zmat.safe_loc[:, 'angle']), dtype=self.torch_data_type).to(self.device)
            fp[i, 2, :] = torch.as_tensor(np.array(zmat.safe_loc[:, 'dihedral']), dtype=self.torch_data_type).to(self.device)

        return fp

    def filter_similar_data(self, threshold=0.1):
        """
        Remove training data that is too close below the threshold

        Parameters:
            threshold (float): minimum descriptor distance
        """
        X = self.generate_internal(self.images)
        N = X.shape[0]

        keep_indices = []
        remove_indices = []

        for i in range(N):
            xi = X[i].flatten()
            keep = True
            for idx in keep_indices:
                xj = X[idx].flatten()
                dist = torch.linalg.norm(xi - xj)
                if dist < threshold:
                    keep = False
                    break
            if keep:
                keep_indices.append(i)
            else:
                remove_indices.append(i)

        if remove_indices != []:
            print("images", remove_indices, "are removed from training data since there are too close images")
            print(", which can lead to ill-conditioned covariance")
            keep_images = [self.images[i] for i in keep_indices]
            self.read_structure_files(structure_files=keep_images, file_format='ase')
            self.set_data()

    def config_calculator(self, kerneltype='sqexp', scale_b=0.4, scale_a=0.1, scale_d=0.1, weight=1.0, noise=1e-6, noisefactor=0.5,
                          use_forces=True, sparse=None, sparse_derivative=None, autograd=False,
                          train_batch_size=25, eval_batch_size=25,
                          fit_weight=True, fit_scale=False):

        # X_train_tensor = torch.as_tensor(self.fp, dtype=self.torch_data_type).to(self.device)
        # dX_train_tensor = torch.as_tensor(self.dfp_dr, dtype=self.torch_data_type).to(self.device)

        if self.standardization:
            Y_train_tensor = torch.as_tensor(self.energy_scale, dtype=self.torch_data_type).to(self.device)
            dY_train_tensor = torch.as_tensor(self.force_scale, dtype=self.torch_data_type).to(self.device)
        else:
            Y_train_tensor = torch.as_tensor(self.energy, dtype=self.torch_data_type).to(self.device)
            dY_train_tensor = torch.as_tensor(self.force, dtype=self.torch_data_type).to(self.device)

        if self.data_process == 'batch':
            self.calculator = GaussianProcessInternal(kerneltype=kerneltype,
                                                      scale_b=scale_b, scale_a=scale_a, scale_d=scale_d,
                                                      weight=weight,
                                                      noise=noise,
                                                      noisefactor=noisefactor,
                                                      use_forces=use_forces,
                                                      images=self.images,
                                                      function=Y_train_tensor,
                                                      derivative=dY_train_tensor,
                                                      sparse=sparse,
                                                      sparse_derivative=sparse_derivative,
                                                      autograd=autograd,
                                                      train_batch_size=train_batch_size,
                                                      eval_batch_size=eval_batch_size,
                                                      data_type=self.data_type,
                                                      device=self.device,
                                                      soap_param=self.soap_param,
                                                      descriptor=self.descriptor,
                                                      atoms_mask=self.atoms_mask,
                                                      c_table=self.c_table)

        self.calculator.train_model()
        if fit_scale:
            self.fit_scale_only(candidates=5)
        if fit_weight:
            self.fit_weight_only(use_forces=True)

    def fit_weight_only(self, use_forces=True):
        """
        Fit weight of the kernel keeping all other hyperparameters fixed.
        Here we assume the kernel k(x,x',theta) can be factorized as:
        k = weight**2 * sqexp(x,x',other hyperparameters)
        """

        if use_forces:
            _prior_array = self.calculator.prior.potential_batch(self.images, get_forces=True)
            _factor = torch.sqrt(torch.matmul(self.calculator.YdY.flatten() - _prior_array,
                                              self.calculator.model_vector) / _prior_array.shape[0])
            self.calculator.weight *= _factor.squeeze()
            self.calculator.hyper_params.update(dict(weight=self.calculator.weight))
            self.calculator.kernel.set_params(self.calculator.hyper_params)

        else:
            _prior_array = self.calculator.prior.potential_batch(self.images, get_forces=False)
            _n_train = len(self.images)
            _factor = torch.sqrt(torch.matmul(self.calculator.YdY.flatten()[:_n_train] - _prior_array,
                                              self.calculator.model_vector[:_n_train]) / _prior_array.shape[0])
            self.calculator.weight *= _factor.squeeze()
            self.calculator.hyper_params.update(dict(weight=self.calculator.weight))
            self.calculator.kernel.set_params(self.calculator.hyper_params)

        print('Updated weight:', self.calculator.weight)

        self.calculator.train_model()

        return

    def fit_scale_only(self, candidates=5, factor=4.0):
        """
        Update the kernel scale keeping all other hyperparameters fixed by evaluating marginal likelihood over candidate scales.

        Parameters:
        X: ndarray (training inputs)
        Y: ndarray (training targets)
        candidates: int (number of candidate scales)
        factor: float (multiplicative factor to generate candidate scales)

        Returns:
        Updated scale value
        """

        for key in self.calculator.scale.keys():
            current_scale = self.calculator.scale.get(key)
            best_scale = current_scale
            best_logp = -torch.inf

            # Generate candidate scales logarithmically spaced around current scale
            candidate_scales = current_scale * factor ** torch.linspace(-1, 1, candidates)

            print(f'Candidate scales({key}):', candidate_scales)
            for candidate_scale in candidate_scales:
                # Set temporary scale
                self.calculator.scale.update({key: candidate_scale})
                self.calculator.hyper_params.update(dict(scale=self.calculator.scale))
                self.calculator.kernel.set_params(self.calculator.hyper_params)

                # Train GP with current candidate scale
                self.calculator.train_model()

                # Evaluate marginal likelihood
                y_flat = self.calculator.YdY.flatten()
                m_ = self.calculator.prior.potential_batch(self.images, get_forces=True)
                a_ = self.calculator.model_vector

                logP = -0.5 * torch.matmul(y_flat - m_, a_) - torch.sum(torch.log(torch.diag(self.calculator.K_XX_L))) \
                       - self.calculator.Ntrain * 0.5 * torch.log(torch.as_tensor(2 * torch.pi, dtype=self.torch_data_type).to(self.device))

                # Keep track of best scale
                if logP > best_logp:
                    best_logp = logP
                    best_scale = candidate_scale

            # Update to best scale found
            self.calculator.scale.update({key: best_scale})
            self.calculator.hyper_params.update(dict(scale=self.calculator.scale))
            self.calculator.kernel.set_params(self.calculator.hyper_params)
            print(f'Updated scale({key}):', best_scale)

            # Train GP with best candidate scale
            self.calculator.train_model()

        print('Updated scale:', self.calculator.scale)

        return

    def evaluation(self, get_variance=False):

        self.calculator.eval()
        with torch.no_grad():
            if get_variance:
                energy_gpr_scale, force_gpr_scale, uncertainty_gpr = self.calculator(eval_images=self.images,
                                                                                     get_variance=get_variance)

                return energy_gpr_scale.cpu().detach().numpy(), force_gpr_scale.cpu().detach().numpy(), uncertainty_gpr.cpu().detach().numpy()

            else:
                energy_gpr_scale, force_gpr_scale, _ = self.calculator(eval_images=self.images,
                                                                       get_variance=get_variance)

                return energy_gpr_scale.cpu().detach().numpy(), force_gpr_scale.cpu().detach().numpy(), None

    def standardize_energy_force(self, reference_training_energy):
        """
        Y = [n_systems]
        dY = [n_systems, Natom, 3]
        :return:
        """
        self.energy_scale, self.force_scale = standard_output(reference_training_energy,
                                                              self.energy,
                                                              self.force)

    def inverse_standardize_energy_force(self, reference_training_energy):
        """
        Y = [n_systems]
        dY = [n_systems, Natom, 3]
        :return:
        """
        self.energy, self.force = inverse_standard_output(reference_training_energy,
                                                          self.energy_scale,
                                                          self.force_scale)

    def read_xsf_image(self, path):

        image = ase.io.read(path, index=':', format='xsf')
        with open(path, 'r') as infile:
            lines = infile.readlines()

        structure = np.empty((len(image[0]), 3), dtype=self.numpy_data_type)

        try:
            energy = np.asarray(lines[0].split()[4], dtype=self.numpy_data_type)
        except IndexError:
            energy = None
        except ValueError:
            energy = None

        force = np.empty((len(image[0]), 3), dtype=self.numpy_data_type)
        if "ATOMS" in lines[2]:
            for i, line in enumerate(lines[3:]):
                structure[i, :] = np.asarray(line.split()[1:4], dtype=self.numpy_data_type)
                try:
                    force[i, :] = np.asarray(line.split()[4:], dtype=self.numpy_data_type)
                except:
                    force = None

        elif "CRYSTAL" in lines[2]:
            for i, line in enumerate(lines[9:]):
                structure[i, :] = np.asarray(line.split()[1:4], dtype=self.numpy_data_type)
                try:
                    force[i, :] = np.asarray(line.split()[4:], dtype=self.numpy_data_type)
                except:
                    force = None

        image[0].calc = SinglePointCalculator(image[0].copy(), energy=energy, forces=force)

        return image, structure, energy, force

    def write_image_xsf(self, path):

        i = 0
        for image in self.images:
            ase.io.write(os.path.join(path, "file_{0:0>5}.xsf".format(i)), image, format="xsf")
            i = i + 1

        files = glob.glob(os.path.join(path, "file_*.xsf"))
        files.sort()

        for i, file in enumerate(files):
            with open(file, 'r') as infile:
                lines = infile.readlines()

            if self.pbc:
                new_lines = lines[:7]
                del lines[:7]
            else:
                new_lines = [lines[0]]
                del lines[0]

            for j, line in enumerate(lines):
                tmp = line.split()
                tmp[0] = self.species[j]

                try:
                    tmp[4] = "%16.14f" % self.force[i, j, 0]
                    tmp[5] = "%16.14f" % self.force[i, j, 1]
                    tmp[6] = "%16.14f" % self.force[i, j, 2]
                except:
                    tmp.append("%16.14f" % self.force[i, j, 0])
                    tmp.append("%16.14f" % self.force[i, j, 1])
                    tmp.append("%16.14f" % self.force[i, j, 2])

                new_line = "     ".join(tmp)
                new_lines.append(new_line + "\n")

            with open(file, "w") as outfile:
                comment = "# total energy = %20.16f eV\n\n" % float(self.energy[i])
                outfile.write("%s" % comment)

                for new_line in new_lines:
                    outfile.write(new_line)

    def save_data(self, file="data_dict.pt"):
        """
        self.data_type
        self.numpy_data_type
        self.torch_data_type

        self.images -> self.structures (numpy)
        self.energy (numpy)
        self.force (numpy)

        self.device
        self.descriptor
        self.standardization
        self.calculator -> calculator.save

        self.fix_ind
        self.pbc
        self.species
        self.num_atom

        self.soap -> self.soap.parameters
        """

        state = {
            'data_type': self.data_type,
            'data_process': self.data_process,
            'numpy_data_type': self.numpy_data_type,
            'torch_data_type': self.torch_data_type,
            'lattice': torch.as_tensor(self.lattice),
            'structure': torch.as_tensor(self.structure),
            'energy': torch.as_tensor(self.energy),
            'force': torch.as_tensor(self.force),
            'device': self.device,
            'descriptor': self.descriptor,
            'standardization': self.standardization,
            'fix_ind': self.fix_ind,
            'pbc': self.pbc,
            'species': self.species,
            'num_atom': self.num_atom,
            'soap_param': self.soap_param,
        }
        torch.save(state, file)
        self.calculator.save_data()

    def load_data(self, file="data_dict.pt"):
        state = torch.load(file)

        self.data_process = state.get('data_process')
        self.data_type = state.get('data_type')
        self.numpy_data_type = state.get('numpy_data_type')
        self.torch_data_type = state.get('torch_data_type')

        self.images = []
        self.lattice = state.get('lattice').cpu().detach().numpy()
        self.structure = state.get('structure').cpu().detach().numpy()
        self.energy = state.get('energy').cpu().detach().numpy()
        self.force = state.get('force').cpu().detach().numpy()

        self.device = state.get('device')
        self.descriptor = state.get('descriptor')
        self.standardization = state.get('standardization')

        self.fix_ind = state.get('fix_ind')
        self.pbc = state.get('pbc')
        self.species = state.get('species')
        self.num_atom = state.get('num_atom')

        self.soap_param = state.get('soap_param')

        if self.pbc:
            atom = Atoms(self.species)
            atom.set_pbc(self.pbc)
            for i, pos in enumerate(self.structure):
                atom.set_positions(pos)
                atom.set_cell(self.lattice[i])
                self.images.append(copy.deepcopy(atom))

        else:
            atom = Atoms(self.species)
            atom.set_pbc(self.pbc)
            for i, pos in enumerate(self.structure):
                atom.set_positions(pos)
                self.images.append(copy.deepcopy(atom))

        if self.data_process == 'batch':
            self.calculator = GaussianProcessInternal(images=self.images,
                                                      data_type=self.data_type,
                                                      device=self.device,
                                                      soap_param=self.soap_param,
                                                      descriptor=self.descriptor)
            self.calculator.load_data()
