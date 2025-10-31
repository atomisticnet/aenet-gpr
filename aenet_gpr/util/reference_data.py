import os
import glob
import copy

import numpy as np
import torch

import ase.io
from ase import Atoms

from aenet_gpr.src import gpr_iterative
from aenet_gpr.src import gpr_batch
from aenet_gpr.util.prepare_data import standard_output, inverse_standard_output, read_xsf_image


class ReferenceData(object):
    def __init__(self, structure_files: list = None,
                 file_format: str = 'xsf',
                 device='cpu',
                 descriptor='cartesian coordinates',
                 standardization=False,
                 data_type='float64',
                 data_process='batch',
                 soap_param=None,
                 mace_param=None,
                 cheb_param=None,
                 mask_constraints=True):

        self.data_process = data_process
        if data_type == 'float32':
            self.data_type = 'float32'
            self.numpy_data_type = np.float32
            self.torch_data_type = torch.float32
        else:
            self.data_type = 'float64'
            self.numpy_data_type = np.float64
            self.torch_data_type = torch.float64

        self.device = device

        self.images = []
        self.lattice = np.array([], dtype=self.numpy_data_type)  # [Ndata, 3, 3] if pbc
        self.structure = np.array([], dtype=self.numpy_data_type)  # [Ndata, Natom, 3]
        self.energy = np.array([], dtype=self.numpy_data_type)  # [Ndata]
        self.force = np.array([], dtype=self.numpy_data_type)  # [Ndata, Natom, 3]

        self.descriptor = descriptor
        self.standardization = standardization
        self.calculator = None

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

        if mace_param is None:
            self.mace_param = {'system': "materials",
                               'model': "small",
                               'delta': 1e-5,
                               'num_layers': -1,
                               'mace_n_jobs': 1}
        else:
            self.mace_param = mace_param

        if cheb_param is None:
            self.cheb_param = {'rad_order': 10,
                               'rad_cutoff': 5.0,
                               'ang_order': 6,
                               'ang_cutoff': 3.0,
                               'delta': 0.001}
        else:
            self.cheb_param = cheb_param

        self.mask_constraints = mask_constraints
        self.atoms_mask = self.create_mask()

    def read_structure_files(self, structure_files=None, file_format: str = 'xsf'):
        # check 1: force unit
        # check 2: self.images, self.energy, self.force consistency
        if file_format == 'xsf':
            structures = []
            energies = []
            forces = []
            for structure_file in structure_files:
                image, structure, energy, force = read_xsf_image(structure_file)
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

    def generate_cartesian(self, images):

        fp = []
        for image in images:
            fp.append(torch.as_tensor(image.get_positions(wrap=False).reshape(-1), dtype=self.torch_data_type).to(
                self.device))

        fp = torch.stack(fp).to(self.device)  # (Ndata, Natom*3)
        fp = fp.unsqueeze(1)  # (Ndata, 1, Natom*3)

        return fp

    def generate_cartesian_per_data(self, image):

        fp = torch.as_tensor(image.get_positions(wrap=False).reshape(-1), dtype=self.torch_data_type).to(self.device)

        fp = fp.unsqueeze(0)  # (1, Natom*3)
        fp = fp.unsqueeze(0)  # (1, 1, Natom*3)

        return fp

    def filter_similar_data(self, threshold=0.1):
        """
        Remove training data that is too close below the threshold

        Parameters:
            threshold (float): minimum descriptor distance
        """
        X = self.generate_cartesian(self.images)
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
            print(
                f"Removed images {remove_indices} from training data (near-duplicates; risk of ill-conditioned covariance).")

            keep_images = [self.images[i] for i in keep_indices]
            self.read_structure_files(structure_files=keep_images, file_format='ase')
            self.set_data()

    def config_calculator(self, prior=None, prior_update=True, kerneltype='sqexp', scale=0.4, weight=1.0, noise=1e-6,
                          noisefactor=0.5,
                          use_forces=True, sparse=None, sparse_derivative=None, autograd=False,
                          train_batch_size=25, eval_batch_size=25,
                          fit_weight=True, fit_scale=True,
                          descriptor_standardization=False):

        if self.standardization:
            Y_train_tensor = torch.as_tensor(self.energy_scale, dtype=self.torch_data_type).to(self.device)
            dY_train_tensor = torch.as_tensor(self.force_scale, dtype=self.torch_data_type).to(self.device)
        else:
            Y_train_tensor = torch.as_tensor(self.energy, dtype=self.torch_data_type).to(self.device)
            dY_train_tensor = torch.as_tensor(self.force, dtype=self.torch_data_type).to(self.device)

        if self.data_process == 'iterative':
            self.calculator = gpr_iterative.GaussianProcess(prior=prior,
                                                            prior_update=prior_update,
                                                            kerneltype=kerneltype,
                                                            scale=scale,
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
                                                            data_type=self.data_type,
                                                            device=self.device,
                                                            soap_param=self.soap_param,
                                                            mace_param=self.mace_param,
                                                            cheb_param=self.cheb_param,
                                                            descriptor=self.descriptor,
                                                            atoms_mask=self.atoms_mask)
        elif self.data_process == 'batch':
            self.calculator = gpr_batch.GaussianProcess(prior=prior,
                                                        prior_update=prior_update,
                                                        kerneltype=kerneltype,
                                                        scale=scale,
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
                                                        mace_param=self.mace_param,
                                                        cheb_param=self.cheb_param,
                                                        descriptor=self.descriptor,
                                                        descriptor_standardization=descriptor_standardization,
                                                        atoms_mask=self.atoms_mask)

        self.calculator.train_model()
        if fit_scale:
            self.fit_scale_only()
        if fit_weight:
            self.fit_weight_only(use_forces=True)

    def fit_weight_only(self, use_forces=True):
        """
        Fit weight of the kernel keeping all other hyperparameters fixed.
        Here we assume the kernel k(x,x',theta) can be factorized as:
        k = weight**2 * sqexp(x,x',other hyperparameters)
        """

        max_weight = 100.0
        prev_weight = copy.deepcopy(self.calculator.weight)

        try:
            _prior_array = self.calculator.prior.potential_batch(len(self.images), len(self.images[0]))
            _factor = torch.sqrt(torch.matmul(self.calculator.YdY.flatten() - _prior_array,
                                              self.calculator.model_vector) / _prior_array.shape[0])
            self.calculator.weight *= _factor.squeeze()
            self.calculator.hyper_params.update(dict(weight=self.calculator.weight))
            self.calculator.kernel.set_params(self.calculator.hyper_params)

            if self.calculator.weight < max_weight:
                pass
            else:
                raise ValueError(f"Weight parameter is too high ({self.calculator.weight}).")

        except Exception as e:
            print(f"{e}")
            print("Fix the weight parameter")
            self.calculator.weight = prev_weight
            self.calculator.hyper_params.update(dict(weight=self.calculator.weight))
            self.calculator.kernel.set_params(self.calculator.hyper_params)

        print('Updated weight:', self.calculator.weight.item())

        self.calculator.train_model()

        return

    def fit_scale_only(self, candidates=3, factor=2.0):
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
        current_scale = self.calculator.scale
        best_scale = current_scale
        best_logp = -torch.inf

        # Generate candidate scales logarithmically spaced around current scale
        candidate_scales = current_scale * factor ** torch.linspace(-1, 1, candidates, device=self.device)

        print('Candidate scales:', candidate_scales)
        for candidate_scale in candidate_scales:
            # Set temporary scale
            self.calculator.scale = candidate_scale
            self.calculator.hyper_params.update(dict(scale=self.calculator.scale))
            self.calculator.kernel.set_params(self.calculator.hyper_params)

            # Train GP with current candidate scale
            try:
                self.calculator.train_model()
            except Exception as e:
                print(f"{e}")
                continue

            # Evaluate marginal likelihood
            y_flat = self.calculator.YdY.flatten()
            m_ = self.calculator.prior.potential_batch(len(self.images), len(self.images[0]))
            a_ = self.calculator.model_vector

            logP = -0.5 * torch.matmul(y_flat - m_, a_) - torch.sum(torch.log(torch.diag(self.calculator.K_XX_L))) \
                   - self.calculator.Ntrain * 0.5 * torch.log(
                torch.as_tensor(2 * torch.pi, dtype=self.torch_data_type).to(self.device))

            # MAP 보정: log-normal prior centered at previous (or initial) scale
            log_l = torch.log(candidate_scale)
            mu = torch.log(torch.tensor(0.4, dtype=self.torch_data_type, device=self.device))
            sigma = torch.tensor(0.5, dtype=self.torch_data_type, device=self.device)
            log_prior = -0.5 * ((log_l - mu) / sigma) ** 2 - log_l
            lambda_prior = 1.0  # 튜닝
            logP_total = logP + lambda_prior * log_prior

            # Keep track of best scale
            if logP_total > best_logp:
                best_logp = logP_total
                best_scale = candidate_scale

        # Update to best scale found
        self.calculator.scale = best_scale
        self.calculator.hyper_params.update(dict(scale=self.calculator.scale))
        self.calculator.kernel.set_params(self.calculator.hyper_params)
        print('Updated scale:', best_scale.item())

        # Train GP with best candidate scale
        self.calculator.train_model()

        return

    def evaluation(self, get_variance=False):

        if get_variance:
            energy_gpr_scale, force_gpr_scale, unc_e_gpr, unc_f_gpr = self.calculator.eval_batch(
                eval_images=self.images,
                get_variance=get_variance)

            return energy_gpr_scale.cpu().detach().numpy(), force_gpr_scale.cpu().detach().numpy(), unc_e_gpr.cpu().detach().numpy(), unc_f_gpr.cpu().detach().numpy()

        else:
            energy_gpr_scale, force_gpr_scale, _, _ = self.calculator.eval_batch(eval_images=self.images,
                                                                                 get_variance=get_variance)

            return energy_gpr_scale.cpu().detach().numpy(), force_gpr_scale.cpu().detach().numpy(), None, None

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
            'mace_param': self.mace_param,
            'cheb_param': self.cheb_param,
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
        self.mace_param = state.get('mace_param')
        self.cheb_param = state.get('cheb_param')

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

        if self.data_process == 'iterative':
            self.calculator = gpr_iterative.GaussianProcess(images=self.images,
                                                            data_type=self.data_type,
                                                            device=self.device,
                                                            soap_param=self.soap_param,
                                                            mace_param=self.mace_param,
                                                            cheb_param=self.cheb_param,
                                                            descriptor=self.descriptor)
            self.calculator.load_data()

        elif self.data_process == 'batch':
            self.calculator = gpr_batch.GaussianProcess(images=self.images,
                                                        data_type=self.data_type,
                                                        device=self.device,
                                                        soap_param=self.soap_param,
                                                        mace_param=self.mace_param,
                                                        cheb_param=self.cheb_param,
                                                        descriptor=self.descriptor)
            self.calculator.load_data()
