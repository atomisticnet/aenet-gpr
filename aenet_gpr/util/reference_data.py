import os
import glob
import copy

import numpy as np
import torch

import ase.io
from ase import Atoms

from aenet_gpr.src import gpr_iterative
from aenet_gpr.src import gpr_batch
from aenet_gpr.util.prepare_data import standard_output, inverse_standard_output


class ReferenceData(object):
    def __init__(self, structure_files: list = None,
                 file_format: str = 'xsf',
                 device='cpu',
                 descriptor='cartesian coordinates',
                 standardization=True,
                 data_type='float64',
                 data_process='batch',
                 soap_param=None):

        self.data_process = data_process
        if data_type == 'float32':
            self.data_type = 'float32'
            self.numpy_data_type = np.float32
            self.torch_data_type = torch.float32
        else:
            self.data_type = 'float64'
            self.numpy_data_type = np.float64
            self.torch_data_type = torch.float64

        self.images = []
        self.lattice = np.array([], dtype=self.numpy_data_type)  # [Ndata, 3, 3] if pbc
        self.structure = np.array([], dtype=self.numpy_data_type)  # [Ndata, Natom, 3]
        self.energy = np.array([], dtype=self.numpy_data_type)  # [Ndata]
        self.force = np.array([], dtype=self.numpy_data_type)  # [Ndata, Natom, 3]
        if structure_files is not None:
            self.read_structure_files(structure_files, file_format)

        self.device = device
        self.descriptor = descriptor
        self.standardization = standardization
        self.calculator = None

        self.fix_ind = None
        self.pbc = None
        self.species = None
        self.num_atom = None

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

    def set_fix_index_by_coord(self, max, axis=2):
        pos = self.images[0].get_positions()
        self.fix_ind = np.where(pos[:, axis] < max)

    def set_fix_index(self, fix_ind):
        self.fix_ind = fix_ind

    def config_calculator(self, kerneltype='sqexp', scale=0.4, weight=1.0, noise=1e-6, noisefactor=0.5,
                          use_forces=True, sparse=None, sparse_derivative=None, autograd=False,
                          train_batch_size=25, eval_batch_size=25):

        # X_train_tensor = torch.as_tensor(self.fp, dtype=self.torch_data_type).to(self.device)
        # dX_train_tensor = torch.as_tensor(self.dfp_dr, dtype=self.torch_data_type).to(self.device)

        if self.standardization:
            Y_train_tensor = torch.as_tensor(self.energy_scale, dtype=self.torch_data_type).to(self.device)
            dY_train_tensor = torch.as_tensor(self.force_scale, dtype=self.torch_data_type).to(self.device)
        else:
            Y_train_tensor = torch.as_tensor(self.energy, dtype=self.torch_data_type).to(self.device)
            dY_train_tensor = torch.as_tensor(self.force, dtype=self.torch_data_type).to(self.device)

        if self.data_process == 'iterative':
            self.calculator = gpr_iterative.GaussianProcess(kerneltype=kerneltype,
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
                                                            descriptor=self.descriptor)
        elif self.data_process == 'batch':
            self.calculator = gpr_batch.GaussianProcess(kerneltype=kerneltype,
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
                                                        descriptor=self.descriptor)

        self.calculator.train_model()

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

        if self.data_process == 'iterative':
            self.calculator = gpr_iterative.GaussianProcess(images=self.images,
                                                            data_type=self.data_type,
                                                            device=self.device,
                                                            soap_param=self.soap_param,
                                                            descriptor=self.descriptor)
            self.calculator.load_data()

        elif self.data_process == 'batch':
            self.calculator = gpr_batch.GaussianProcess(images=self.images,
                                                        data_type=self.data_type,
                                                        device=self.device,
                                                        soap_param=self.soap_param,
                                                        descriptor=self.descriptor)
            self.calculator.load_data()
