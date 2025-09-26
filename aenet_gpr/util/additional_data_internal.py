import os
import copy
import glob
import numpy as np

# import chemcoord as cc
import ase.io
import torch

from aenet_gpr.util.reference_data_internal import ReferenceDataInternal
from aenet_gpr.util.prepare_data import standard_output, inverse_standard_output


class AdditionalDataInternal(object):
    def __init__(self, reference_training_data: ReferenceDataInternal, disp_length=0.055, num_copy=36):

        self.reference_training_data = reference_training_data
        self.data_type = self.reference_training_data.data_type
        self.numpy_data_type = self.reference_training_data.numpy_data_type
        self.torch_data_type = self.reference_training_data.torch_data_type
        self.device = reference_training_data.device

        self.disp_length = disp_length
        self.num_copy = num_copy

        self.species = self.reference_training_data.species
        self.pbc = self.reference_training_data.pbc
        self.num_atom = self.reference_training_data.num_atom
        self.fix_ind = self.reference_training_data.fix_ind
        self.descriptor = self.reference_training_data.descriptor

        self.mask_constraints = self.reference_training_data.mask_constraints
        self.atoms_mask = self.reference_training_data.atoms_mask

        self.additional_images = []
        self.energy_additional = np.array([], dtype=self.numpy_data_type)  # [Ndata]
        self.force_additional = np.array([], dtype=self.numpy_data_type)  # [Ndata, Natom, 3]
        self.uncertainty_additional = np.array([], dtype=self.numpy_data_type)  # [Ndata]

        self.energy_scale_additional = np.array([], dtype=self.numpy_data_type)  # [Ndata]
        self.force_scale_additional = np.array([], dtype=self.numpy_data_type)  # [Ndata, Natom, 3]

    def generate_additional_image(self):

        for i, image in enumerate(self.reference_training_data.images):
            pos_ref = image.get_positions()
            rand_disps = np.random.normal(0, self.disp_length, size=(self.num_copy, self.num_atom, 3))

            for j, rand_disp in enumerate(rand_disps):
                image_disp = copy.deepcopy(image)

                rand_disp[:, 0] = rand_disp[:, 0] - rand_disp[:, 0].mean()
                rand_disp[:, 1] = rand_disp[:, 1] - rand_disp[:, 1].mean()
                rand_disp[:, 2] = rand_disp[:, 2] - rand_disp[:, 2].mean()

                if self.reference_training_data.fix_ind is not None:
                    rand_disp[self.reference_training_data.fix_ind] = np.array([0.0, 0.0, 0.0])

                # set displaced atomic positions
                pos_new = pos_ref + rand_disp
                image_disp.set_positions(pos_new)

                self.additional_images.append(image_disp)

    def write_params(self):
        return dict(num_data=len(self.additional_images),
                    calculator=self.reference_training_data.calculator.hyper_params,
                    fix_ind=self.fix_ind,
                    pbc=self.pbc,
                    species=self.species,
                    num_atom=self.num_atom)

    def evaluation_additional(self, get_variance=False):

        self.reference_training_data.calculator.eval()
        with torch.no_grad():
            if get_variance:
                if self.reference_training_data.standardization:
                    energy_scale_additional, force_scale_additional, uncertainty_additional = self.reference_training_data.calculator(
                        eval_images=self.additional_images,
                        get_variance=get_variance)

                    self.energy_scale_additional = energy_scale_additional.cpu().detach().numpy()
                    self.force_scale_additional = force_scale_additional.cpu().detach().numpy()
                    self.uncertainty_additional = uncertainty_additional.cpu().detach().numpy()

                    self.inverse_standardize_energy_force(reference_training_energy=self.reference_training_data.energy)

                else:
                    energy_additional, force_additional, uncertainty_additional = self.reference_training_data.calculator(
                        eval_images=self.additional_images,
                        get_variance=get_variance)

                    self.energy_additional = energy_additional.cpu().detach().numpy()
                    self.force_additional = force_additional.cpu().detach().numpy()
                    self.uncertainty_additional = uncertainty_additional.cpu().detach().numpy()

            else:
                if self.reference_training_data.standardization:
                    energy_scale_additional, force_scale_additional, _ = self.reference_training_data.calculator(
                        eval_images=self.additional_images,
                        get_variance=get_variance)

                    self.energy_scale_additional = energy_scale_additional.cpu().detach().numpy()
                    self.force_scale_additional = force_scale_additional.cpu().detach().numpy()

                    self.inverse_standardize_energy_force(reference_training_energy=self.reference_training_data.energy)

                else:
                    energy_additional, force_additional, _ = self.reference_training_data.calculator(
                        eval_images=self.additional_images,
                        get_variance=get_variance)

                    self.energy_additional = energy_additional.cpu().detach().numpy()
                    self.force_additional = force_additional.cpu().detach().numpy()

    def standardize_energy_force(self, reference_training_energy):
        """
        Y = [n_systems]
        dY = [n_systems, Natom, 3]
        :return:
        """
        self.energy_scale_additional, self.force_scale_additional = standard_output(reference_training_energy,
                                                                                    self.energy_additional,
                                                                                    self.force_additional)

    def inverse_standardize_energy_force(self, reference_training_energy):
        """
        Y = [n_systems]
        dY = [n_systems, Natom, 3]
        :return:
        """
        self.energy_additional, self.force_additional = inverse_standard_output(reference_training_energy,
                                                                                self.energy_scale_additional,
                                                                                self.force_scale_additional)

    def write_additional_image_xsf(self, path):

        i = 0
        for image in self.additional_images:
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
                    tmp[4] = "%16.14f" % self.force_additional[i, j, 0]
                    tmp[5] = "%16.14f" % self.force_additional[i, j, 1]
                    tmp[6] = "%16.14f" % self.force_additional[i, j, 2]
                except:
                    tmp.append("%16.14f" % self.force_additional[i, j, 0])
                    tmp.append("%16.14f" % self.force_additional[i, j, 1])
                    tmp.append("%16.14f" % self.force_additional[i, j, 2])

                new_line = "     ".join(tmp)
                new_lines.append(new_line + "\n")

            with open(file, "w") as outfile:
                comment = "# total energy = %20.16f eV\n\n" % float(self.energy_additional[i])
                outfile.write("%s" % comment)

                for new_line in new_lines:
                    outfile.write(new_line)

