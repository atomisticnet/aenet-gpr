import os
import resource

import numpy as np
import torch
import ase

from aenet_gpr.inout.input_parameter import InputParameters
from aenet_gpr.util import ReferenceData, AdditionalData  # , ReferenceDataInternal, AdditionalDataInternal
from aenet_gpr.inout.io_print import *
from aenet_gpr.util.prepare_data import inverse_standard_output, read_xsf_image


class Train(object):
    def __init__(self, input_param: InputParameters):
        self.input_param = input_param
        self.train_data = None

    def read_reference_train_data(self):
        start = time.time()
        self.train_data = ReferenceData(structure_files=self.input_param.train_file,
                                        file_format=self.input_param.file_format,
                                        device=self.input_param.device,
                                        descriptor=self.input_param.descriptor,
                                        standardization=self.input_param.standardization,
                                        data_type=self.input_param.data_type,
                                        data_process=self.input_param.data_process,
                                        soap_param=self.input_param.soap_param,
                                        mace_param=self.input_param.mace_param,
                                        cheb_param=self.input_param.cheb_param,
                                        mask_constraints=self.input_param.mask_constraints)

        io_data_read_finalize(t=start,
                              mem_CPU=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                              mem_GPU=torch.cuda.max_memory_allocated() / 1024 ** 3,
                              flag='training',
                              energy_shape=self.train_data.energy.shape,
                              force_shape=self.train_data.force.shape)

    def train_model(self):
        start = time.time()

        threshold = self.input_param.filter_threshold

        if self.input_param.filter:
            self.train_data.filter_similar_data(threshold=threshold)

        if self.train_data.standardization:
            self.train_data.standardize_energy_force(self.train_data.energy)

        self.train_data.config_calculator(prior=self.input_param.prior,
                                          prior_update=self.input_param.prior_update,
                                          kerneltype=self.input_param.kerneltype,
                                          scale=self.input_param.scale,
                                          weight=self.input_param.weight,
                                          noise=self.input_param.noise,
                                          noisefactor=self.input_param.noisefactor,
                                          use_forces=self.input_param.use_forces,
                                          sparse=self.input_param.sparse,
                                          sparse_derivative=self.input_param.sparse_derivative,
                                          autograd=self.input_param.autograd,
                                          train_batch_size=self.input_param.train_batch_size,
                                          eval_batch_size=self.input_param.eval_batch_size,
                                          fit_weight=self.input_param.fit_weight,
                                          fit_scale=self.input_param.fit_scale,
                                          descriptor_standardization=self.input_param.descriptor_standardization)

        io_train_finalize(t=start,
                          mem_CPU=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                          mem_GPU=torch.cuda.max_memory_allocated() / 1024 ** 3,
                          data_param=self.train_data.write_params())

    def train_model_save(self):
        start = time.time()
        self.train_data.save_data()
        io_train_model_save(t=start,
                            mem_CPU=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                            mem_GPU=torch.cuda.max_memory_allocated() / 1024 ** 3)

    def write_reference_train_xsf(self):
        start = time.time()
        if not os.path.exists("train_xsf"):
            os.makedirs("train_xsf")
        self.train_data.write_image_xsf(path="train_xsf")
        io_train_write_finalize(t=start,
                                mem_CPU=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                                mem_GPU=torch.cuda.max_memory_allocated() / 1024 ** 3,
                                path="train_xsf")


class Test(object):
    def __init__(self, input_param: InputParameters):
        self.input_param = input_param
        self.train_data = None
        self.images = []
        self.energies = []
        self.forces = []

    def load_train_model(self, train_data: ReferenceData = None):
        start = time.time()
        if train_data is not None:
            self.train_data = train_data
        else:
            self.train_data = ReferenceData()
            self.train_data.load_data()
            io_train_model_load(t=start,
                                mem_CPU=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                                mem_GPU=torch.cuda.max_memory_allocated() / 1024 ** 3)

    def read_reference_test_data(self, structure_files=None, file_format: str = 'xsf'):
        if file_format == 'xsf':
            for structure_file in structure_files:
                image, structure, energy, force = read_xsf_image(structure_file)
                self.images.extend(image)

        elif file_format == 'ase':
            self.images = structure_files

        else:
            for structure_file in structure_files:
                self.images.extend(ase.io.read(structure_file, index=':', format=file_format))

        self.energies = np.array([image.get_potential_energy() for image in self.images],
                                 dtype=self.input_param.data_type)
        self.forces = np.array([image.get_forces() for image in self.images], dtype=self.input_param.data_type)

    def model_test_evaluation(self):
        start = time.time()
        energy_test_gpr, force_test_gpr, unc_e_test_gpr, unc_f_test_gpr = self.train_data.calculator.eval_batch(
            eval_images=self.images,
            get_variance=self.input_param.get_variance)

        energy_test_gpr = energy_test_gpr.cpu().detach().numpy()
        force_test_gpr = force_test_gpr.cpu().detach().numpy()
        unc_e_test_gpr = unc_e_test_gpr.cpu().detach().numpy()
        unc_f_test_gpr = unc_f_test_gpr.cpu().detach().numpy()

        if self.train_data.standardization:
            energy_test_gpr, force_test_gpr = inverse_standard_output(energy_ref=self.train_data.energy,
                                                                      scaled_energy_target=energy_test_gpr,
                                                                      scaled_force_target=force_test_gpr)

        io_test_evaluation(t=start,
                           mem_CPU=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                           mem_GPU=torch.cuda.max_memory_allocated() / 1024 ** 3)

        abs_F_test_gpr = np.linalg.norm(force_test_gpr, axis=2)
        abs_F_test = np.linalg.norm(self.forces, axis=2)

        print("GPR energy MAE (eV):", np.absolute(np.subtract(energy_test_gpr, self.energies)).mean())
        print("GPR force MAE (eV/Ang):", np.absolute(np.subtract(abs_F_test_gpr, abs_F_test)).mean())

        print("")
        print("Saving test target to [energy_test_reference.npy] and [force_test_reference.npy]")
        np.save("./energy_test_reference.npy", self.energies)
        np.save("./force_test_reference.npy", self.forces)

        if self.input_param.get_variance:
            print("Saving GPR prediction to [energy_test_gpr.npy], [force_test_gpr.npy], [unc_e_test_gpr.npy], and [unc_f_test_gpr.npy]")
            np.save("./energy_test_gpr.npy", energy_test_gpr)
            np.save("./force_test_gpr.npy", force_test_gpr)
            np.save("./unc_e_test_gpr.npy", unc_e_test_gpr)
            np.save("./unc_f_test_gpr.npy", unc_f_test_gpr)

        else:
            print("Saving GPR prediction to [energy_test_gpr.npy] and [force_test_gpr.npy]")
            np.save("./energy_test_gpr.npy", energy_test_gpr)
            np.save("./force_test_gpr.npy", force_test_gpr)
        print("")
        print("")


class Augmentation(object):
    def __init__(self, input_param: InputParameters):
        self.input_param = input_param
        self.train_data = None
        self.additional_data = None

    def load_train_model(self, train_data: ReferenceData = None):
        start = time.time()
        if train_data is not None:
            self.train_data = train_data
        else:
            self.train_data = ReferenceData()
            self.train_data.load_data()
            io_train_model_load(t=start,
                                mem_CPU=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                                mem_GPU=torch.cuda.max_memory_allocated() / 1024 ** 3)

    def generate_additional_structures(self):
        start = time.time()
        self.additional_data = AdditionalData(reference_training_data=self.train_data,
                                                  disp_length=self.input_param.disp_length,
                                                  num_copy=self.input_param.num_copy)

        self.additional_data.generate_additional_image()
        io_additional_generate_finalize(t=start,
                                        mem_CPU=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                                        mem_GPU=torch.cuda.max_memory_allocated() / 1024 ** 3,
                                        disp_length=self.additional_data.disp_length,
                                        num_copy=self.additional_data.num_copy,
                                        num_reference=len(self.train_data.images),
                                        num_additional=len(self.additional_data.additional_images))

    def model_additional_evaluation(self):
        start = time.time()
        self.additional_data.evaluation_additional(get_variance=self.input_param.get_variance)
        io_additional_evaluation(t=start,
                                 mem_CPU=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                                 mem_GPU=torch.cuda.max_memory_allocated() / 1024 ** 3,
                                 data_param=self.additional_data.write_params())

        print("")
        if self.input_param.get_variance:
            print("Saving GPR prediction to [energy_additional_gpr.npy], [force_additional_gpr.npy], [unc_e_additional_gpr.npy], and [unc_f_additional_gpr.npy]")
            np.save("./energy_additional_gpr.npy", self.additional_data.energy_additional)
            np.save("./force_additional_gpr.npy", self.additional_data.force_additional)
            np.save("./unc_e_additional_gpr.npy", self.additional_data.unc_e_additional)
            np.save("./unc_f_additional_gpr.npy", self.additional_data.unc_f_additional)

        else:
            print("Saving GPR prediction to [energy_additional_gpr.npy] and [force_additional_gpr.npy]")
            np.save("./energy_additional_gpr.npy", self.additional_data.energy_additional)
            np.save("./force_additional_gpr.npy", self.additional_data.force_additional)

    def write_additional_xsf(self):
        start = time.time()
        if not os.path.exists("additional_xsf"):
            os.makedirs("additional_xsf")
        self.additional_data.write_additional_image_xsf(path="additional_xsf")
        io_additional_write_finalize(t=start,
                                     mem_CPU=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                                     mem_GPU=torch.cuda.max_memory_allocated() / 1024 ** 3,
                                     path="additional_xsf")
