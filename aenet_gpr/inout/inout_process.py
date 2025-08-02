import os
import time
import resource

import numpy as np
import torch

from aenet_gpr.inout.input_parameter import InputParameters
from aenet_gpr.util import ReferenceData, ReferenceDataInternal, AdditionalData, AdditionalDataInternal
from aenet_gpr.inout.io_print import *
from aenet_gpr.util.prepare_data import standard_output, inverse_standard_output


class Train(object):
    def __init__(self, input_param: InputParameters):
        self.input_param = input_param
        self.train_data = None

    def read_reference_train_data(self):
        start = time.time()
        if self.input_param.descriptor == "internal":
            self.train_data = ReferenceDataInternal(structure_files=self.input_param.train_file,
                                                    file_format=self.input_param.file_format,
                                                    device=self.input_param.device,
                                                    descriptor=self.input_param.descriptor,
                                                    standardization=self.input_param.standardization,
                                                    data_type=self.input_param.data_type,
                                                    data_process=self.input_param.data_process,
                                                    soap_param=self.input_param.soap_param,
                                                    mask_constraints=self.input_param.mask_constraints)

        else:
            self.train_data = ReferenceData(structure_files=self.input_param.train_file,
                                            file_format=self.input_param.file_format,
                                            device=self.input_param.device,
                                            descriptor=self.input_param.descriptor,
                                            standardization=self.input_param.standardization,
                                            data_type=self.input_param.data_type,
                                            data_process=self.input_param.data_process,
                                            soap_param=self.input_param.soap_param,
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
        max_weight = 4.0
        while True:
            if self.input_param.filter:
                self.train_data.filter_similar_data(threshold=threshold)

            if self.train_data.standardization:
                self.train_data.standardize_energy_force(self.train_data.energy)

            try:
                if self.input_param.descriptor == "internal":
                    self.train_data.config_calculator(kerneltype=self.input_param.kerneltype,
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
                                                      fit_scale=self.input_param.fit_scale)

                else:
                    self.train_data.config_calculator(kerneltype=self.input_param.kerneltype,
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
                                                      fit_scale=self.input_param.fit_scale)

                if self.train_data.calculator.weight < max_weight:
                    break
                else:
                    raise ValueError(f"Weight parameter too high ({self.train_data.calculator.weight}).")

            except Exception as e:
                print(f"{e} Increasing threshold and retrying.")
                threshold += 0.2

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
        self.test_data = None

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

    def read_reference_test_data(self):
        start = time.time()
        if self.input_param.descriptor == "internal":
            self.test_data = ReferenceDataInternal(structure_files=self.input_param.test_file,
                                                   file_format=self.input_param.file_format,
                                                   device=self.input_param.device,
                                                   descriptor=self.input_param.descriptor,
                                                   standardization=self.input_param.standardization,
                                                   data_type=self.input_param.data_type,
                                                   data_process=self.input_param.data_process,
                                                   soap_param=self.input_param.soap_param,
                                                   mask_constraints=self.input_param.mask_constraints,
                                                   c_table=self.train_data.c_table)

        else:
            self.test_data = ReferenceData(structure_files=self.input_param.test_file,
                                           file_format=self.input_param.file_format,
                                           device=self.input_param.device,
                                           descriptor=self.input_param.descriptor,
                                           standardization=self.input_param.standardization,
                                           data_type=self.input_param.data_type,
                                           data_process=self.input_param.data_process,
                                           soap_param=self.input_param.soap_param,
                                           mask_constraints=self.input_param.mask_constraints)

        if self.test_data.energy is not None:
            test_data_energy_shape = self.test_data.energy.shape
        else:
            test_data_energy_shape = None

        if self.test_data.force is not None:
            test_data_force_shape = self.test_data.force.shape
        else:
            test_data_force_shape = None
        io_data_read_finalize(t=start,
                              mem_CPU=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                              mem_GPU=torch.cuda.max_memory_allocated() / 1024 ** 3,
                              flag='test',
                              energy_shape=test_data_energy_shape,
                              force_shape=test_data_force_shape)

    def model_test_evaluation(self):
        start = time.time()
        self.test_data.calculator = self.train_data.calculator
        energy_test_gpr, force_test_gpr, uncertainty_test_gpr = self.test_data.evaluation(
            get_variance=self.input_param.get_variance)

        if self.train_data.standardization:
            energy_test_gpr, force_test_gpr = inverse_standard_output(energy_ref=self.train_data.energy,
                                                                      scaled_energy_target=energy_test_gpr,
                                                                      scaled_force_target=force_test_gpr)
        else:
            pass

        io_test_evaluation(t=start,
                           mem_CPU=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                           mem_GPU=torch.cuda.max_memory_allocated() / 1024 ** 3,
                           data_param=self.test_data.write_params())

        if self.test_data.energy is not None:
            abs_F_test_gpr = np.linalg.norm(force_test_gpr, axis=2)
            abs_F_test = np.linalg.norm(self.test_data.force, axis=2)

            print("GPR energy MAE (eV):", np.absolute(np.subtract(energy_test_gpr, self.test_data.energy)).mean())
            print("GPR force MAE (eV/Ang):", np.absolute(np.subtract(abs_F_test_gpr, abs_F_test)).mean())
            print(
                "GPR uncertainty mean ± std: {0} ± {1}".format(uncertainty_test_gpr.mean(), uncertainty_test_gpr.std()))

            print("")
            print("Saving test target to [energy_test_reference.npy] and [force_test_reference.npy]")
            np.save("./energy_test_reference.npy", self.test_data.energy)
            np.save("./force_test_reference.npy", self.test_data.force)
        else:
            pass

        if self.input_param.get_variance:
            print(
                "Saving GPR prediction to [energy_test_gpr.npy], [force_test_gpr.npy], and [uncertainty_test_gpr.npy]")
            np.save("./energy_test_gpr.npy", energy_test_gpr)
            np.save("./force_test_gpr.npy", force_test_gpr)
            np.save("./uncertainty_test_gpr.npy", uncertainty_test_gpr)
        else:
            print("Saving GPR prediction to [energy_test_gpr.npy] and [force_test_gpr.npy]")
            np.save("./energy_test_gpr.npy", energy_test_gpr)
            np.save("./force_test_gpr.npy", force_test_gpr)
        print("")
        print("")

    def write_reference_test_xsf(self):
        start = time.time()
        if not os.path.exists("test_xsf"):
            os.makedirs("test_xsf")
        self.test_data.write_image_xsf(path="test_xsf")
        io_test_write_finalize(t=start,
                               mem_CPU=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                               mem_GPU=torch.cuda.max_memory_allocated() / 1024 ** 3,
                               path="test_data")


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
        if self.input_param.descriptor == "internal":
            self.additional_data = AdditionalDataInternal(reference_training_data=self.train_data,
                                                          disp_length=self.input_param.disp_length,
                                                          num_copy=self.input_param.num_copy)

        else:
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
            print(
                "Saving GPR prediction to [energy_additional_gpr.npy], [force_additional_gpr.npy], and [uncertainty_additional_gpr.npy]")
            np.save("./energy_additional_gpr.npy", self.additional_data.energy_additional)
            np.save("./force_additional_gpr.npy", self.additional_data.force_additional)
            np.save("./uncertainty_additional_gpr.npy", self.additional_data.uncertainty_additional)
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
