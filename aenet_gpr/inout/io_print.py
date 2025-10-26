import datetime
import time


def io_print(text):
    print(text, flush=True)


def io_current_time():
    aux = str(datetime.datetime.now())[:-6]
    io_print(aux)


def io_data_read_finalize(t, mem_CPU, mem_GPU, flag, energy_shape, force_shape):

    io_print("")
    io_print("Read reference {0} data".format(flag))
    io_print("")
    io_print("Time needed for reading data:  {: 18.6f} s".format(time.time() - t))
    io_print("Maximum CPU memory used:   {: 18.6f} GB".format(mem_CPU))
    io_print("Maximum GPU memory used:   {: 18.6f} GB".format(mem_GPU))
    print("Energy data size:", energy_shape, " # (N_data, )")
    print("Force data size:", force_shape, " # (N_data, N_atom, 3)")
    io_line()


def io_set_descriptor_finalize(t, mem_CPU, mem_GPU, flag, descriptor, fp_shape, dfp_dr_shape):

    io_line()
    io_print("Generate {0} descriptors".format(flag))
    io_print("")
    io_print("Time needed for setting descriptor:  {: 18.6f} s".format(time.time() - t))
    io_print("Maximum CPU memory used:   {: 18.6f} GB".format(mem_CPU))
    io_print("Maximum GPU memory used:   {: 18.6f} GB".format(mem_GPU))
    print("Descriptor:", descriptor)
    print("Descriptor size:", fp_shape, " # (N_data, N_center, N_feature)")
    print("Descriptor derivatives size:", dfp_dr_shape, " # (N_data, N_center, N_atom, 3, N_feature)")
    io_line()


def io_train_finalize(t, mem_CPU, mem_GPU, data_param: dict):

    io_line()
    io_print("Model train")
    io_train_parameters(data_param=data_param)
    io_print("Time needed for training:  {: 18.6f} s".format(time.time() - t))
    io_print("Maximum CPU memory used:   {: 18.6f} GB".format(mem_CPU))
    io_print("Maximum GPU memory used:   {: 18.6f} GB".format(mem_GPU))
    io_line()


def io_train_parameters(data_param: dict):

    io_print("")
    io_print("Training parameters")
    io_print(data_param)
    io_print("")


def io_train_model_save(t, mem_CPU, mem_GPU):

    io_line()
    io_print("Trained model save")
    io_print("")
    io_print("Saving training data to [data_dict.pt]")
    io_print("Saving GPR model parameters to [calc_dict.pt]")
    io_print("")
    io_print("Time needed for saving:  {: 18.6f} s".format(time.time() - t))
    io_print("Maximum CPU memory used:   {: 18.6f} GB".format(mem_CPU))
    io_print("Maximum GPU memory used:   {: 18.6f} GB".format(mem_GPU))
    io_double_line()
    io_print("")
    io_print("")


def io_train_model_load(t, mem_CPU, mem_GPU):

    io_double_line()
    io_print("Model load")
    io_print("")
    io_print("Loading Training data from [data_dict.pt]")
    io_print("Loading GPR model parameters from [calc_dict.pt]")
    io_print("")
    io_print("Time needed for loading:  {: 18.6f} s".format(time.time() - t))
    io_print("Maximum CPU memory used:   {: 18.6f} GB".format(mem_CPU))
    io_print("Maximum GPU memory used:   {: 18.6f} GB".format(mem_GPU))
    io_double_line()


def io_test_evaluation(t, mem_CPU, mem_GPU):

    io_line()
    io_print("Model evaluation for test set")
    io_print("Time needed for test evaluation:  {: 18.6f} s".format(time.time() - t))
    io_print("Maximum CPU memory used:   {: 18.6f} GB".format(mem_CPU))
    io_print("Maximum GPU memory used:   {: 18.6f} GB".format(mem_GPU))
    io_line()


def io_test_parameters(data_param: dict):
    io_print("")
    io_print("Test parameters")
    io_print(data_param)
    io_print("")


def io_additional_generate_finalize(t, mem_CPU, mem_GPU, disp_length, num_copy, num_reference, num_additional):

    io_line()
    io_print("Additional structure generation")
    io_print("")
    io_print("Time needed for additional generation:  {: 18.6f} s".format(time.time() - t))
    io_print("Maximum CPU memory used:   {: 18.6f} GB".format(mem_CPU))
    io_print("Maximum GPU memory used:   {: 18.6f} GB".format(mem_GPU))
    print("Displacement length (Ang):", disp_length)
    print("Multiple:", num_copy)
    print("N_additional = N_reference ({0}) * Multiple ({1}) = {2}".format(num_reference, num_copy, num_additional))
    io_line()


def io_additional_evaluation(t, mem_CPU, mem_GPU, data_param: dict):
    io_line()
    io_print("Model evaluation for additional set")
    io_test_parameters(data_param=data_param)
    io_print("Time needed for additional evaluation:  {: 18.6f} s".format(time.time() - t))
    io_print("Maximum CPU memory used:   {: 18.6f} GB".format(mem_CPU))
    io_print("Maximum GPU memory used:   {: 18.6f} GB".format(mem_GPU))
    io_line()


def io_additional_parameters(data_param: dict):
    io_print("")
    io_print("Augmentation parameters")
    io_print(data_param)
    io_print("")


def io_train_write_finalize(t, mem_CPU, mem_GPU, path):

    io_double_line()
    io_print("Writing train xsf files to {0}".format(path))
    io_print("")
    io_print("Time needed for writing xsf files:  {: 18.6f} s".format(time.time() - t))
    io_print("Maximum CPU memory used:   {: 18.6f} GB".format(mem_CPU))
    io_print("Maximum GPU memory used:   {: 18.6f} GB".format(mem_GPU))
    io_double_line()


def io_test_write_finalize(t, mem_CPU, mem_GPU, path):

    io_double_line()
    io_print("Writing test xsf files to {0}".format(path))
    io_print("")
    io_print("Time needed for writing xsf files:  {: 18.6f} s".format(time.time() - t))
    io_print("Maximum CPU memory used:   {: 18.6f} GB".format(mem_CPU))
    io_print("Maximum GPU memory used:   {: 18.6f} GB".format(mem_GPU))
    io_double_line()


def io_additional_write_finalize(t, mem_CPU, mem_GPU, path):

    io_double_line()
    io_print("Writing additional xsf files to {0}".format(path))
    io_print("")
    io_print("Time needed for writing xsf files:  {: 18.6f} s".format(time.time() - t))
    io_print("Maximum CPU memory used:   {: 18.6f} GB".format(mem_CPU))
    io_print("Maximum GPU memory used:   {: 18.6f} GB".format(mem_GPU))
    io_double_line()


def io_line():
    io_print("----------------------------------------------------------------------")


def io_print_title(text):
    io_double_line()
    io_print(text)
    io_double_line()


def io_double_line():
    io_print("======================================================================")


def io_print_header(version):
    io_double_line()
    io_print("aenet-GPR: Gaussian Process Regression Surrogate Models for Accelerating Data Generation")
    io_print("version {0}".format(version))
    io_double_line()
    io_print("")
    io_current_time()
    io_print("")
    io_print("Developed by In Won Yeu")
    io_print("")
    io_print("This program performs three main steps:")
    io_print("1. Train: Generates a GPR model using the provided structure, energy, and force data.")
    io_print("2. Test: Uses the generated GPR model to predict values for the test set structures.")
    io_print("3. Augmentation: Performs data augmentation in xsf file format, compatible with aenet-(PyTorch),")
    io_print("")
    # io_print("Each of these steps is executed once the input file (train.in) contains the keywords:")
    # io_print("Train_file [train file path]")
    # io_print("Test_file [test file path]")
    # io_print("Additional_write [True]")
    # io_print("Once the Train step is completed, the generated GPR model is saved in [data_dict.pt] and [calc_dict.pt].")
    # io_print("Using these saved model files, you can later run only the Test or Augmentation steps separately.")
    # io_print("This program is distributed in the hope that it will be useful,")
    # io_print("but WITHOUT ANY WARRANTY; without even the implied warranty of")
    # io_print("MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the")
    # io_print("GNU General Public License in file 'LICENSE' for more details.")
    io_print("")
    io_print("")

