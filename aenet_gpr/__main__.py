import sys
import torch

from aenet_gpr.inout import Train, Test, Augmentation
from aenet_gpr.inout.read_input import read_train_in
from aenet_gpr.inout.io_print import *


def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    io_print_header()

    # 1. Read train input
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "train.in"
    input_param = read_train_in(input_file)
    input_param.update_soap_param()
    input_param.device = device

    # if input_param.verbose:
    #     io_input_reading(input_param)

    if len(input_param.train_file) != 0:
        io_print_title(text='Train')

        train_process = Train(input_param=input_param)
        train_process.read_reference_train_data()
        train_process.train_model()
        if input_param.train_model_save:
            train_process.train_model_save()
        if input_param.train_write:
            train_process.write_reference_train_xsf()

    if len(input_param.test_file) != 0:
        io_print_title(text='Test')

        test_process = Test(input_param=input_param)
        if len(input_param.train_file) != 0:
            test_process.load_train_model(train_data=train_process.train_data)
        else:
            test_process.load_train_model(train_data=None)
        test_process.read_reference_test_data()
        test_process.model_test_evaluation()
        if input_param.test_write:
            test_process.write_reference_test_xsf()

    if input_param.additional_write:
        io_print_title(text='Augmentation')

        augmentation_process = Augmentation(input_param=input_param)
        if len(input_param.train_file) != 0:
            augmentation_process.load_train_model(train_data=train_process.train_data)
        else:
            augmentation_process.load_train_model(train_data=None)
        augmentation_process.generate_additional_structures()
        augmentation_process.model_additional_evaluation()
        augmentation_process.write_additional_xsf()


if __name__ == "__main__":
    main()