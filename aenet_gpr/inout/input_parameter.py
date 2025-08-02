class InputParameters(object):
    """
    Information from the input file (train.in)
    train_file     :: [TRAININGSET] List of training set files
    test_file      :: [TESTPERCENT] List of test set files
    """

    def __init__(self):
        self.file_format       = 'xsf'
        self.device            = 'cpu'
        self.train_file        = []
        self.test_file         = []

        self.data_type = 'float64'
        self.data_process = 'batch'
        self.descriptor        = 'cartesian coordinates'
        self.standardization   = False
        self.mask_constraints  = False

        self.fit_weight        = True
        self.fit_scale         = True

        self.filter            = True
        self.filter_threshold  = 0.1

        self.soap_r_cut        = 5.0
        self.soap_n_max        = 6
        self.soap_l_max        = 4
        self.soap_sigma        = 0.5
        self.soap_rbf          = 'gto'
        self.soap_sparse       = False
        self.soap_centers      = None
        self.soap_method       = 'numerical'
        self.soap_n_jobs       = 1
        self.soap_param = {'r_cut': self.soap_r_cut,
                           'n_max': self.soap_n_max,
                           'l_max': self.soap_l_max,
                           'sigma': self.soap_sigma,
                           'rbf': self.soap_rbf,
                           'sparse': self.soap_sparse,
                           'centers': self.soap_centers,
                           'method': self.soap_method,
                           'n_jobs': self.soap_n_jobs}

        self.kerneltype        = 'sqexp'
        self.scale             = 0.4
        self.weight            = 1.0
        self.noise             = 1e-6
        self.noisefactor       = 0.5
        self.use_forces        = True
        self.sparse            = None
        self.sparse_derivative = None
        self.autograd          = False
        self.train_batch_size  = 25
        self.eval_batch_size   = 25
        self.get_variance      = True

        self.train_write       = False
        self.test_write        = False
        self.train_model_save  = False

        self.additional_write  = False
        self.disp_length       = 0.055
        self.num_copy          = 25

    def update_soap_param(self):
        self.soap_param = {'r_cut': self.soap_r_cut,
                           'n_max': self.soap_n_max,
                           'l_max': self.soap_l_max,
                           'sigma': self.soap_sigma,
                           'rbf': self.soap_rbf,
                           'sparse': self.soap_sparse,
                           'centers': self.soap_centers,
                           'method': self.soap_method,
                           'n_jobs': self.soap_n_jobs}
