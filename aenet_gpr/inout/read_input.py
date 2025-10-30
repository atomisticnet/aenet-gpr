import glob
import ast
from aenet_gpr.inout.input_parameter import InputParameters


def read_keyword_argument_same_line(keyword, lines):
	value = None
	found = False
	for line in lines:
		read_keyword = line.split()[0].lower()

		if read_keyword == keyword:
			found = True
			value = line.split()[1]
	return value, found


def read_keyword_list_same_line(keyword, lines):
	value = None
	found = False
	for line in lines:
		read_keyword = line.split()[0].lower()

		if read_keyword == keyword:
			found = True
			idx = line.find('[')
			if idx != -1:
				list_str = line[idx:]
				value = ast.literal_eval(list_str)
			break
	return value, found


def read_train_in(infile):

	with open(infile, "r") as f:

		# Initialize InputParameters with default values
		input_param = InputParameters()

		# Remove comments from input file:
		lines = f.readlines()
		list_comments = []
		for i in range(len(lines)-1, -1, -1):
			if lines[i][0] in ["!", "#"] or len(lines[i].split()) == 0:
				list_comments.append(i)

		for i in list_comments:
			lines.pop(i)

		# List parameters:
		soap_centers, found = read_keyword_list_same_line("soap_centers", lines)
		if found:
			input_param.soap_centers = soap_centers

		# Logical parameters:
		prior_update, found = read_keyword_argument_same_line("prior_update", lines)
		if found:
			if "F" in prior_update.upper():
				input_param.prior_update = False
			else:
				input_param.prior_update = True

		standardization, found = read_keyword_argument_same_line("standardization", lines)
		if found:
			if "F" in standardization.upper():
				input_param.standardization = False
			else:
				input_param.standardization = True

		descriptor_standardization, found = read_keyword_argument_same_line("descriptor_standardization", lines)
		if found:
			if "F" in descriptor_standardization.upper():
				input_param.descriptor_standardization = False
			else:
				input_param.descriptor_standardization = True

		constraints, found = read_keyword_argument_same_line("constraints", lines)
		if found:
			if "F" in constraints.upper():
				input_param.mask_constraints = False
			else:
				input_param.mask_constraints = True

		fit_weight, found = read_keyword_argument_same_line("fit_weight", lines)
		if found:
			if "F" in fit_weight.upper():
				input_param.fit_weight = False
			else:
				input_param.fit_weight = True

		fit_scale, found = read_keyword_argument_same_line("fit_scale", lines)
		if found:
			if "F" in fit_scale.upper():
				input_param.fit_scale = False
			else:
				input_param.fit_scale = True

		use_forces, found = read_keyword_argument_same_line("force", lines)
		if found:
			if "F" in use_forces.upper():
				input_param.use_forces = False
			else:
				input_param.use_forces = True

		get_variance, found = read_keyword_argument_same_line("get_variance", lines)
		if found:
			if "F" in get_variance.upper():
				input_param.get_variance = False
			else:
				input_param.get_variance = True

		autograd, found = read_keyword_argument_same_line("autograd", lines)
		if found:
			if "F" in autograd.upper():
				input_param.autograd = False
			else:
				input_param.autograd = True

		train_write, found = read_keyword_argument_same_line("train_write", lines)
		if found:
			if "F" in train_write.upper():
				input_param.train_write = False
			else:
				input_param.train_write = True

		test_write, found = read_keyword_argument_same_line("test_write", lines)
		if found:
			if "F" in test_write.upper():
				input_param.test_write = False
			else:
				input_param.test_write = True

		train_model_save, found = read_keyword_argument_same_line("train_model_save", lines)
		if found:
			if "F" in train_model_save.upper():
				input_param.train_model_save = False
			else:
				input_param.train_model_save = True

		additional_write, found = read_keyword_argument_same_line("additional_write", lines)
		if found:
			if "F" in additional_write.upper():
				input_param.additional_write = False
			else:
				input_param.additional_write = True

		filter, found = read_keyword_argument_same_line("filter", lines)
		if found:
			if "F" in filter.upper():
				input_param.filter = False
			else:
				input_param.filter = True

		mace_invariants, found = read_keyword_argument_same_line("mace_invariants", lines)
		if found:
			if "F" in mace_invariants.upper():
				input_param.mace_invariants = False
			else:
				input_param.mace_invariants = True

		# Optional parameters:
		train_file, found = read_keyword_argument_same_line("train_file", lines)
		if found:
			input_param.train_file = glob.glob(train_file)
			input_param.train_file.sort()

		test_file, found = read_keyword_argument_same_line("test_file", lines)
		if found:
			input_param.test_file = glob.glob(test_file)
			input_param.test_file.sort()

		file_format, found = read_keyword_argument_same_line("file_format", lines)
		if found:
			input_param.file_format = str(file_format).lower()

		# data_process, found = read_keyword_argument_same_line("data_process", lines)
		# if found:
		# 	if 'bat' in data_process.lower():
		# 		input_param.data_process = 'batch'
		# 	elif 'iter' in data_process.lower():
		# 		input_param.data_process = 'iterative'
		# 	else:
		# 		raise Exception("data_process should be either batch or iterative")

		descriptor, found = read_keyword_argument_same_line("descriptor", lines)
		if found:
			if "soap" in descriptor.lower():
				input_param.descriptor = 'soap'
			elif "mace" in descriptor.lower():
				input_param.descriptor = 'mace'
			elif "cheb" in descriptor.lower():
				input_param.descriptor = 'chebyshev'
			elif "internal" in descriptor.lower():
				input_param.descriptor = 'internal'
			else:
				input_param.descriptor = 'cartesian coordinates'

		cheb_rad_cutoff, found = read_keyword_argument_same_line("cheb_rad_cutoff", lines)
		if found:
			input_param.cheb_rad_cutoff = float(cheb_rad_cutoff)

		cheb_ang_cutoff, found = read_keyword_argument_same_line("cheb_ang_cutoff", lines)
		if found:
			input_param.cheb_ang_cutoff = float(cheb_ang_cutoff)

		cheb_rad_order, found = read_keyword_argument_same_line("cheb_rad_order", lines)
		if found:
			input_param.cheb_rad_order = int(cheb_rad_order)

		cheb_ang_order, found = read_keyword_argument_same_line("cheb_ang_order", lines)
		if found:
			input_param.cheb_ang_order = int(cheb_ang_order)

		cheb_delta, found = read_keyword_argument_same_line("cheb_delta", lines)
		if found:
			input_param.cheb_delta = float(cheb_delta)

		soap_r_cut, found = read_keyword_argument_same_line("soap_r_cut", lines)
		if found:
			input_param.soap_r_cut = float(soap_r_cut)

		soap_n_max, found = read_keyword_argument_same_line("soap_n_max", lines)
		if found:
			input_param.soap_n_max = int(soap_n_max)

		soap_l_max, found = read_keyword_argument_same_line("soap_l_max", lines)
		if found:
			input_param.soap_l_max = int(soap_l_max)

		soap_sigma, found = read_keyword_argument_same_line("soap_sigma", lines)
		if found:
			input_param.soap_sigma = float(soap_sigma)

		soap_rbf, found = read_keyword_argument_same_line("soap_rbf", lines)
		if found:
			if 'poly' in soap_rbf:
				input_param.soap_rbf = 'polynomial'
			else:
				input_param.soap_rbf = 'gto'

		soap_n_jobs, found = read_keyword_argument_same_line("soap_n_jobs", lines)
		if found:
			input_param.soap_n_jobs = int(soap_n_jobs)

		mace_system, found = read_keyword_argument_same_line("mace_system", lines)
		if found:
			input_param.mace_system = mace_system

		mace_model, found = read_keyword_argument_same_line("mace_model", lines)
		if found:
			input_param.mace_model = mace_model

		mace_delta, found = read_keyword_argument_same_line("mace_delta", lines)
		if found:
			input_param.mace_delta = float(mace_delta)

		mace_num_layers, found = read_keyword_argument_same_line("mace_num_layers", lines)
		if found:
			input_param.mace_num_layers = int(mace_num_layers)

		mace_n_jobs, found = read_keyword_argument_same_line("mace_n_jobs", lines)
		if found:
			input_param.mace_n_jobs = int(mace_n_jobs)

		scale, found = read_keyword_argument_same_line("scale", lines)
		if found:
			input_param.scale = float(scale)

		weight, found = read_keyword_argument_same_line("weight", lines)
		if found:
			input_param.weight = float(weight)

		noise, found = read_keyword_argument_same_line("noise", lines)
		if found:
			input_param.noise = float(noise)

		noisefactor, found = read_keyword_argument_same_line("noisefactor", lines)
		if found:
			input_param.noisefactor = float(noisefactor)

		batch_size, found = read_keyword_argument_same_line("batch_size", lines)
		if found:
			input_param.train_batch_size = int(batch_size)
			input_param.eval_batch_size = int(batch_size)

		disp_length, found = read_keyword_argument_same_line("disp_length", lines)
		if found:
			input_param.disp_length = float(disp_length)

		num_copy, found = read_keyword_argument_same_line("num_copy", lines)
		if found:
			input_param.num_copy = int(num_copy)

		filter_threshold, found = read_keyword_argument_same_line("filter_threshold", lines)
		if found:
			input_param.filter_threshold = float(filter_threshold)

		return input_param
