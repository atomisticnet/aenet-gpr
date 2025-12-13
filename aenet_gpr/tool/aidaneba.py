import numpy as np
import copy
import time

import torch
import ase.io
from ase.atoms import Atoms
from ase.optimize import FIRE, MDMin, LBFGS, BFGS
from ase.parallel import parprint, parallel_function

try:
    from ase.mep import NEB, DyNEB
except ModuleNotFoundError:
    from ase.neb import NEB, DyNEB

from aenet_gpr.src import GPRCalculator
from aenet_gpr.util import ReferenceData
from aenet_gpr.tool import acquisition, dump_observation, get_fmax
from aenet_gpr.inout.input_parameter import InputParameters


def is_duplicate_position(is_pos, train_image_positions):
    # is_pos: 1D numpy array (Eg, shape = (3N,))
    # train_image_positions: list of 1D numpy arrays
    for pos in train_image_positions:
        if np.array_equal(is_pos, pos):
            return True
    return False


class AIDANEBA:

    def __init__(self, start, end, input_param: InputParameters, model_calculator=None, calculator=None,
                 interpolation='idpp', n_images=5, n_train_images=3, n_flags=3, step_cutoff=15, k=None, mic=False,
                 neb_method='improvedtangent',
                 remove_rotation_and_translation=False, force_consistent=None,
                 trajectory='AIDNEB.traj',
                 use_previous_observations=False):

        """
        Artificial Intelligence-Driven Nudged Elastic Band (AID-NEB) algorithm.
        Optimize a NEB using a surrogate GPR model [1-3].
        Potential energies and forces at a given position are
        supplied to the model calculator to build a modelled PES in an
        active-learning fashion. This surrogate relies on NEB theory to
        optimize the images along the path in the predicted PES. Once the
        predicted NEB is optimized, the acquisition function collect a new
        observation based on the predicted energies and uncertainties of the
        optimized images. Gaussian Process Regression, aenet-gpr, is used to
        build the model as implemented in [4].

        [1] J. A. Garrido Torres, P. C. Jennings, M. H. Hansen,
        J. R. Boes, and T. Bligaard, Phys. Rev. Lett. 122, 156001 (2019).
        https://doi.org/10.1103/PhysRevLett.122.156001
        [2] O. Koistinen, F. B. Dagbjartsdóttir, V. Ásgeirsson, A. Vehtari,
        and H. Jónsson, J. Chem. Phys. 147, 152720 (2017).
        https://doi.org/10.1063/1.4986787
        [3] E. Garijo del Río, J. J. Mortensen, and K. W. Jacobsen,
        Phys. Rev. B 100, 104103 (2019).
        https://doi.org/10.1103/PhysRevB.100.104103
        [4] I. W. Yeu, A. Stuke, J. López-Zorrilla, J. M Stevenson, D. R Reichman, R. A Friesner,
        A. Urban, and N. Artrith, npj Computational Materials 11, 156 (2025).
        https://doi.org/10.1038/s41524-025-01651-0

        NEB Parameters
        --------------
        initial: Trajectory file (in ASE format) or Atoms object.
            Initial end-point of the NEB path.

        final: Trajectory file (in ASE format) or Atoms object.
            Final end-point of the NEB path.

        model_calculator: Model object.
            Model calculator to be used for predicting the potential energy
            surface. The default is None which uses a GP model with the Squared
            Exponential Kernel and other default parameters. See
            *ase.optimize.activelearning.gp.calculator* GPModel for default GP
            parameters.

        interpolation: string or Atoms list or Trajectory
            NEB interpolation.

            options:
                - 'linear' linear interpolation.
                - 'idpp'  image dependent pair potential interpolation.
                - Trajectory file (in ASE format) or list of Atoms.
                The user can also supply a manual interpolation by passing
                the name of the trajectory file  or a list of Atoms (ASE
                format) containing the interpolation images.

        mic: boolean
            Use mic=True to use the Minimum Image Convention and calculate the
            interpolation considering periodic boundary conditions.

        n_images: int or float
            Number of images of the path. Only applicable if 'linear' or
            'idpp' interpolation has been chosen.
            options:
                - int: Number of images describing the NEB. The number of
                images include the two (initial and final) end-points of the
                NEB path.
                - float: Spacing of the images along the NEB. The number of
                images is calculated as the length of the interpolated
                initial path divided by the spacing (Ang^-1).

        k: float or list
            Spring constant(s) in eV/Angstrom.

        neb_method: string
            NEB method as implemented in ASE. ('aseneb', 'improvedtangent'
            or 'eb'). See https://wiki.fysik.dtu.dk/ase/ase/neb.html.

        calculator: ASE calculator Object.
            ASE calculator.
            See https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K). By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back on force_consistent=False if not.

        trajectory: string
            Filename to store the predicted NEB paths.
                Additional information:
                - Energy uncertain: The energy uncertainty in each image
                position can be accessed in image.info['uncertainty'].

        use_previous_observations: boolean
            If False. The optimization starts from scratch.
            A *trajectory_observations.traj* file is automatically generated
            in each step of the optimization, which contains the
            observations collected by the surrogate. If
            (a) *use_previous_observations* is True and (b) a previous
            *trajectory_observations.traj* file is found in the working
            directory: the algorithm will be use the previous observations
            to train the model with all the information collected in
            *trajectory_observations.traj*.

        max_train_data: int
            Number of observations that will effectively be included in the
            model. See also *max_data_strategy*.

        max_train_data_strategy: string
            Strategy to decide the observations that will be included in the
            model.

            options:
                'last_observations': selects the last observations collected by
                the surrogate.
                'lowest_energy': selects the lowest energy observations
                collected by the surrogate.
                'nearest_observations': selects the observations which
                positions are nearest to the positions of the Atoms to test.

            For instance, if *max_train_data* is set to 50 and
            *max_train_data_strategy* to 'lowest energy', the surrogate model
            will be built in each iteration with the 50 lowest energy
            observations collected so far.

        """

        # Convert Atoms and list of Atoms to trajectory files.
        if isinstance(start, Atoms):
            ase.io.write('initial.traj', start)
            start = 'initial.traj'
        if isinstance(end, Atoms):
            ase.io.write('final.traj', end)
            end = 'final.traj'

        interp_path = None
        if interpolation != 'idpp' and interpolation != 'linear':
            interp_path = interpolation
        if isinstance(interp_path, list):
            ase.io.write('initial_path.traj', interp_path)
            interp_path = 'initial_path.traj'

        # NEB parameters.
        self.input_param = input_param
        self.start = start
        self.end = end
        self.n_images = n_images
        self.n_train_images = n_train_images
        self.n_flags = n_flags
        self.step_cutoff = step_cutoff
        self.level = 1
        self.mic = mic
        self.rrt = remove_rotation_and_translation
        self.neb_method = neb_method
        self.spring = k
        self.i_endpoint = ase.io.read(self.start, '-1')
        self.e_endpoint = ase.io.read(self.end, '-1')

        # GP calculator:
        self.model_calculator = model_calculator

        # Active Learning setup (Single-point calculations).
        self.step = 0
        self.function_calls = 0
        self.force_calls = 0
        self.ase_calc = calculator
        self.atoms = ase.io.read(self.start, '-1')

        self.constraints = self.atoms.constraints
        self.force_consistent = force_consistent
        self.use_previous_observations = use_previous_observations
        self.trajectory = trajectory

        # Make sure that the initial and endpoints are near the interpolation.
        if self.mic:
            mic_initial = self.i_endpoint[:]
            mic_final = self.e_endpoint[:]
            mic_images = [mic_initial]
            for i in range(10000):
                mic_images += [mic_initial.copy()]
            mic_images += [mic_final]
            neb_mic = NEB(mic_images, climb=False, method=self.neb_method, remove_rotation_and_translation=self.rrt)
            neb_mic.interpolate(method='linear', mic=self.mic)
            self.i_endpoint.positions = mic_images[1].positions[:]
            self.e_endpoint.positions = mic_images[-2].positions[:]

        # Calculate the initial and final end-points (if necessary).
        if self.i_endpoint.calc is None:
            self.i_endpoint.calc = copy.deepcopy(self.ase_calc)
        if self.e_endpoint.calc is None:
            self.e_endpoint.calc = copy.deepcopy(self.ase_calc)
        self.very_i_endpoint_energy = self.i_endpoint.get_potential_energy(force_consistent=force_consistent)
        self.i_endpoint.get_forces()
        self.very_e_endpoint_energy = self.e_endpoint.get_potential_energy(force_consistent=force_consistent)
        self.e_endpoint.get_forces()

        if isinstance(self.i_endpoint, Atoms):
            ase.io.write(f'initial_level{self.level:02d}.traj', self.i_endpoint)
        if isinstance(self.e_endpoint, Atoms):
            ase.io.write(f'final_level{self.level:02d}.traj', self.e_endpoint)

        # A) Create images using interpolation if user does define a path.
        if interp_path is None:
            self.images = make_neb(self)

            neb_interpolation = NEB(self.images, climb=False, method=self.neb_method,
                                    remove_rotation_and_translation=self.rrt)
            neb_interpolation.interpolate(method='linear', mic=self.mic)
            if interpolation == 'idpp':
                neb_interpolation = NEB(self.images, climb=True, method=self.neb_method,
                                        remove_rotation_and_translation=self.rrt)
                neb_interpolation.interpolate(method='idpp', mic=self.mic)

        # B) Alternatively, the user can propose an initial path.
        if interp_path is not None:
            images_path = ase.io.read(interp_path, ':')
            first_image = images_path[0].get_positions().reshape(-1)
            last_image = images_path[-1].get_positions().reshape(-1)

            is_pos = self.i_endpoint.get_positions().reshape(-1)
            fs_pos = self.e_endpoint.get_positions().reshape(-1)

            if not np.array_equal(first_image, is_pos):
                images_path.insert(0, self.i_endpoint)
            if not np.array_equal(last_image, fs_pos):
                images_path.append(self.e_endpoint)

            self.n_images = len(images_path)
            self.images = make_neb(self, images_interpolation=images_path)

        # Guess spring constant (k) if not defined by the user.
        self.total_path_length = 0.0
        for i in range(len(self.images) - 1):
            pos1 = self.images[i].positions.flatten()
            pos2 = self.images[i + 1].positions.flatten()
            distance = np.linalg.norm(pos2 - pos1)
            self.total_path_length += distance

        if self.spring is None:
            self.spring = 0.1 * (self.n_images - 1) / self.total_path_length

        # Save initial interpolation.
        self.initial_interpolation = self.images[:]

        filename_extxyz = f'initial_interpolation_level{self.level:02d}.extxyz'
        filename_traj = f'initial_interpolation_level{self.level:02d}.traj'
        ase.io.write(filename_extxyz, self.initial_interpolation, format='extxyz')
        ase.io.write(filename_traj, self.initial_interpolation)

        print()
        print('Total path length (Å): ', self.total_path_length)
        print('Spring constant (eV/Å): ', self.spring)

    def save_neb_predictions_to_extxyz(self, predictions, image_neb_force, filename):
        """
        Store NEB predictions (energy, force, unc_energy, unc_force)
        into Atoms objects and write to extxyz file.

        Parameters:
            predictions (dict): Dictionary with 'energy', 'forces', 'unc_energy', 'unc_forces'
            filename (str): Output file path (.extxyz)
        """
        out = []
        for i, image in enumerate(self.images):
            atoms = image.copy()

            # Energies
            atoms.info['energy'] = float(predictions['energy'][i])
            atoms.info['unc_energy'] = float(predictions['unc_energy'][i])

            # Forces
            atoms.arrays['pes_forces'] = np.linalg.norm(predictions['forces'][i], axis=1)
            atoms.arrays['neb_forces'] = np.linalg.norm(image_neb_force[i], axis=1)
            atoms.arrays['unc_forces'] = np.linalg.norm(predictions['unc_forces'][i], axis=1)

            out.append(atoms)

        # Write to extxyz file
        ase.io.write(filename, out, format='extxyz')

    def run(self,
            fmax=0.05,
            unc_convergence=0.05,
            dt=0.05,
            ml_steps=150,
            optimizer="MDMin",
            update_step=1,
            uncertainty='force',
            check_ref_force=False,
            climbing=False):

        """
        Executing run will start the NEB optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angstrom).

        unc_convergence: float
            Maximum uncertainty for convergence (in eV). The algorithm's
            convergence criteria will not be satisfied if the uncertainty
            on any of the NEB images in the predicted path is above this
            threshold.

        dt : float
            dt parameter for MDMin.

        ml_steps: int
            Maximum number of steps for the NEB optimization on the
            modelled potential energy surface.

        max_unc_trheshold: float
            Safe control parameter. This parameter controls the degree of
            freedom of the NEB optimization in the modelled potential energy
            surface or the. If the uncertainty of the NEB lies above the
            'max_unc_trheshold' threshold the NEB won't be optimized and the image
            with maximum uncertainty is evaluated. This prevents exploring
            very uncertain regions which can lead to probe unrealistic
            structures.

        Returns
        -------
        Minimum Energy Path from the initial to the final states.

        """
        trajectory_main = self.trajectory.split('.')[0]
        trajectory_observations = trajectory_main + '_observations.traj'

        # Start by saving the initial and final states.
        dump_observation(atoms=self.i_endpoint,
                         filename=trajectory_observations,
                         restart=self.use_previous_observations)
        self.use_previous_observations = True  # Switch on active learning.
        dump_observation(atoms=self.e_endpoint,
                         filename=trajectory_observations,
                         restart=self.use_previous_observations)

        # Saving the middle image
        train_images = ase.io.read(trajectory_observations, ':')
        if len(train_images) == 2:
            self.atoms.positions = self.images[2].get_positions()
            self.atoms.calc = self.ase_calc

            # Reference calculation
            self.atoms.get_potential_energy(force_consistent=self.force_consistent)
            self.atoms.get_forces()
            dump_observation(atoms=self.atoms, method='neb',
                             filename=trajectory_observations,
                             restart=self.use_previous_observations)
            self.function_calls += 1
            self.force_calls += 1

        else:
            self.function_calls = len(train_images)
            self.force_calls = len(train_images)

        self.step += 1

        weight_update = self.input_param.weight
        scale_update = self.input_param.scale

        self.max_unc_hist = []

        while True:

            # 1. Collect observations.
            train_images = ase.io.read(trajectory_observations, ':')

            train_data = ReferenceData(structure_files=train_images,
                                       file_format='ase',
                                       device=self.input_param.device,
                                       descriptor=self.input_param.descriptor,
                                       standardization=self.input_param.standardization,
                                       data_type=self.input_param.data_type,
                                       data_process=self.input_param.data_process,
                                       soap_param=self.input_param.soap_param,
                                       mace_param=self.input_param.mace_param,
                                       mask_constraints=self.input_param.mask_constraints)

            if train_data.standardization:
                train_data.standardize_energy_force(train_data.energy)

            # 2. Prepare a calculator.
            print('Training data size: ', len(train_images))
            print('Descriptor: ', self.input_param.descriptor)

            if update_step is not None and self.step >= update_step:
                update_step *= 2
                self.input_param.fit_weight = True
                self.input_param.fit_scale = True

                train_data.config_calculator(prior=self.input_param.prior,
                                             prior_update=self.input_param.prior_update,
                                             kerneltype='sqexp',
                                             scale=scale_update,
                                             weight=weight_update,
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
                self.input_param.fit_weight = False
                self.input_param.fit_scale = False

                train_data.config_calculator(prior=self.input_param.prior,
                                             prior_update=self.input_param.prior_update,
                                             kerneltype='sqexp',
                                             scale=scale_update,
                                             weight=weight_update,
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

            print()
            print('GPR model hyperparameters: ', train_data.calculator.hyper_params)

            self.model_calculator = GPRCalculator(calculator=train_data.calculator, train_data=train_data)
            weight_update = train_data.calculator.weight.clone().detach().item()
            scale_update = train_data.calculator.scale.clone().detach().item()

            # Detach calculator from the prev. optimized images (speed up).
            for i in self.images:
                i.calc = None
            # Train only one process.
            # calc.update_train_data(train_images, test_images=self.images)
            # Attach the trained calculator to each image.
            for i in self.images:
                i.calc = copy.deepcopy(self.model_calculator)

            # 3. Optimize the NEB in the predicted PES.
            # Use previous path uncertainty for deciding whether NEB or CI-NEB.
            # Climbing image NEB mode is risky when the model is trained with a few data points.
            # Switch on climbing image only when the uncertainty of the NEB the force of the climbing image are low.
            climbing_neb = False
            if climbing:
                if self.step > 1 and (self.level == self.n_flags or self.total_path_length < 0.8) and self.max_unc_hist[-1] <= unc_convergence:
                    parprint(f"Climbing image is now activated.")
                    climbing_neb = True
            else:
                pass

            ml_neb = NEB(self.images, climb=climbing_neb, method=self.neb_method, k=self.spring)
            # FIRE, MDMin, LBFGS, BFGS
            if optimizer.lower() == 'mdmin':
                neb_opt = MDMin(ml_neb, dt=dt, trajectory="gpr_neb.traj")
            elif optimizer.lower() == 'lbfgs':
                neb_opt = LBFGS(ml_neb, trajectory="gpr_neb.traj")
            elif optimizer.lower() == 'bfgs':
                neb_opt = BFGS(ml_neb, trajectory="gpr_neb.traj")
            else:
                neb_opt = FIRE(ml_neb, dt=dt, trajectory="gpr_neb.traj")

            # Optimize the images
            if self.level == self.n_flags or self.total_path_length < 0.8:
                neb_opt.run(fmax=fmax * 1.0, steps=ml_steps)
            else:
                neb_opt.run(fmax=fmax * 2.0, steps=ml_steps)

            nim = len(self.images) - 2
            nat = len(self.images[0])

            # PES prediction
            predictions = get_neb_predictions(self.images)

            # NEB force prediction (not potential energy force)
            neb_force = ml_neb.get_forces()  # (N_mobile_image * N_atom, 3) flat
            neb_force = neb_force.reshape(nim, nat, 3)
            max_f_image = np.sqrt((neb_force ** 2).sum(-1)).max().item()

            image_neb_force = np.zeros((len(self.images), nat, 3))
            image_neb_force[1:-1, :, :] = neb_force

            # Write trajectories
            filename_extxyz = f'gpr_neb_results_level{self.level:02d}_step{self.step:04d}.extxyz'
            self.save_neb_predictions_to_extxyz(predictions=predictions, image_neb_force=image_neb_force, filename=filename_extxyz)

            filename_traj = f'gpr_neb_results_level{self.level:02d}_step{self.step:04d}.traj'
            ase.io.write(filename_traj, self.images)

            # 5. Print output.
            neb_pred_energy = predictions['energy']
            max_e = np.max(neb_pred_energy)

            if uncertainty == 'energy':
                neb_pred_unc_energy = predictions['unc_energy']
                max_unc = np.max(neb_pred_unc_energy)
            else:
                neb_pred_unc_forces = predictions['unc_forces']
                max_unc = max(np.linalg.norm(f_unc, axis=1).max() for f_unc in neb_pred_unc_forces)
            self.max_unc_hist.append(max_unc)

            # Calculator of train_images is reference, while Calculator of self.images is GP
            if check_ref_force:
                max_f = get_fmax(train_images[-1])
            else:
                max_f = max_f_image

            pbf = max_e - self.very_i_endpoint_energy
            pbb = max_e - self.very_e_endpoint_energy

            msg = "--------------------------------------------------------"
            parprint(msg)
            parprint('Level:', self.level)
            parprint('Step:', self.step)
            parprint('Time:', time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))
            parprint('Predicted barrier (-->):', pbf)
            parprint('Predicted barrier (<--):', pbb)
            parprint('Number of images:', len(self.images))
            parprint('Max. uncertainty:', max_unc)
            parprint("Max. force:", max_f)
            msg = "--------------------------------------------------------\n"
            parprint(msg)

            # 6. Check convergence.
            if self.level == self.n_flags or self.total_path_length < 0.8:
                ok_unc = max_unc <= unc_convergence * 1.0
                if len(train_images) > 2 and max_f <= fmax and ok_unc and (climbing_neb or not climbing):
                    parprint('A saddle point was found.')

                    ase.io.write(self.trajectory, self.images)
                    parprint('Uncertainty of the images below threshold.')
                    parprint('NEB converged.')
                    parprint('The NEB path can be found in:', self.trajectory)
                    msg = "Visualize the last path using 'ase gui "
                    msg += self.trajectory
                    parprint(msg)
                    break

                # 7. Select next point to train (acquisition function):
                candidates = copy.deepcopy(self.images)[1:-1]

                if max_unc > unc_convergence:
                    sorted_candidates = acquisition(uncertainty=uncertainty,
                                                    candidates=candidates,
                                                    mode='uncertainty',
                                                    objective='max')
                else:
                    if self.step % 5 == 0:
                        sorted_candidates = acquisition(uncertainty=uncertainty,
                                                        candidates=candidates,
                                                        mode='fmax',
                                                        objective='min')
                    else:
                        sorted_candidates = acquisition(uncertainty=uncertainty,
                                                        candidates=candidates,
                                                        mode='ucb',
                                                        objective='max')

                # Select the best candidate.
                chosen_candidate = sorted_candidates.pop(0)

                # 8. Evaluate the target function and save it in *observations*.
                self.atoms.positions = chosen_candidate.get_positions()
                is_pos = self.atoms.get_positions().reshape(-1)
                train_image_positions = [image.get_positions().reshape(-1) for image in train_images]
                if is_duplicate_position(is_pos, train_image_positions) and max_f <= fmax * 4.0 and ok_unc and (climbing_neb or not climbing):
                    parprint('A saddle point was found.')

                    ase.io.write(self.trajectory, self.images)
                    parprint('Uncertainty of the images below threshold.')
                    parprint('NEB converged.')
                    parprint('The NEB path can be found in:', self.trajectory)
                    msg = "Visualize the last path using 'ase gui "
                    msg += self.trajectory
                    parprint(msg)
                    break

                self.atoms.calc = self.ase_calc
                self.atoms.get_potential_energy(force_consistent=self.force_consistent)
                self.atoms.get_forces()
                dump_observation(atoms=self.atoms,
                                 filename=trajectory_observations,
                                 restart=self.use_previous_observations)
                self.function_calls += 1
                self.force_calls += 1
                self.step += 1

            else:
                ok_unc = max_unc <= unc_convergence * 1.0
                if len(train_images) > 2 and max_f <= fmax and ok_unc and (climbing_neb or not climbing):
                    parprint('A saddle point was found.')

                    ase.io.write(self.trajectory, self.images)
                    parprint('Uncertainty of the images below threshold.')
                    parprint('NEB converged.')
                    parprint('The NEB path can be found in:', self.trajectory)
                    msg = "Visualize the last path using 'ase gui "
                    msg += self.trajectory
                    parprint(msg)
                    break

                ok_unc = max_unc <= unc_convergence * 10.0
                if (len(train_images) > 2 and ok_unc) or self.step > self.step_cutoff:
                    parprint('Level up.')
                    self.level += 1
                    self.step = 1
                    update_step = 1

                    train_image_positions = [image.get_positions().reshape(-1) for image in train_images]

                    # Check slope of images to set a narrow down range
                    slopes = [0.0]
                    for i in range(1, 4):
                        direction = self.images[i + 1].get_positions() - self.images[i - 1].get_positions()
                        direction /= np.linalg.norm(direction)
                        slopes.append(-(self.images[i].get_forces() * direction).sum())
                    slopes.append(0.0)
                    print("slopes:", slopes)

                    slope_signs = []
                    for s in slopes[1:-1]:
                        if s > fmax / 2.0:
                            slope_signs.append(1.0)
                        elif s < -fmax / 2.0:
                            slope_signs.append(-1.0)
                        else:
                            slope_signs.append(0.0)
                    first = slope_signs[0]
                    second = slope_signs[1]
                    third = slope_signs[2]

                    # case [3]
                    slope_case = 3

                    # case [0]
                    if first == 0.0 and third == 0.0:
                        if second >= 0.0:
                            slope_case = 1
                        else:
                            slope_case = 2

                    # case [1]: All signs match the first slope
                    elif all(s == 1.0 for s in slope_signs) or (third == 0.0):
                        slope_case = 1

                    # case [2]: All signs match the third slope
                    elif all(s == -1.0 for s in slope_signs) or (first == 0.0):
                        slope_case = 2

                    # Reset initial and final
                    if slope_case == 1:
                        print("slope_case:", slope_case)

                        # Reset only initial image
                        self.i_endpoint = self.images[1]
                        is_pos = self.i_endpoint.get_positions().reshape(-1)

                        if not is_duplicate_position(is_pos, train_image_positions):
                            self.atoms.positions = self.i_endpoint.get_positions()
                            self.atoms.calc = self.ase_calc
                            self.atoms.get_potential_energy(force_consistent=self.force_consistent)
                            self.atoms.get_forces()
                            dump_observation(atoms=self.atoms,
                                             filename=trajectory_observations,
                                             restart=self.use_previous_observations)

                        ase.io.write(f'initial_level{self.level:02d}.traj', self.i_endpoint)

                    elif slope_case == 2:
                        print("slope_case:", slope_case)

                        # Reset only final image
                        self.e_endpoint = self.images[3]
                        fs_pos = self.e_endpoint.get_positions().reshape(-1)

                        if not is_duplicate_position(fs_pos, train_image_positions):
                            self.atoms.positions = self.e_endpoint.get_positions()
                            self.atoms.calc = self.ase_calc
                            self.atoms.get_potential_energy(force_consistent=self.force_consistent)
                            self.atoms.get_forces()
                            dump_observation(atoms=self.atoms,
                                             filename=trajectory_observations,
                                             restart=self.use_previous_observations)

                        ase.io.write(f'final_level{self.level:02d}.traj', self.e_endpoint)

                    elif slope_case == 3:
                        print("slope_case:", slope_case)

                        # Reset both initial and final images
                        self.i_endpoint = self.images[1]
                        is_pos = self.i_endpoint.get_positions().reshape(-1)

                        if not is_duplicate_position(is_pos, train_image_positions):
                            self.atoms.positions = self.i_endpoint.get_positions()
                            self.atoms.calc = self.ase_calc
                            self.atoms.get_potential_energy(force_consistent=self.force_consistent)
                            self.atoms.get_forces()
                            dump_observation(atoms=self.atoms,
                                             filename=trajectory_observations,
                                             restart=self.use_previous_observations)

                        ase.io.write(f'initial_level{self.level:02d}.traj', self.i_endpoint)

                        self.e_endpoint = self.images[3]
                        fs_pos = self.e_endpoint.get_positions().reshape(-1)

                        if not is_duplicate_position(fs_pos, train_image_positions):
                            self.atoms.positions = self.e_endpoint.get_positions()
                            self.atoms.calc = self.ase_calc
                            self.atoms.get_potential_energy(force_consistent=self.force_consistent)
                            self.atoms.get_forces()
                            dump_observation(atoms=self.atoms,
                                             filename=trajectory_observations,
                                             restart=self.use_previous_observations)

                        ase.io.write(f'final_level{self.level:02d}.traj', self.e_endpoint)

                    # A) Create images using interpolation if user does define a path.
                    self.images = make_neb(self)

                    neb_interpolation = NEB(self.images, climb=False, method=self.neb_method,
                                            remove_rotation_and_translation=self.rrt)
                    neb_interpolation.interpolate(method='linear', mic=self.mic)

                    neb_interpolation = NEB(self.images, climb=True, method=self.neb_method,
                                            remove_rotation_and_translation=self.rrt)
                    neb_interpolation.interpolate(method='idpp', mic=self.mic)

                    # Guess spring constant (k) if not defined by the user.
                    self.total_path_length = 0.0
                    for i in range(len(self.images) - 1):
                        pos1 = self.images[i].positions.flatten()
                        pos2 = self.images[i + 1].positions.flatten()
                        distance = np.linalg.norm(pos2 - pos1)
                        self.total_path_length += distance

                    self.spring = 0.1 * (self.n_images - 1) / self.total_path_length

                    # Save initial interpolation.
                    self.initial_interpolation = self.images[:]

                    filename_extxyz = f'initial_interpolation_level{self.level:02d}.extxyz'
                    filename_traj = f'initial_interpolation_level{self.level:02d}.traj'
                    ase.io.write(filename_extxyz, self.initial_interpolation, format='extxyz')
                    ase.io.write(filename_traj, self.initial_interpolation)

                    print()
                    print('Total path length (Å): ', self.total_path_length)
                    print('Spring constant (eV/Å): ', self.spring)
                    continue

                # 7. Select next point to train (acquisition function):
                candidates = copy.deepcopy(self.images)[1:-1]
                sorted_candidates = acquisition(uncertainty=uncertainty,
                                                candidates=candidates,
                                                mode='uncertainty',
                                                objective='max')

                # Select the best candidate.
                chosen_candidate = sorted_candidates.pop(0)

                # 8. Evaluate the target function and save it in *observations*.
                self.atoms.positions = chosen_candidate.get_positions()
                self.atoms.calc = self.ase_calc
                self.atoms.get_potential_energy(force_consistent=self.force_consistent)
                self.atoms.get_forces()
                dump_observation(atoms=self.atoms,
                                 filename=trajectory_observations,
                                 restart=self.use_previous_observations)
                self.function_calls += 1
                self.force_calls += 1
                self.step += 1

        print_cite_neb()


@parallel_function
def make_neb(self, images_interpolation=None):
    """
    Creates a NEB from a set of images.
    """
    imgs = [self.i_endpoint[:]]
    for i in range(1, self.n_images - 1):
        image = self.i_endpoint[:]
        if images_interpolation is not None:
            image.set_positions(images_interpolation[i].get_positions())
        image.set_constraint(self.constraints)
        imgs.append(image)
    imgs.append(self.e_endpoint[:])
    return imgs


@parallel_function
def get_neb_predictions(images):
    neb_pred_energy = []
    neb_pred_forces = []
    neb_pred_unc_energy = []
    neb_pred_unc_forces = []

    for i in images[1:-1]:
        neb_pred_energy.append(i.get_potential_energy())
        neb_pred_forces.append(i.get_forces())

        unc_energy = i.calc.results['unc_energy']
        neb_pred_unc_energy.append(unc_energy)

        unc_forces = i.calc.results['unc_forces']
        neb_pred_unc_forces.append(unc_forces)

    neb_pred_energy.insert(0, images[0].get_potential_energy())
    neb_pred_energy.append(images[-1].get_potential_energy())

    neb_pred_forces.insert(0, np.zeros_like(neb_pred_forces[1]))
    neb_pred_forces.append(np.zeros_like(neb_pred_forces[1]))

    neb_pred_unc_energy.insert(0, 0.0)
    neb_pred_unc_energy.append(0.0)

    neb_pred_unc_forces.insert(0, np.zeros_like(neb_pred_unc_forces[1]))
    neb_pred_unc_forces.append(np.zeros_like(neb_pred_unc_forces[1]))

    predictions = {'energy': neb_pred_energy, 'forces': neb_pred_forces,
                   'unc_energy': neb_pred_unc_energy, 'unc_forces': neb_pred_unc_forces}

    return predictions


@parallel_function
def print_cite_neb():
    msg = "\n" + "-" * 79 + "\n"
    msg += "You are using GPR-accelerated NEB. Please cite: \n"
    msg += "[1] J. A. Garrido Torres, P. C. Jennings, M. H. Hansen, "
    msg += "J. R. Boes, and T. Bligaard, Phys. Rev. Lett. 122, 156001 (2019). "
    msg += "https://doi.org/10.1103/PhysRevLett.122.156001 \n"
    msg += "[2] O. Koistinen, F. B. Dagbjartsdóttir, V. Ásgeirsson, A. Vehtari,"
    msg += " and H. Jónsson, J. Chem. Phys. 147, 152720 (2017). "
    msg += "https://doi.org/10.1063/1.4986787 \n"
    msg += "[3] E. Garijo del Río, J. J. Mortensen, and K. W. Jacobsen, "
    msg += "Phys. Rev. B 100, 104103 (2019). "
    msg += "https://doi.org/10.1103/PhysRevB.100.104103. \n"
    msg += "[4] I. W. Yeu, A. Stuke, J. López-Zorrilla, J. M Stevenson, D. R Reichman, R. A Friesner, "
    msg += "A. Urban, and N. Artrith, npj Computational Materials 11, 156 (2025)."
    msg += "https://doi.org/10.1038/s41524-025-01651-0. \n"
    msg += "-" * 79 + '\n'
    parprint(msg)
