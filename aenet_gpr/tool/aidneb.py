import numpy as np
import copy
import time

from ase import io
from ase.atoms import Atoms
from ase.optimize import MDMin, FIRE
from ase.parallel import parprint, parallel_function

try:
    from ase.mep import NEB
except ModuleNotFoundError:
    from ase.neb import NEB

from aenet_gpr.src import GPRCalculator
from aenet_gpr.util import ReferenceData
from aenet_gpr.tool import acquisition
from aenet_gpr.tool import dump_observation, get_fmax
from aenet_gpr.inout.input_parameter import InputParameters


class AIDNEB:

    def __init__(self, start, end, input_param: InputParameters, model_calculator=None, calculator=None,
                 interpolation='idpp', n_images=15, k=None, mic=False,
                 neb_method='improvedtangent',  #'improvedtangent', 'aseneb'
                 remove_rotation_and_translation=False,
                 max_train_data=25, force_consistent=None,
                 max_train_data_strategy='nearest_observations',
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
            io.write('initial.traj', start)
            start = 'initial.traj'
        if isinstance(end, Atoms):
            io.write('final.traj', end)
            end = 'final.traj'
        interp_path = None
        if interpolation != 'idpp' and interpolation != 'linear':
            interp_path = interpolation
        if isinstance(interp_path, list):
            io.write('initial_path.traj', interp_path)
            interp_path = 'initial_path.traj'

        # NEB parameters.
        self.input_param = input_param
        self.start = start
        self.end = end
        self.n_images = n_images
        self.mic = mic
        self.rrt = remove_rotation_and_translation
        self.neb_method = neb_method
        self.spring = k
        self.i_endpoint = io.read(self.start, '-1')
        self.e_endpoint = io.read(self.end, '-1')

        # GP calculator:
        self.model_calculator = model_calculator
        # if model_calculator is None:
        #     self.model_calculator = GPCalculator(
        #         train_images=[],
        #         prior=None, kernel=None,
        #         fit_weight='update',
        #         update_prior_strategy='fit',
        #         kernel_params={'weight': 1.0, 'scale': 0.4},
        #         noise=0.005,
        #         max_train_data_strategy=max_train_data_strategy,
        #         max_train_data=max_train_data)

        # Active Learning setup (Single-point calculations).
        self.step = 0
        self.function_calls = 0
        self.force_calls = 0
        self.ase_calc = calculator
        self.atoms = io.read(self.start, '-1')

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
            neb_mic = NEB(mic_images, climb=False, method=self.neb_method,
                          remove_rotation_and_translation=self.rrt)
            neb_mic.interpolate(method='linear', mic=self.mic)
            self.i_endpoint.positions = mic_images[1].positions[:]
            self.e_endpoint.positions = mic_images[-2].positions[:]

        # Calculate the initial and final end-points (if necessary).
        self.i_endpoint.calc = copy.deepcopy(self.ase_calc)
        self.e_endpoint.calc = copy.deepcopy(self.ase_calc)
        self.i_endpoint.get_potential_energy(force_consistent=force_consistent)
        self.i_endpoint.get_forces()
        self.e_endpoint.get_potential_energy(force_consistent=force_consistent)
        self.e_endpoint.get_forces()

        # Calculate the distance between the initial and final endpoints.
        # d_start_end = sqeuclidean(self.i_endpoint.positions.flatten(),
        #                           self.e_endpoint.positions.flatten())
        d_start_end = np.sum((self.i_endpoint.positions.flatten() -
                              self.e_endpoint.positions.flatten()) ** 2)

        # A) Create images using interpolation if user does define a path.
        if interp_path is None:
            if isinstance(self.n_images, float):
                self.n_images = int(d_start_end / self.n_images)
            if self. n_images <= 3:
                self.n_images = 3
            self.images = make_neb(self)
            self.spring = 2. * np.sqrt(self.n_images - 1) / d_start_end

            neb_interpolation = NEB(self.images, climb=False, k=self.spring,
                                    method=self.neb_method,
                                    remove_rotation_and_translation=self.rrt)
            neb_interpolation.interpolate(method='linear', mic=self.mic)
            if interpolation == 'idpp':
                neb_interpolation = NEB(
                    self.images, climb=True,
                    k=self.spring, method=self.neb_method,
                    remove_rotation_and_translation=self.rrt)
                neb_interpolation.interpolate(method='idpp', mic=self.mic)
                # neb_interpolation.idpp_interpolate(optimizer=FIRE, mic=self.mic)

        # B) Alternatively, the user can propose an initial path.
        if interp_path is not None:
            images_path = io.read(interp_path, ':')
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
        if self.spring is None:
            self.spring = 2. * (np.sqrt(self.n_images - 1) / d_start_end)
        # Save initial interpolation.
        self.initial_interpolation = self.images[:]

    def run(self, fmax=0.05, unc_convergence=0.025, dt=0.05, ml_steps=100,
            max_step=2.0):

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

        max_step: float
            Safe control parameter. This parameter controls the degree of
            freedom of the NEB optimization in the modelled potential energy
            surface or the. If the uncertainty of the NEB lies above the
            'max_step' threshold the NEB won't be optimized and the image
            with maximum uncertainty is evaluated. This prevents exploring
            very uncertain regions which can lead to probe unrealistic
            structures.

        Returns
        -------
        Minimum Energy Path from the initial to the final states.

        """
        trajectory_main = self.trajectory.split('.')[0]
        trajectory_observations = trajectory_main + '_observations.traj'
        trajectory_candidates = trajectory_main + '_candidates.traj'

        # Start by saving the initial and final states.
        dump_observation(atoms=self.i_endpoint,
                         filename=trajectory_observations,
                         restart=self.use_previous_observations)
        self.use_previous_observations = True  # Switch on active learning.
        dump_observation(atoms=self.e_endpoint,
                         filename=trajectory_observations,
                         restart=self.use_previous_observations)

        train_images = io.read(trajectory_observations, ':')
        if len(train_images) == 2:
            middle = int(self.n_images * (2. / 3.))
            e_is = self.i_endpoint.get_potential_energy()
            e_fs = self.e_endpoint.get_potential_energy()

            if e_is > e_fs:
                middle = int(self.n_images * (1. / 3.))

            self.atoms.positions = self.images[middle].get_positions()
            self.atoms.calc = self.ase_calc
            self.atoms.get_potential_energy(
                force_consistent=self.force_consistent)
            self.atoms.get_forces()
            dump_observation(atoms=self.atoms, method='neb',
                             filename=trajectory_observations,
                             restart=self.use_previous_observations)
            self.function_calls += 1
            self.force_calls += 1
            self.step += 1

        weight_update = 1.0
        while True:

            # 0. Start from initial interpolation every 50 steps.
            if self.step % 50 == 0:
                parprint('Starting from initial interpolation...')
                self.images = copy.deepcopy(self.initial_interpolation)

            # 1. Collect observations.
            # This serves to use_previous_observations from a previous
            # (and/or parallel) runs.
            train_images = io.read(trajectory_observations, ':')
            train_data = ReferenceData(structure_files=train_images,
                                       file_format='ase',
                                       device=self.input_param.device,
                                       descriptor=self.input_param.descriptor,
                                       data_type=self.input_param.data_type,
                                       data_process=self.input_param.data_process,
                                       soap_param=self.input_param.soap_param,
                                       standardization=self.input_param.standardization,
                                       mask_constraints=True,
                                       fit_weight=self.input_param.fit_weight)
            train_data.set_data()
            train_data.standardize_energy_force(train_data.energy)

            # 2. Prepare a calculator.
            print('Training data size: ', len(train_data.images))
            train_data.config_calculator(kerneltype='sqexp', scale=0.4, weight=weight_update)
            print('GPR model hyperparameters: ', train_data.calculator.hyper_params)
            self.model_calculator = GPRCalculator(calculator=train_data.calculator, train_data=train_data)
            weight_update = train_data.calculator.weight.clone().detach().item()

            # Detach calculator from the prev. optimized images (speed up).
            # for i in self.images:
            #     i.calc = None
            # # Train only one process.
            # calc.update_train_data(train_images, test_images=self.images)
            # Attach the calculator (already trained) to each image.
            for i in self.images:
                i.calc = copy.deepcopy(self.model_calculator)

            # 3. Optimize the NEB in the predicted PES.
            # Get path uncertainty for deciding whether NEB or CI-NEB.
            predictions = get_neb_predictions(self.images)
            neb_pred_uncertainty = predictions['uncertainty']

            # Climbing image NEB mode is risky when the model is trained
            # with a few data points. Switch on climbing image (CI-NEB) only
            # when the uncertainty of the NEB is low.
            climbing_neb = False
            if np.max(neb_pred_uncertainty) <= unc_convergence:
                parprint('Climbing image is now activated.')
                climbing_neb = True
            ml_neb = NEB(self.images, climb=climbing_neb,
                         method=self.neb_method, k=self.spring)
            neb_opt = FIRE(ml_neb, trajectory=self.trajectory)

            # Safe check to optimize the images.
            if np.max(neb_pred_uncertainty) <= max_step:
                neb_opt.run(fmax=(fmax * 0.80), steps=ml_steps)

            predictions = get_neb_predictions(self.images)
            neb_pred_energy = predictions['energy']
            neb_pred_uncertainty = predictions['uncertainty']

            # 5. Print output.
            max_e = np.max(neb_pred_energy)
            pbf = max_e - self.i_endpoint.get_potential_energy(
                force_consistent=self.force_consistent)
            pbb = max_e - self.e_endpoint.get_potential_energy(
                force_consistent=self.force_consistent)
            msg = "--------------------------------------------------------"
            parprint(msg)
            parprint('Step:', self.step)
            parprint('Time:', time.strftime("%m/%d/%Y, %H:%M:%S",
                                            time.localtime()))
            parprint('Predicted barrier (-->):', pbf)
            parprint('Predicted barrier (<--):', pbb)
            parprint('Max. uncertainty:', np.max(neb_pred_uncertainty))
            parprint('Number of images:', len(self.images))
            parprint("fmax:", get_fmax(train_images[-1]))
            msg = "--------------------------------------------------------\n"
            parprint(msg)

            # 6. Check convergence.
            # Max.forces and NEB images uncertainty must be below *fmax* and
            # *unc_convergence* thresholds.
            if len(train_images) > 2 and get_fmax(train_images[-1]) <= fmax:
                parprint('A saddle point was found.')
                if np.max(neb_pred_uncertainty[1:-1]) < unc_convergence:
                    io.write(self.trajectory, self.images)
                    parprint('Uncertainty of the images above threshold.')
                    parprint('NEB converged.')
                    parprint('The NEB path can be found in:', self.trajectory)
                    msg = "Visualize the last path using 'ase gui "
                    msg += self.trajectory
                    parprint(msg)
                    break

            # 7. Select next point to train (acquisition function):

            # Candidates are the optimized NEB images in the predicted PES.
            candidates = copy.deepcopy(self.images)[1:-1]

            if np.max(neb_pred_uncertainty) > unc_convergence:
                sorted_candidates = acquisition(train_images=train_images,
                                                candidates=candidates,
                                                mode='uncertainty',
                                                objective='max')
            else:
                if self.step % 5 == 0:
                    sorted_candidates = acquisition(train_images=train_images,
                                                    candidates=candidates,
                                                    mode='fmax',
                                                    objective='min')
                else:
                    sorted_candidates = acquisition(train_images=train_images,
                                                    candidates=candidates,
                                                    mode='ucb',
                                                    objective='max')

            # Select the best candidate.
            best_candidate = sorted_candidates.pop(0)

            # Save the other candidates for multi-task optimization.
            io.write(trajectory_candidates, sorted_candidates)

            # 8. Evaluate the target function and save it in *observations*.
            self.atoms.positions = best_candidate.get_positions()
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
    neb_pred_unc = []
    for i in images:
        neb_pred_energy.append(i.get_potential_energy())
        unc = i.calc.results['uncertainty']
        neb_pred_unc.append(unc)
    neb_pred_unc[0] = 0.0
    neb_pred_unc[-1] = 0.0
    predictions = {'energy': neb_pred_energy, 'uncertainty': neb_pred_unc}
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
