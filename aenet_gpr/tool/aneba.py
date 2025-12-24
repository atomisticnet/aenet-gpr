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


def is_duplicate_position(is_pos, train_image_positions):
    # is_pos: 1D numpy array (Eg, shape = (3N,))
    # train_image_positions: list of 1D numpy arrays
    for pos in train_image_positions:
        if np.array_equal(is_pos, pos):
            return True
    return False


class ANEBA:

    def __init__(self, start, end, calculator=None,
                 interpolation='idpp', n_images=5, n_flags=3, step_cutoff=15, k=None, mic=False,
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
        self.start = start
        self.end = end
        self.n_images = n_images
        self.n_flags = n_flags
        self.step_cutoff = step_cutoff
        self.level = 1
        self.mic = mic
        self.rrt = remove_rotation_and_translation
        self.neb_method = neb_method
        self.spring = k
        self.i_endpoint = ase.io.read(self.start, '-1')
        self.e_endpoint = ase.io.read(self.end, '-1')

        # Model calculator
        self.model_calculator = calculator

        self.step = 0
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
            self.i_endpoint.calc = copy.deepcopy(self.model_calculator)
        if self.e_endpoint.calc is None:
            self.e_endpoint.calc = copy.deepcopy(self.model_calculator)
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
        Store NEB predictions (energy, force)
        into Atoms objects and write to extxyz file.

        Parameters:
            predictions (dict): Dictionary with 'energy', 'forces'
            filename (str): Output file path (.extxyz)
        """
        out = []
        for i, image in enumerate(self.images):
            atoms = image.copy()

            # Energies
            atoms.info['energy'] = float(predictions['energy'][i])

            # Forces
            atoms.arrays['pes_forces'] = np.linalg.norm(predictions['forces'][i], axis=1)
            atoms.arrays['neb_forces'] = np.linalg.norm(image_neb_force[i], axis=1)

            out.append(atoms)

        # Write to extxyz file
        ase.io.write(filename, out, format='extxyz')

    def run(self,
            fmax=0.05,
            dt=0.05,
            ml_steps=150,
            optimizer="MDMin",
            climbing=False):

        """
        Executing run will start the NEB optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angstrom).

        dt : float
            dt parameter for MDMin.

        ml_steps: int
            Maximum number of steps for the NEB optimization on the
            modelled potential energy surface.

        Returns
        -------
        Minimum Energy Path from the initial to the final states.

        """

        self.step += 1

        while True:

            # Detach calculator from the prev. optimized images (speed up).
            for i in self.images:
                i.calc = None

            # Attach the trained calculator to each image.
            for i in self.images:
                i.calc = copy.deepcopy(self.model_calculator)

            # 3. Optimize the NEB in the predicted PES.
            # Climbing image NEB mode is risky when the model is trained with a few data points.
            climbing_neb = False
            if climbing:
                if self.step > 1 and (self.level == self.n_flags or self.total_path_length < 0.5):
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
            if self.level == self.n_flags or self.total_path_length < 0.5:
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
            parprint("Max. force:", max_f)
            msg = "--------------------------------------------------------\n"
            parprint(msg)

            # 6. Check convergence.
            if self.level == self.n_flags or self.total_path_length < 0.5:
                if max_f <= fmax and (climbing_neb or not climbing):
                    parprint('A saddle point was found.')

                    ase.io.write(self.trajectory, self.images)
                    parprint('NEB converged.')
                    parprint('The NEB path can be found in:', self.trajectory)
                    msg = "Visualize the last path using 'ase gui "
                    msg += self.trajectory
                    parprint(msg)
                    break

            else:
                if max_f <= fmax and (climbing_neb or not climbing):
                    parprint('A saddle point was found.')

                    ase.io.write(self.trajectory, self.images)
                    parprint('NEB converged.')
                    parprint('The NEB path can be found in:', self.trajectory)
                    msg = "Visualize the last path using 'ase gui "
                    msg += self.trajectory
                    parprint(msg)
                    break

                if max_f <= fmax * 2.0 or self.step > self.step_cutoff:
                    parprint('Level up.')
                    self.level += 1
                    self.step = 1

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
                        ase.io.write(f'initial_level{self.level:02d}.traj', self.i_endpoint)

                    elif slope_case == 2:
                        print("slope_case:", slope_case)

                        # Reset only final image
                        self.e_endpoint = self.images[3]
                        ase.io.write(f'final_level{self.level:02d}.traj', self.e_endpoint)

                    elif slope_case == 3:
                        print("slope_case:", slope_case)

                        # Reset both initial and final images
                        self.i_endpoint = self.images[1]
                        ase.io.write(f'initial_level{self.level:02d}.traj', self.i_endpoint)

                        self.e_endpoint = self.images[3]
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

    for i in images[1:-1]:
        neb_pred_energy.append(i.get_potential_energy())
        neb_pred_forces.append(i.get_forces())

    neb_pred_energy.insert(0, images[0].get_potential_energy())
    neb_pred_energy.append(images[-1].get_potential_energy())

    neb_pred_forces.insert(0, np.zeros_like(neb_pred_forces[1]))
    neb_pred_forces.append(np.zeros_like(neb_pred_forces[1]))

    predictions = {'energy': neb_pred_energy, 'forces': neb_pred_forces}

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
