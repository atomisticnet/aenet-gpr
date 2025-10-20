import numpy as np
import copy
import glob

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet
from ase.md import MDLogger

from ase import io
from ase.atoms import Atoms, units
from ase.parallel import parprint, parallel_function

from aenet_gpr.src import GPRCalculator
from aenet_gpr.util import ReferenceData
from aenet_gpr.tool import dump_observation
from aenet_gpr.inout.input_parameter import InputParameters


class AIDMD:

    def __init__(self, start, input_param: InputParameters, model_calculator=None, calculator=None,
                 trajectory='AIDMD.traj', use_previous_observations=True, force_consistent=None):

        # Convert Atoms and list of Atoms to trajectory files.
        if isinstance(start, Atoms):
            io.write('initial.traj', start)
            start = 'initial.traj'

        # NEB parameters.
        self.input_param = input_param
        self.start = start

        # GP calculator:
        self.model_calculator = model_calculator

        # Active Learning setup (Single-point calculations).
        self.force_calls = 0

        self.ase_calc = calculator
        self.atoms = io.read(self.start, '-1')

        self.constraints = self.atoms.constraints
        self.trajectory = trajectory
        self.use_previous_observations = use_previous_observations
        self.force_consistent = force_consistent

    def save_md_predictions_to_extxyz(self, atoms, predictions, filename):

        atoms = atoms.copy()
        atoms.info['energy'] = float(predictions['energy'])
        atoms.info['unc_energy'] = float(predictions['unc_energy'])

        atoms.arrays['forces'] = np.linalg.norm(predictions['forces'], axis=1)
        atoms.arrays['unc_forces'] = np.linalg.norm(predictions['unc_forces'], axis=1)

        io.write(filename, atoms, format='extxyz')

    def update_train_data(self, trajectory_observations):
        train_images = io.read(trajectory_observations, ':')
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

        train_data.config_calculator(prior=self.input_param.prior,
                                     prior_update=self.input_param.prior_update,
                                     kerneltype='sqexp',
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

        return train_data

    def run(self,
            max_unc=0.1,
            temp=300,
            time_step=2,
            md_steps=500,
            interval=5):

        trajectory_main = self.trajectory.split('.')[0]
        trajectory_observations = trajectory_main + '_observations.traj'

        # Start by saving the initial states.
        dump_observation(atoms=self.atoms,
                         filename=trajectory_observations,
                         restart=self.use_previous_observations)

        MaxwellBoltzmannDistribution(self.atoms, temperature_K=temp)
        dyn = VelocityVerlet(self.atoms, time_step * units.fs)

        train_images = io.read(trajectory_observations, ':')
        self.force_calls = len(train_images)

        # 1. Collect observations.
        train_data = self.update_train_data(trajectory_observations)

        for step in range(md_steps):

            # 2. GPR calculator
            self.model_calculator = GPRCalculator(calculator=train_data.calculator, train_data=train_data)
            self.atoms.calc = copy.deepcopy(self.model_calculator)

            # 3. MD run
            dyn.run(1)

            # 4. Write log within trajectory file
            predictions = get_predictions(self.atoms)
            if step % interval == 0:
                filename = f'MD_step_{step:04d}.extxyz'
                self.save_md_predictions_to_extxyz(atoms=self.atoms, predictions=predictions, filename=filename)

            # 5. Check force uncertainty.
            neb_pred_unc_forces = predictions['unc_forces']
            max_unc_f = np.linalg.norm(neb_pred_unc_forces, axis=1).max()

            print(f"[Step {step:04d}] Energy = {predictions['energy']:.6f} eV, "
                  f"Max Uncertainty = {max_unc_f:.4f} eV/Å")

            # 6. Check convergence.
            if max_unc_f <= max_unc:
                continue
            else:
                print(f"  ➤ Reference calculation triggered at step {step}")
                # 7. Evaluate the target function and save it in *observations*.
                filename = f'For_training_step_{step:04d}.extxyz'
                self.save_md_predictions_to_extxyz(atoms=self.atoms, predictions=predictions, filename=filename)

                try:
                    uncertain_atom = self.atoms.copy()
                    uncertain_atom.calc = self.ase_calc
                    uncertain_atom.get_potential_energy(force_consistent=self.force_consistent)
                    uncertain_atom.get_forces()
                    dump_observation(atoms=uncertain_atom,
                                     filename=trajectory_observations,
                                     restart=self.use_previous_observations)

                    self.force_calls += 1
                    train_data = self.update_train_data(trajectory_observations)
                except Exception as e:
                    print(f"Reference calculation failed at step {step}: {e}")
                    continue

        # Combined trajectory info
        print(f"\n Total reference calculations: {self.force_calls}")

        files = sorted(glob.glob('MD_step_*.extxyz'))
        images = [io.read(f) for f in files]
        io.write('MD_trajectory_combined.extxyz', images)

        print_cite_neb()


def get_predictions(atoms: Atoms):
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    unc_energy = atoms.calc.results['unc_energy']
    unc_forces = atoms.calc.results['unc_forces']

    predictions = {'energy': energy,
                   'forces': forces,
                   'unc_energy': unc_energy,
                   'unc_forces': unc_forces}

    return predictions


def print_cite_neb():
    msg = "\n" + "-" * 79 + "\n"
    msg += "You are using GPR-accelerated MD. Please cite: \n"
    msg += "[1] I. W. Yeu, A. Stuke, J. López-Zorrilla, J. M Stevenson, D. R Reichman, R. A Friesner, "
    msg += "A. Urban, and N. Artrith, npj Computational Materials 11, 156 (2025)."
    msg += "https://doi.org/10.1038/s41524-025-01651-0. \n"
    msg += "-" * 79 + '\n'
    parprint(msg)
