import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class GPRCalculator(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(self, calculator, train_energy, **kwargs):
        Calculator.__init__(self, **kwargs)

        self.calculator = calculator
        self.train_energy = train_energy
        self.results = {}

    def calculate(self, atoms=None,
                  properties=['energy', 'forces'],
                  system_changes=all_changes):
        '''
        Calculate the energy and forces for a given Atoms structure.
        Predicted energies can be obtained by *atoms.get_potential_energy()*,
        predicted forces using *atoms.get_forces()*
        '''

        Calculator.calculate(self, atoms, properties, system_changes)

        pred, _ = self.calculator.eval_data_per_data(eval_image=atoms)
        energy_gpr_scale = pred[0]
        force_gpr_scale = pred[1:].view(len(atoms), 3)

        mean_energy = np.mean(self.train_energy)
        std_energy = np.std(self.train_energy)

        # Restore Energy: scaled_energy_target * std + mean
        energy_gpr = energy_gpr_scale.cpu().detach().numpy() * std_energy + mean_energy

        # Restore Force: scaled_force_target * std
        force_gpr = force_gpr_scale.cpu().detach().numpy() * std_energy

        self.results['energy'] = energy_gpr
        self.results['forces'] = force_gpr
