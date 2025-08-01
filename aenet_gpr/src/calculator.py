import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes


class GPRCalculator(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(self, calculator, train_data, **kwargs):
        Calculator.__init__(self, **kwargs)

        self.calculator = calculator
        self.train_data = train_data
        self.results = {}

    def calculate(self, atoms=None,
                  properties=['energy', 'forces', 'uncertainty'],
                  system_changes=all_changes):
        '''
        Calculate the energy and forces for a given Atoms structure.
        Predicted energies can be obtained by *atoms.get_potential_energy()*,
        predicted forces using *atoms.get_forces()*
        '''

        Calculator.calculate(self, atoms, properties, system_changes)

        pred, kernel = self.calculator.eval_data_per_data(eval_image=atoms)
        energy_gpr = pred[0].cpu().detach().numpy()
        if self.train_data.mask_constraints:
            force_gpr = torch.zeros((len(atoms) * 3), dtype=pred.dtype)
            force_gpr[self.train_data.atoms_mask] = pred[1:]
            force_gpr = force_gpr.view(len(atoms), 3).cpu().detach().numpy()
        else:
            force_gpr = pred[1:].view(len(atoms), 3).cpu().detach().numpy()

        var = self.calculator.eval_variance_per_data(get_variance=True, eval_image=atoms, k=kernel)
        uncertainty_gpr = torch.sqrt(var[0, 0]) / self.calculator.weight
        uncertainty_gpr = uncertainty_gpr.cpu().detach().numpy()

        if self.train_data.standardization:
            mean_energy = np.mean(self.train_data.energy)
            std_energy = np.std(self.train_data.energy)

            # Restore Energy: scaled_energy_target * std + mean
            energy_gpr = energy_gpr * std_energy + mean_energy

            # Restore Force: scaled_force_target * std
            force_gpr = force_gpr * std_energy

        self.results['energy'] = energy_gpr
        self.results['forces'] = force_gpr
        self.results['uncertainty'] = uncertainty_gpr
