import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class GPRCalculator(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(self, calculator, train_data, **kwargs):
        Calculator.__init__(self, **kwargs)

        self.calculator = calculator
        self.train_data = train_data
        self.results = {}

    def calculate(self, atoms=None,
                  properties=None,
                  system_changes=all_changes):
        '''
        Calculate the energy and forces for a given Atoms structure.
        Predicted energies can be obtained by *atoms.get_potential_energy()*,
        predicted forces using *atoms.get_forces()*
        '''

        if properties is None:
            properties = ['energy', 'forces']
        Calculator.calculate(self, atoms, properties, system_changes)

        energy_gpr, force_gpr, unc_e_gpr, unc_f_gpr = self.calculator.eval_per_data(eval_image=atoms, get_variance=True)

        energy_gpr = energy_gpr.cpu().detach().numpy()
        force_gpr = force_gpr.cpu().detach().numpy()
        unc_e_gpr = unc_e_gpr.cpu().detach().numpy()
        unc_f_gpr = unc_f_gpr.cpu().detach().numpy()

        if self.train_data.standardization:
            mean_energy = np.mean(self.train_data.energy)
            std_energy = np.std(self.train_data.energy)

            # Restore Energy: scaled_energy_target * std + mean
            energy_gpr = energy_gpr * std_energy + mean_energy

            # Restore Force: scaled_force_target * std
            force_gpr = force_gpr * std_energy

        self.results['energy'] = energy_gpr
        self.results['forces'] = force_gpr
        self.results['unc_energy'] = unc_e_gpr
        self.results['unc_forces'] = unc_f_gpr
