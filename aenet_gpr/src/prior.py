import torch


class ConstantPrior:
    '''Constant prior, with energy = constant and zero forces

    Parameters:

    constant: energy value for the constant.
    '''

    def __init__(self,
                 constant,
                 data_type=torch.float64,
                 device='cpu',
                 atoms_xyz_mask=None,
                 **kwargs):
        self.data_type = data_type
        self.device = device
        self.atoms_xyz_mask = atoms_xyz_mask
        self.constant = torch.tensor(constant, dtype=self.data_type, device=self.device)

    def set_constant(self, constant):
        self.constant = constant

    def potential_per_data(self, Natom):
        if self.atoms_xyz_mask is not None:
            Nmask = self.atoms_xyz_mask.shape[0]
        else:
            Nmask = 3 * Natom

        output = torch.zeros((1 + Nmask,), dtype=self.data_type, device=self.device)
        output[0] = self.constant
        return output

    def potential_batch(self, Ndata, Natom):
        if self.atoms_xyz_mask is not None:
            Nmask = self.atoms_xyz_mask.shape[0]
        else:
            Nmask = 3 * Natom

        output_array = torch.zeros(((1 + Nmask) * Ndata,), dtype=self.data_type, device=self.device)
        output_array[:Ndata] = self.constant
        return output_array

    def update(self, Ndata, Natom, Y, L):
        """Update the constant to maximize the marginal likelihood.

        The optimization problem:
        m = argmax [-1/2 (y-m).T K^-1(y-m)]

        can be turned into an algebraic problem
        m = [ u.T K^-1 y]/[u.T K^-1 u]

        where u is the constant prior with energy 1 (eV).

        parameters:
        ------------
        X: training data
        Y: training targets
        L: Cholesky factor of the kernel """

        self.set_constant(torch.tensor(1.0, dtype=self.data_type, device=self.device))
        u = self.potential_batch(Ndata, Natom)

        # Solve the system L * L^T * v = Y (Cholesky solve)
        w = torch.cholesky_solve(u.view(-1, 1), L, upper=False)
        m = torch.matmul(w.view(-1), Y.flatten()) / torch.matmul(w.view(-1), u)

        self.set_constant(m)
