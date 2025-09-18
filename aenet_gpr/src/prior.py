import torch


class ConstantPrior:
    '''Constant prior, with energy = constant and zero forces

    Parameters:

    constant: energy value for the constant.
    '''

    def __init__(self, constant, data_type=torch.float64, device='cpu', atoms_mask=None, **kwargs):
        self.data_type = data_type
        self.device = device
        self.constant = torch.tensor(constant, dtype=self.data_type, device=self.device)
        self.atoms_mask = atoms_mask

    def set_constant(self, constant):
        self.constant = constant

    def potential_per_data(self, image=None, get_forces=True):
        if get_forces:
            if self.atoms_mask is not None:
                Nmask = self.atoms_mask.shape[0]
            else:
                Nmask = len(image.get_chemical_symbols()) * 3
            output = torch.zeros((1 + Nmask,), dtype=self.data_type, device=self.device)

            output[0] = self.constant
            return output
        else:
            return self.constant

    def potential_batch(self, images=None, get_forces=True):
        if get_forces:
            if self.atoms_mask is not None:
                Nmask = self.atoms_mask.shape[0]
            else:
                Nmask = len(images[0].get_chemical_symbols()) * 3

            output_array = torch.zeros(((1 + Nmask) * len(images),), dtype=self.data_type, device=self.device)
            output_array[:len(images)] = self.constant
            return output_array
        else:
            output_array = torch.tensor([self.constant for i in range(len(images))], dtype=self.data_type, device=self.device)
            return output_array

    def update(self, images, Y, L, use_forces):
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
        if use_forces:
            u = self.potential_batch(images)
        else:
            u = self.potential_batch(images, get_forces=False)

        # Solve the system L * L^T * v = Y (Cholesky solve)
        w = torch.cholesky_solve(u.view(-1, 1), L, upper=False)
        m = torch.matmul(w.view(-1), Y.flatten()) / torch.matmul(w.view(-1), u)

        self.set_constant(m)
