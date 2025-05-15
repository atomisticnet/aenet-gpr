import torch


class EuclideanDistance:

    @staticmethod
    def distance(fp1, fp2):
        '''
        Distance function between two fingerprints.
        '''
        return torch.linalg.norm(fp1 - fp2)


class BaseKernelType:
    '''
    Base class for all kernel types with common properties,
    attributes and methods.
    '''

    def __init__(self, weight=1.0, scale=1.0, metric=EuclideanDistance):
        # Currently, all the kernel types take weight and scale as parameters

        self.params = {'scale': scale, 'weight': weight}
        self.metric = metric

    @property
    def scale(self):        
        return self.params['scale']

    @property
    def weight(self):
        return self.params['weight']

    def update(self, params):
        '''
        Update the kernel function hyperparameters.
        '''

        self.params.update(params)


class SquaredExp(BaseKernelType):
    '''
    Squared Exponential kernel function
    '''

    def kernel_function(self, fp1, fp2):
        '''
        Kernel function between two atomic fingerprints 'fp1' and 'fp2'.
        '''
        
        return (self.weight**2 *
                torch.exp(-self.metric.distance(fp1, fp2)**2 / 2 / self.scale**2))
    
    def kernel_gradient(self, fp1, fp2, fp1_grad):
        '''
        Calculates the derivative of the kernel between
        fp1 and fp2 w.r.t. all coordinates in fp1.

        Chain rule:

                d k(x, x')    dk      d D(x, x')
                ---------- = ----  X  ----------
                   d xi       dD         d xi
        '''

        prefactor = - self.kernel_function(fp1, fp2) / self.scale**2
    
        return prefactor * torch.einsum('i,hi->h', fp1 - fp2, fp1_grad)

    def kernel_hessian(self, fp1, fp2, fp1_grad, fp2_grad):
        '''
        Kernel hessian w.r.t. atomic coordinates in both 'fp1' and 'fp2'

                    d^2 k(x, x')
                    ------------
                     dx_i dx'_j
        '''

        prefactor = self.kernel_function(fp1, fp2) / self.scale**2

        # [Nfeatures] * [Natoms * 3, Nfeatures] -> [Natoms * 3]
        DdD_dr1 = torch.einsum('i,hi->h', fp1 - fp2, fp1_grad)
        DdD_dr2 = torch.einsum('i,hi->h', fp2 - fp1, fp2_grad)

        # [Natoms * 3] * [Natoms * 3] -> [Natoms * 3, Natoms * 3]
        C0 = prefactor * (1 / self.scale**2) * torch.einsum('i,j->ij', DdD_dr1, DdD_dr2)
        
        # [Natoms * 3, Nfeatures] * [Natoms * 3, Nfeatures] -> [Natoms * 3, Natoms * 3]
        C1 = prefactor * torch.einsum('hi,ki->hk', fp1_grad, fp2_grad)

        return C0 + C1
