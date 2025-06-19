import numpy as np
import os
import warnings

from ase.parallel import parallel_function
from ase import io
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator, all_properties
from ase.calculators.calculator import PropertyNotImplementedError


@parallel_function
def dump_observation(atoms, filename, restart, method='-'):
    """
    Saves a trajectory file containing the atoms observations.

    Parameters
    ----------
    atoms: object
        Atoms object to be appended to previous observations.
    filename: string
        Name of the trajectory file to save the observations.
    restart: boolean
        Append mode (true or false).
    method: string
        Label with the optimizer name to be appended in atoms.info['method'].
     """
    atoms.info['method'] = method

    if restart:
        if os.path.exists(filename):
            prev_atoms = io.read(filename, ':')  # Active learning.
            if atoms not in prev_atoms:  # Avoid duplicates.
                # Update observations.
                trj = TrajectoryWriter(atoms=atoms, filename=filename,
                                       mode='a')
                trj.write()
        else:
            io.write(filename=filename, images=atoms, append=False)
    if not restart:
        io.write(filename=filename, images=atoms, append=False)


@parallel_function
def get_fmax(atoms):
    """
    Returns fmax for a given atoms structure.
    """
    forces = atoms.get_forces()
    return np.sqrt((forces**2).sum(axis=1).max())


def dump2list(atoms, destination, method):
    """
    Append the atoms object together with its calculator into
    a list. This is some sort of a copy and paste method.
    """

    # Copy the calculator
    calc = atoms.calc

    if calc is not None:
        results = {}
        implemented_properties = []

        for prop in all_properties:
            try:
                x = calc.get_property(prop, atoms,
                                      allow_calculation=False)

            except PropertyNotImplementedError:
                x = None
            results[prop] = x
            implemented_properties.append(prop)
        calculator = SinglePointCalculator(atoms, **results)
        calculator.name = calc.name
        calculator.implemented_properties = implemented_properties

        if hasattr(calc, 'parameters'):
            calculator.parameters.update(calc.parameters)

    # Copy the atoms object. The new object has no calculator
    atoms = atoms.copy()
    atoms.info['method'] = method
    atoms.calc = calculator

    destination.append(atoms)


class TrainingSet:

    """
    Training set class to store calculations as they are run
    by the different AID methods.

    It allows to save stuff to a trajectory file or to keep
    observations in memory as a list. In the future it would
    be nice to have them also stored to a database

    TODO: Add a db mode.

    Parameters:
    -----------
    destination: str or list
        This is where the training set is going to be stored
        options:
            trajectory file name: atoms are stored in a traj file.
                This is, written to disk.
            list: atoms are kept in memory, kept in a list.
    use_previous_observations: bool
        This is some sort of restart for the dataset,
        whether you would like to use the other stuff in your
        destination or not.
    atoms: Atoms object
        Atoms object to be written or dumpen into the training set
    """

    def __init__(self, destination, use_previous_observations, atoms=None):

        self.atoms = atoms
        if type(destination) is list:
            if not use_previous_observations:
                if len(destination) > 0:
                    warning = ("use_previous_observations == False together "
                               "with a non empty list as destintion deletes "
                               "the content of the list. If this is an "
                               "unwanted behaviour, consider passing an empty "
                               "list.")
                    warnings.warn(warning)

                self.destination = []
            else:
                self.destination = destination
            self.mode = 'list'
        elif type(destination) is str:
            if not destination.endswith('.traj'):
                msg = "*destination* should be a trajectory file"
                raise NotImplementedError(msg)
            else:
                self.destination = destination
                self.use_prev_obs = use_previous_observations
                self.mode = 'traj'
        else:
            msg = "*destination* should be a file or a list"
            raise NotImplementedError(msg)

    def dump(self, atoms=None, method='-'):
        """
        Saves the atoms observations into the training set.

        Parameters
        ----------
        atoms: object
            Atoms object to be appended to previous observations.
        method: string
            Label with the optimizer name to be appended in
            atoms.info['method'].
         """

        if atoms is None:
            atoms = self.atoms

        if self.mode == 'list':
            dump2list(atoms, destination=self.destination, method=method)

        elif self.mode == 'traj':
            if not self.use_prev_obs:
                dump_observation(atoms, filename=self.destination,
                                 method=method, restart=False)
                self.use_prev_obs = True
            else:
                dump_observation(atoms, filename=self.destination,
                                 method=method, restart=True)
        else:
            raise NotImplementedError()

    def write(self, atoms=None, method='-'):
        """
        Copy of the dump method,
        so that the training set can become an observer.
        """
        self.dump(atoms, method)

    def load_set(self):
        """
        Loads the training set

        Returns:
        --------
        A list of ase.Atoms objects
        """
        if self.mode == 'list':
            return self.destination.copy()
        elif self.mode == 'traj':
            return io.read(self.destination, ':')
        else:
            raise NotImplementedError()

    def load_last(self):
        """
        Last entry written to the training set

        Returns:
        --------
        An atoms object
        """
        if self.mode == 'list':
            return self.destination[-1]
        elif self.mode == 'traj':
            return io.read(self.destination, -1)
        else:
            raise NotImplementedError()
