import torch
import numpy as np


def get_N_batch(len_dataset, batch_size):
    """
    Returns the number of batches for a given batch size and dataset size
    """
    N_batch = int(len_dataset / batch_size)
    residue = len_dataset - N_batch * batch_size

    if residue >= int(batch_size / 2) or N_batch == 0:
        if residue != 0:
            N_batch += 1

    return N_batch


def get_batch_indexes_N_batch(len_dataset, N_batch):
    """
    Returns the indexes of the structures in StructureDataset that belong to each batch
    """
    finish = 0
    indexes = []

    base, extra = divmod(len_dataset, N_batch)
    N_per_batch = base + (torch.arange(N_batch) < extra).long()

    for i in range(N_batch):
        start = finish
        finish = start + N_per_batch[i]
        indexes.append([start, finish])

    return indexes


def standard_output(energy_ref, energy_target, force_target):
    """
    energy_ref: (Ndata,) numpy array
    energy_target: (Ndata,) numpy array
    force_target: (Ndata, Natom, 3) numpy array

    Returns:
        scaled_energy_target: (Ndata,) numpy array of standardized energy
        scaled_force_target: (Ndata, Natom, 3) numpy array of standardized atomic forces
    """
    n_system = force_target.shape[0]
    n_atom = force_target.shape[1]

    # Mean and std of energy (population standard deviation)
    mean_energy = np.mean(energy_ref)
    std_energy = np.std(energy_ref)

    # Standardize: (energy_target - mean) / std
    scaled_energy_target = (energy_target - mean_energy) / std_energy

    # Average of force is approximately 0
    scaled_force_target = force_target.reshape(n_system, -1) / std_energy
    scaled_force_target = scaled_force_target.reshape(n_system, n_atom, 3)

    return scaled_energy_target, scaled_force_target


def inverse_standard_output(energy_ref, scaled_energy_target, scaled_force_target):
    """
    energy_ref: (Ndata,) numpy array of reference energy
    scaled_energy_target: (Ndata,) numpy array of standardized energy
    scaled_force_target: (Ndata, Natom, 3) numpy array of standardized atomic forces

    Returns:
        energy_target: (Ndata,) numpy array of unscaled energy
        force_target: (Ndata, Natom, 3) numpy array of unscaled atomic forces
    """
    n_system = scaled_force_target.shape[0]
    n_atom = scaled_force_target.shape[1]

    # Mean and std of energy (population standard deviation)
    mean_energy = np.mean(energy_ref)
    std_energy = np.std(energy_ref)

    # Restore Energy: scaled_energy_target * std + mean
    energy_target = scaled_energy_target * std_energy + mean_energy

    # Restore Force: scaled_force_target * std
    force_target = scaled_force_target.reshape(n_system, -1) * std_energy
    force_target = force_target.reshape(n_system, n_atom, 3)

    return energy_target, force_target
