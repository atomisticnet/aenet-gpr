from e3nn import o3
from e3nn.util.jit import compile_mode
from mace.modules.utils import get_edge_vectors_and_lengths
from mace import data
from mace.tools import torch_geometric, utils

import torch


def numerical_mace_descriptor_gradient(atoms, model, delta=1e-4, num_layers=-1, device="cpu"):
    """
    atoms: ASE Atoms object
    model: pre-trained MACE model
    delta: displacement for numerical differentiation

    Returns:
        gradient: shape (N_atoms, N_atoms, 3, descriptor_dim)
    """
    aenet_mace = AenetMACE(model)

    atoms = atoms.copy()
    n_atoms = len(atoms)

    desc_0 = aenet_mace.get_descriptors(atoms, num_interaction_layers=num_layers, device=device)  # shape: (n_atoms, D)
    D = desc_0.shape[1]

    grad = torch.empty((n_atoms, n_atoms, 3, D))

    for i in range(n_atoms):
        for j in range(3):  # x, y, z
            # forward step
            atoms_f = atoms.copy()
            atoms_f.positions[i, j] += delta
            desc_f = aenet_mace.get_descriptors(atoms_f, num_interaction_layers=num_layers)

            # backward step
            atoms_b = atoms.copy()
            atoms_b.positions[i, j] -= delta
            desc_b = aenet_mace.get_descriptors(atoms_b, num_interaction_layers=num_layers)

            # central difference
            grad[:, i, j, :] = (desc_f - desc_b) / (2 * delta)

    return desc_0, grad


def extract_scalar_irreps(irreps):
    # Only take the first (scalar) irrep component
    mul, (l, p) = irreps[0]
    return slice(0, mul * (2 * l + 1))


@compile_mode("script")
class AenetMACE(torch.nn.Module):
    def __init__(self, mace_model):
        super().__init__()
        self.mace_model = mace_model
        self.r_max = mace_model.r_max
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.mace_model.atomic_numbers]
        )

    def atoms_to_batch(self, atoms, z_table):
        info_keys = {"total_spin": "spin", "total_charge": "charge"}
        arrays_keys = {}
        keyspec = data.KeySpecification(
            info_keys=info_keys, arrays_keys=arrays_keys
        )
        config = data.config_from_atoms(atoms, key_specification=keyspec)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config,
                    z_table=z_table,
                    cutoff=self.r_max)
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader))  # .to(self.device)
        return batch

    def get_descriptors(self, atoms, num_interaction_layers=-1, device="cpu"):
        """
        Lightweight descriptor extractor for pretrained MACE models.

        Parameters:
            mace_model (torch.nn.Module): Loaded MACE model.
            atomic_data_dict (dict): Dict with keys like 'node_attrs', 'atom_pos', 'edge_index', 'shifts'.
            num_interaction_layers (int): Number of interaction layers to extract. -1 = all.
            device (str): 'cpu' or 'cuda'.

        Returns:
            desc (torch.Tensor): Descriptor tensor (N_atoms, descriptor_dim).
        """

        # Get input components
        atomic_data_dict = self.atoms_to_batch(atoms).to_dict()

        node_attrs = atomic_data_dict["node_attrs"].to(device)
        atom_pos = atomic_data_dict["atom_pos"].to(device)
        edge_index = atomic_data_dict["edge_index"].to(device)
        shifts = atomic_data_dict["shifts"].to(device)

        with torch.no_grad():
            node_feats = self.mace_model.node_embedding(node_attrs)
            vectors, lengths = get_edge_vectors_and_lengths(
                atom_pos, edge_index, shifts
            )
            edge_attrs = self.mace_model.spherical_harmonics(vectors)
            edge_feats = self.mace_model.radial_embedding(
                lengths, node_attrs, edge_index, self.mace_model.atomic_numbers
            )
            if isinstance(edge_feats, tuple):
                edge_feats, cutoff = edge_feats
            else:
                cutoff = None

            node_feats_list = []
            num_layers = len(self.mace_model.interactions) if num_interaction_layers == -1 else num_interaction_layers

            for i in range(num_layers):
                interaction = self.mace_model.interactions[i]
                product = self.mace_model.products[i]

                node_feats, sc = interaction(
                    node_attrs=node_attrs,
                    node_feats=node_feats,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                    cutoff=cutoff,
                    first_layer=(i == 0),
                )

                node_feats = product(
                    node_feats=node_feats, sc=sc, node_attrs=node_attrs
                )
                irr_slice = extract_scalar_irreps(product.linear.irreps_out)
                node_feats_list.append(node_feats[..., irr_slice])

            desc = torch.cat(node_feats_list, dim=-1)
            return desc