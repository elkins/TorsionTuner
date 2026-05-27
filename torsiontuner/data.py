import biotite.structure as stripe
import biotite.structure.io as stripeio
import jax
import jax.numpy as jnp
import numpy as np
from diff_biophys.geometry import (
    compute_bond_angles,
    compute_bond_lengths,
    compute_dihedrals,
)

RESIDUE_TYPES = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
RESIDUE_MAP = {res: i for i, res in enumerate(RESIDUE_TYPES)}


def load_pdb(file_path: str) -> dict:
    """
    Load a PDB file and extract backbone information.

    Args:
        file_path: Path to the PDB file.

    Returns:
        A dictionary containing coordinates, residue indices, elements,
        and initial geometry parameters.
    """
    struct = stripeio.load_structure(file_path)
    if isinstance(struct, stripe.AtomArrayStack):
        struct = struct[0]

    # Filter for backbone atoms (N, CA, C)
    mask = np.isin(struct.atom_name, ["N", "CA", "C"])
    backbone = struct[mask]

    res_names = []
    coords = []
    elements = []

    # Iterate through residues
    res_ids = np.unique(backbone.res_id)
    for rid in res_ids:
        res_atoms = backbone[backbone.res_id == rid]

        # Check if all three backbone atoms are present
        n_atom = res_atoms[res_atoms.atom_name == "N"]
        ca_atom = res_atoms[res_atoms.atom_name == "CA"]
        c_atom = res_atoms[res_atoms.atom_name == "C"]

        if len(n_atom) == 1 and len(ca_atom) == 1 and len(c_atom) == 1:
            coords.extend([n_atom.coord[0], ca_atom.coord[0], c_atom.coord[0]])
            elements.extend([n_atom.element[0], ca_atom.element[0], c_atom.element[0]])
            res_names.append(res_atoms.res_name[0])

    coords_array = jnp.array(coords)
    res_indices = jnp.array([RESIDUE_MAP.get(name, 0) for name in res_names])
    elements_array = np.array(elements)

    # Initial geometry for NeRF
    init_coords = coords_array[:3]
    lengths = compute_bond_lengths(coords_array)
    angles = compute_bond_angles(coords_array)
    dihedrals = compute_dihedrals(coords_array)

    return {
        "coords": coords_array,
        "res_indices": res_indices,
        "elements": elements_array,
        "init_coords": init_coords,
        "lengths": lengths,
        "angles": angles,
        "dihedrals": dihedrals,
    }


def get_graph_features(data: dict) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convert loaded data into graph features.

    Args:
        data: The dictionary returned by load_pdb.

    Returns:
        A tuple of (node_features, adjacency_matrix, edge_features).
    """
    res_indices = data["res_indices"]
    coords = data["coords"]
    n_residues = len(res_indices)

    # Node features: residue type
    node_features = jax.nn.one_hot(res_indices, 20)

    # Sequence adjacency
    seq_adj = jnp.eye(n_residues, k=1) + jnp.eye(n_residues, k=-1)

    # Spatial adjacency (CA-CA)
    ca_coords = coords[1::3]
    dist_matrix = jnp.sqrt(
        jnp.sum((ca_coords[:, None, :] - ca_coords[None, :, :]) ** 2, axis=-1)
    )
    spatial_adj = (dist_matrix < 10.0).astype(jnp.float32)

    adj = (seq_adj + spatial_adj > 0).astype(jnp.float32)

    # Edge features: normalized distances
    edge_features = dist_matrix[:, :, None] / 10.0  # Normalize by cutoff

    return node_features, adj, edge_features


if __name__ == "__main__":
    import jax

    # Quick test if possible
    print("Data module ready.")
