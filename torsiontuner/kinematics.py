import jax.numpy as jnp
from diff_biophys.geometry import chain_nerf


def rebuild_backbone(
    init_coords: jnp.ndarray,
    initial_lengths: jnp.ndarray,
    initial_angles: jnp.ndarray,
    initial_dihedrals: jnp.ndarray,
    delta_phi_psi: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Rebuild backbone coordinates from initial state and predicted deltas.

    Args:
        init_coords: (3, 3) - N1, CA1, C1
        initial_lengths: (N_atoms - 3,)
        initial_angles: (N_atoms - 3,)
        initial_dihedrals: (N_atoms - 3,)
        delta_phi_psi: (N_residues, 2) - [delta_phi, delta_psi]

    Returns:
        (N_atoms, 3) coordinates
    """
    n_residues = delta_phi_psi.shape[0]

    # Copy initial dihedrals to avoid in-place (not an issue in JAX but for clarity)
    updated_dihedrals = jnp.array(initial_dihedrals)

    # Map deltas to the dihedrals array
    # dihedrals[0] = psi1
    # dihedrals[1] = omega1 (fixed)
    # dihedrals[2] = phi2
    # dihedrals[3] = psi2
    # ...

    # Update psis: residue i (1-indexed) -> index 3(i-1)
    # delta_phi_psi[i-1, 1] is delta_psi_i
    # We'll use vmap or simple slice addition

    # PSI updates (residues 1 to N-1)
    psi_indices = jnp.arange(0, 3 * (n_residues - 1), 3)
    updated_dihedrals = updated_dihedrals.at[psi_indices].add(delta_phi_psi[:-1, 1])

    # PHI updates (residues 2 to N)
    phi_indices = jnp.arange(2, 3 * (n_residues - 1), 3)
    updated_dihedrals = updated_dihedrals.at[phi_indices].add(delta_phi_psi[1:, 0])

    # Reconstruct
    # initial_lengths is (N-1,), we need [2:] for atoms 4..N
    # initial_angles is (N-2,), we need [1:] for atoms 4..N
    coords = chain_nerf(
        init_coords, initial_lengths[2:], initial_angles[1:], updated_dihedrals
    )
    return coords, updated_dihedrals
