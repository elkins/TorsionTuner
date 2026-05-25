import jax
import jax.numpy as jnp
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts, RANDOM_COIL_CA
from src.data import RESIDUE_TYPES

# Pre-compute a JAX array for random coil shifts to allow JIT-compatible indexing
_RC_ARRAY = jnp.array([RANDOM_COIL_CA[res] for res in RESIDUE_TYPES])

def get_residue_rc_shifts(res_indices):
    """Map residue indices to their random coil CA shifts."""
    # Use JAX indexing instead of list comprehension
    return _RC_ARRAY[res_indices]

def montelione_loss(phi, psi, target_shifts, res_indices):
    """
    A loss function inspired by the Montelione group's approach of 
    validating structures against experimental chemical shifts.
    """
    rc_shifts = get_residue_rc_shifts(res_indices)
    pred_shifts = predict_ca_shifts(phi, psi, rc_shifts)
    
    # MSE between predicted and experimental CA shifts
    return jnp.mean((pred_shifts - target_shifts)**2)

def calculate_ansurr_proxy(phi, psi):
    """
    A simplified proxy for the ANSURR score, measuring structural 
    consistency with secondary structure propensity.
    
    This penalizes phi/psi values that fall outside the broad 'favored' 
    regions of the Ramachandran plot using a soft potential.
    """
    # Alpha region: phi ~ -60, psi ~ -45
    alpha_dist = jnp.sqrt((phi + 1.05)**2 + (psi + 0.78)**2)
    # Beta region: phi ~ -120, psi ~ 135
    beta_dist = jnp.sqrt((phi + 2.09)**2 + (psi - 2.35)**2)
    # Left-handed alpha: phi ~ 60, psi ~ 45
    l_alpha_dist = jnp.sqrt((phi - 1.05)**2 + (psi - 0.78)**2)
    
    # Soft minimum distance to any favored region
    # Using a softmin-like approach
    sigma = 0.5
    dist_sq = jnp.stack([alpha_dist, beta_dist, l_alpha_dist], axis=-1)**2
    penalty = -sigma * jax.nn.logsumexp(-dist_sq / sigma, axis=-1)
    
    return jnp.mean(penalty)
