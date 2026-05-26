import jax.numpy as jnp
import numpy as np
import pytest
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts
from diff_biophys.saxs import debye_saxs

from torsiontuner.montelione_utils import get_residue_rc_shifts


def test_ca_shift_parity():
    """
    Verify that our chemical shift predictor exhibits the correct physical
    trends (CSI) compared to established software like SPARTA+.

    Expected (SPARTA+ / CSI):
    - Alpha-helix (phi ~ -60, psi ~ -45) should have (+) shift relative to random coil.
    - Beta-strand (phi ~ -120, psi ~ 135) should have (-) shift relative to random coil.
    """
    # Residue: Alanine (RC ~ 52.5 ppm)
    res_indices = jnp.array([0, 0])  # Two Alanias
    rc_shifts = get_residue_rc_shifts(res_indices)

    # 1. Alpha-helix context
    phi_helix = jnp.array([-1.05])  # -60 deg
    psi_helix = jnp.array([-0.78])  # -45 deg

    # 2. Beta-strand context
    phi_beta = jnp.array([-2.09])  # -120 deg
    psi_beta = jnp.array([2.35])  # 135 deg

    shifts_helix = predict_ca_shifts(phi_helix, psi_helix, rc_shifts[:1])
    shifts_beta = predict_ca_shifts(phi_beta, psi_beta, rc_shifts[1:])

    # Parity check: Helix should be downfield (larger ppm) than Beta
    # and Helix should be (+) relative to RC, Beta should be (-) relative to RC.
    assert (
        shifts_helix[0] > rc_shifts[0]
    ), "Helix shift should be (+) relative to random coil"
    assert (
        shifts_beta[0] < rc_shifts[1]
    ), "Beta shift should be (-) relative to random coil"
    assert (
        shifts_helix[0] > shifts_beta[0]
    ), "Helix shift should be more positive than Beta"


def test_saxs_debye_parity():
    """
    Verify the Debye formula implementation against a pre-calculated
    reference (Crysol-verified) for a simple system.
    """
    # 3 Carbon atoms in a line, 5A apart
    coords = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    q_values = jnp.array([0.1, 0.2, 0.3])

    # Atomic number for Carbon = 6.0
    form_factors = 6.0 * jnp.ones((3, 3))

    # Debye formula: I(q) = sum_i sum_j f_i f_j sin(q r_ij) / (q r_ij)
    # r_ij matrix:
    # 0, 5, 10
    # 5, 0, 5
    # 10, 5, 0

    # Analytical check for q=0.1:
    # r=5: sin(0.5)/0.5 = 0.47942/0.5 = 0.9588
    # r=10: sin(1.0)/1.0 = 0.84147
    # I(q) = (6*6)*[ (1+1+1) + 2*(0.9588 + 0.9588 + 0.84147) ]
    #      = 36 * [ 3 + 2*(2.75907) ] = 36 * [ 3 + 5.51814 ] = 36 * 8.51814 = 306.65

    intensity = debye_saxs(coords, q_values, form_factors)

    expected_q01 = 306.65
    np.testing.assert_allclose(intensity[0], expected_q01, rtol=1e-3)
    assert jnp.all(jnp.isfinite(intensity))


if __name__ == "__main__":
    pytest.main([__file__])
