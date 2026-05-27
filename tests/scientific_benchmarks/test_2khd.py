import os

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
import pandas as pd
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts

from torsiontuner.data import get_graph_features, load_pdb
from torsiontuner.kinematics import rebuild_backbone
from torsiontuner.model import FineTunerGNN
from torsiontuner.montelione_utils import get_residue_rc_shifts, ramachandran_penalty


def calculate_cs_rmsd(phi, psi, target_shifts, res_indices):
    rc_shifts = get_residue_rc_shifts(res_indices)
    pred_shifts = predict_ca_shifts(phi, psi, rc_shifts)
    # Align predicated and target (target might be a subset)
    # For this simplified benchmark, we'll assume target_shifts is aligned
    # to res_indices[1:]
    return jnp.sqrt(jnp.mean((pred_shifts - target_shifts) ** 2))


def test_2khd_benchmark():
    """
    Scientific Benchmark: 2KHD (NESG VC_A0919).
    Verify that refinement reduces CSRMSD against experimental BMRB 16238 data.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data", "2khd")
    shift_data = pd.read_csv(os.path.join(data_dir, "shifts.csv"))
    pdb_path = os.path.join(data_dir, "starting_model.pdb")

    # 1. Load the actual AlphaFold model
    data = load_pdb(pdb_path)
    node_features, adj, edge_features = get_graph_features(data)
    res_indices = data["res_indices"]

    # 2. Align Experimental Data (Residues 4-20)
    # Target shifts are for residues 4-20.
    # In our arrays:
    # phi/psi start from residue 2 (index 0)
    # So residue 4 is index 2, residue 20 is index 18.
    target_shifts = jnp.array(shift_data["shift"].values)
    rc_shifts_full = get_residue_rc_shifts(res_indices)

    # Helper to calculate CSRMSD on the 4-20 subset
    def get_subset_cs_rmsd(phi, psi):
        # Slice for residues 4-20
        phi_subset = phi[2:19]
        psi_subset = psi[2:19]
        rc_subset = rc_shifts_full[3:20]  # res_indices 3..19 (4th to 20th residue)
        pred_shifts = predict_ca_shifts(phi_subset, psi_subset, rc_subset)
        return jnp.sqrt(jnp.mean((pred_shifts - target_shifts) ** 2))

    # Initial State
    phi_init = data["dihedrals"][2::3]
    psi_init = data["dihedrals"][0::3]
    initial_cs_rmsd = get_subset_cs_rmsd(phi_init, psi_init)
    print(f"Initial CSRMSD (4-20): {initial_cs_rmsd:.4f} ppm")

    # 3. Initialize Model and Optimizer
    key = jr.PRNGKey(42)
    model = FineTunerGNN(node_dim=20, hidden_dim=64, out_dim=2, n_layers=3, key=key)
    optimizer = optax.adamw(learning_rate=1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # 4. Refinement Loop (targeted at reducing CSRMSD)
    def loss_fn(model):
        deltas = model(node_features, adj, edge_features)
        _, updated_dihedrals = rebuild_backbone(
            data["init_coords"],
            data["lengths"],
            data["angles"],
            data["dihedrals"],
            deltas,
        )
        pred_phi = updated_dihedrals[2::3]
        pred_psi = updated_dihedrals[0::3]

        # Calculate CSRMSD on the subset we have data for
        phi_subset = pred_phi[2:19]
        psi_subset = pred_psi[2:19]
        rc_subset = rc_shifts_full[3:20]
        pred_shifts_subset = predict_ca_shifts(phi_subset, psi_subset, rc_subset)
        cs_loss = jnp.mean((pred_shifts_subset - target_shifts) ** 2)

        # Structural regularity and small deltas
        ansurr_loss = ramachandran_penalty(pred_phi, pred_psi)
        reg_loss = jnp.mean(deltas**2)

        return 1.0 * cs_loss + 0.1 * ansurr_loss + 0.01 * reg_loss

    @eqx.filter_jit
    def make_step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    print("Running refinement...")
    for step in range(51):
        model, opt_state, loss = make_step(model, opt_state)
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    # 5. Final Validation
    final_deltas = model(node_features, adj, edge_features)
    _, final_dihedrals = rebuild_backbone(
        data["init_coords"],
        data["lengths"],
        data["angles"],
        data["dihedrals"],
        final_deltas,
    )
    final_phi = final_dihedrals[2::3]
    final_psi = final_dihedrals[0::3]
    final_cs_rmsd = get_subset_cs_rmsd(final_phi, final_psi)

    print(f"Final CSRMSD (4-20): {final_cs_rmsd:.4f} ppm")

    # Success Criterion: Refinement should improve the fit to experimental data
    assert final_cs_rmsd < initial_cs_rmsd, "Refinement failed to reduce CSRMSD"
    print("Scientific Benchmark 2KHD: PASSED")


if __name__ == "__main__":
    test_2khd_benchmark()
