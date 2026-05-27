import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts

from torsiontuner.data import get_graph_features, load_pdb
from torsiontuner.kinematics import rebuild_backbone
from torsiontuner.model import FineTunerGNN
from torsiontuner.montelione_utils import (
    get_residue_rc_shifts,
    montelione_loss,
    ramachandran_penalty,
)


def test_torsional_regularization_recovery():
    """
    Scientific Validation (Item 3): Torsional Regularization.
    Prove that the model can recover from unphysical 'mangled' geometry
    by minimizing the Ramachandran penalty and experimental loss.
    """
    # 1. Load ground truth structure
    data = load_pdb("test_helix.pdb")
    node_features, adj, edge_features = get_graph_features(data)
    res_indices = data["res_indices"]

    # 2. Generate 'Experimental' Target Data from ground truth
    q_values = jnp.linspace(0.01, 0.5, 50)
    # Simple Carbon form factors
    jnp.ones((len(data["coords"]), len(q_values)))

    true_phi = data["dihedrals"][2::3]
    true_psi = data["dihedrals"][0::3]
    rc_shifts = get_residue_rc_shifts(res_indices)
    target_shifts = predict_ca_shifts(true_phi, true_psi, rc_shifts[1:])

    # 3. Mangle the structure (inject unphysical dihedrals)
    # Move half the residues' phi/psi to (0,0) - a disallowed region
    mangled_dihedrals = jnp.array(data["dihedrals"])
    for i in range(1, 21, 2):  # Mangle every other residue
        psi_idx = 3 * (i - 1)
        phi_idx = 3 * (i - 1) - 1
        if psi_idx >= 0 and psi_idx < len(mangled_dihedrals):
            mangled_dihedrals = mangled_dihedrals.at[psi_idx].set(0.0)
        if phi_idx >= 0 and phi_idx < len(mangled_dihedrals):
            mangled_dihedrals = mangled_dihedrals.at[phi_idx].set(0.0)

    # Initial Ramachandran penalty for mangled structure
    mangled_phi = mangled_dihedrals[2::3]
    mangled_psi = mangled_dihedrals[0::3]
    initial_penalty = ramachandran_penalty(mangled_phi, mangled_psi)

    # 4. Initialize Model
    key = jr.PRNGKey(42)
    model = FineTunerGNN(node_dim=20, hidden_dim=64, out_dim=2, n_layers=3, key=key)

    optimizer = optax.adamw(learning_rate=5e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # 5. Loss Function (Refinement)
    def loss_fn(model):
        deltas = model(node_features, adj, edge_features)
        _, updated_dihedrals = rebuild_backbone(
            data["init_coords"],
            data["lengths"],
            data["angles"],
            mangled_dihedrals,  # Start from mangled
            deltas,
        )

        pred_phi = updated_dihedrals[2::3]
        pred_psi = updated_dihedrals[0::3]

        # Heavy weight on Ramachandran penalty to force physical recovery
        nmr_loss = montelione_loss(pred_phi, pred_psi, target_shifts, res_indices[1:])
        rama_loss = ramachandran_penalty(pred_phi, pred_psi)
        reg_loss = jnp.mean(deltas**2)

        return 0.1 * nmr_loss + 10.0 * rama_loss + 0.001 * reg_loss

    @eqx.filter_jit
    def make_step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # 6. Run Refinement
    print(f"Initial Rama Penalty: {initial_penalty:.6f}")
    for step in range(1001):
        model, opt_state, loss = make_step(model, opt_state)
        if step % 200 == 0:
            # Re-calculate final state for printing
            final_deltas = model(node_features, adj, edge_features)
            _, final_dihedrals = rebuild_backbone(
                data["init_coords"],
                data["lengths"],
                data["angles"],
                mangled_dihedrals,
                final_deltas,
            )
            final_phi = final_dihedrals[2::3]
            final_psi = final_dihedrals[0::3]
            current_penalty = ramachandran_penalty(final_phi, final_psi)
            print(f"Step {step}, Loss: {loss:.6f}, Rama Penalty: {current_penalty:.6f}")

    # 7. Validation
    final_deltas = model(node_features, adj, edge_features)
    _, final_dihedrals = rebuild_backbone(
        data["init_coords"],
        data["lengths"],
        data["angles"],
        mangled_dihedrals,
        final_deltas,
    )
    final_phi = final_dihedrals[2::3]
    final_psi = final_dihedrals[0::3]
    final_penalty = ramachandran_penalty(final_phi, final_psi)

    print(f"Final Rama Penalty: {final_penalty:.6f}")

    # Recovery Criterion: Penalty should decrease significantly
    assert (
        final_penalty < initial_penalty
    ), f"Model failed to improve physical realism: {final_penalty} >= {initial_penalty}"


if __name__ == "__main__":
    test_torsional_regularization_recovery()
