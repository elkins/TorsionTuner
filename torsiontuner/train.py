import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
from torsiontuner.data import load_pdb, get_graph_features
from torsiontuner.model import FineTunerGNN
from torsiontuner.kinematics import rebuild_backbone
from torsiontuner.montelione_utils import montelione_loss, calculate_ansurr_proxy, get_residue_rc_shifts
from diff_biophys.saxs import debye_saxs
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts

from dataclasses import dataclass

@dataclass
class Config:
    """
    Configuration for the structural refinement training process.
    
    Attributes:
        learning_rate: Learning rate for the AdamW optimizer.
        n_steps: Number of training iterations.
        hidden_dim: Hidden dimension size for the GNN layers.
        n_layers: Number of GNN layers.
        w_saxs: Weight for the SAXS loss term.
        w_nmr: Weight for the NMR (chemical shift) loss term.
        w_ansurr: Weight for the structural quality (ANSURR proxy) loss term.
        w_reg: Weight for the regularization loss term.
    """
    learning_rate: float = 5e-4
    n_steps: int = 101
    hidden_dim: int = 64
    n_layers: int = 3
    w_saxs: float = 1.0
    w_nmr: float = 0.1
    w_ansurr: float = 0.05
    w_reg: float = 0.01
    saxs_q_min: float = 0.01
    saxs_q_max: float = 0.5
    saxs_q_points: int = 50

def train(config: Config = Config()):
    """
    Execute the structural refinement training loop.
    
    This function loads a starting PDB, simulates experimental data (SAXS and NMR),
    and trains a GNN to predict torsional adjustments that minimize the 
    multi-objective loss.
    
    Args:
        config: A Config instance containing hyperparameters.
        
    Returns:
        The trained FineTunerGNN model.
    """
    # 1. Load data
    data = load_pdb("test_helix.pdb")
    node_features, adj, edge_features = get_graph_features(data)
    res_indices = data['res_indices']
    
    # Target: let's say we want to recover the original coordinates (mocking SAXS)
    # Generate synthetic SAXS target from "true" data
    q_values = jnp.linspace(config.saxs_q_min, config.saxs_q_max, config.saxs_q_points)
    
    # Form factors based on elements (atomic numbers as a crude approximation)
    element_map = {'C': 6.0, 'N': 7.0, 'O': 8.0, 'S': 16.0}
    atom_charges = jnp.array([element_map.get(el, 6.0) for el in data['elements']])
    form_factors = atom_charges[:, None] * jnp.ones((1, len(q_values)))
    
    target_saxs = debye_saxs(data['coords'], q_values, form_factors)
    
    # Target Chemical Shifts (mocking from true dihedrals)
    true_phi = data['dihedrals'][2::3]
    true_psi = data['dihedrals'][0::3]
    rc_shifts = get_residue_rc_shifts(res_indices)
    target_shifts = predict_ca_shifts(true_phi, true_psi, rc_shifts[1:])
    
    # 2. Initialize Model
    key = jr.PRNGKey(42)
    model_key, train_key = jr.split(key)
    model = FineTunerGNN(
        node_dim=20, # Residue types
        hidden_dim=config.hidden_dim,
        out_dim=2, # delta_phi, delta_psi
        n_layers=config.n_layers,
        key=model_key
    )
    
    # 3. Optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=config.learning_rate)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # 4. Loss Function
    def loss_fn(model, node_features, adj, edge_features, init_coords, lengths, angles, dihedrals, res_indices, target_saxs, target_shifts, q_values, form_factors):
        # Predict deltas
        deltas = model(node_features, adj, edge_features)
        
        # Reconstruct
        refined_coords, updated_dihedrals = rebuild_backbone(
            init_coords,
            lengths,
            angles,
            dihedrals,
            deltas
        )
        
        # Calculate SAXS
        pred_saxs = debye_saxs(refined_coords, q_values, form_factors)
        
        # Normalize intensities for loss
        scale = jnp.max(target_saxs)
        saxs_loss = jnp.mean(((pred_saxs - target_saxs) / scale)**2)
        
        # NMR Loss (Chemical Shifts)
        pred_phi = updated_dihedrals[2::3]
        pred_psi = updated_dihedrals[0::3]
        nmr_loss = montelione_loss(pred_phi, pred_psi, target_shifts, res_indices[1:])
        
        # Quality Loss (ANSURR Proxy)
        ansurr_loss = calculate_ansurr_proxy(pred_phi, pred_psi)
        
        # Regularization: keep deltas small
        reg_loss = jnp.mean(deltas**2)
        
        total_loss = (config.w_saxs * saxs_loss + 
                      config.w_nmr * nmr_loss + 
                      config.w_ansurr * ansurr_loss + 
                      config.w_reg * reg_loss)
        return total_loss

    @eqx.filter_jit
    def make_step(model, opt_state, node_features, adj, edge_features, init_coords, lengths, angles, dihedrals, res_indices, target_saxs, target_shifts, q_values, form_factors):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            model, node_features, adj, edge_features, init_coords, lengths, angles, dihedrals, res_indices, target_saxs, target_shifts, q_values, form_factors
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # 5. Training Loop
    print(f"Starting training for {config.n_steps} steps...")
    for step in range(config.n_steps):
        model, opt_state, loss = make_step(
            model, opt_state, node_features, adj, edge_features, 
            data['init_coords'], data['lengths'], data['angles'], data['dihedrals'], data['res_indices'],
            target_saxs, target_shifts, q_values, form_factors
        )
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss:.6f}")
            
    print("Training complete.")
    
    # Save final structure
    final_deltas = model(node_features, adj, edge_features)
    final_coords, _ = rebuild_backbone(
        data['init_coords'],
        data['lengths'],
        data['angles'],
        data['dihedrals'],
        final_deltas
    )
    
    import biotite.structure as stripe
    import biotite.structure.io.pdb as pdb
    import numpy as np
    
    # Create Biotite structure from final_coords
    atoms = []
    res_indices = data['res_indices']
    # We need to map residue indices back to names for the PDB
    idx_to_res = {i: name for i, name in enumerate(RESIDUE_TYPES)}
    
    for i in range(len(res_indices)):
        res_name = idx_to_res.get(int(res_indices[i]), 'ALA')
        for j, name in enumerate(['N', 'CA', 'C']):
            idx = i * 3 + j
            atom = stripe.Atom(
                coord=np.array(final_coords[idx]),
                chain_id='A',
                res_id=i+1,
                res_name=res_name,
                atom_name=name,
                element='C' if name != 'N' else 'N'
            )
            atoms.append(atom)
            
    final_struct = stripe.array(atoms)
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(final_struct)
    pdb_file.write("refined_helix.pdb")
    print("Saved refined_helix.pdb")
    
    return model

if __name__ == "__main__":
    from torsiontuner.data import RESIDUE_TYPES # Needed for PDB saving refactor
    train()
