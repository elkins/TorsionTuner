import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class GNNLayer(eqx.Module):
    """
    A single Graph Neural Network layer that processes node and edge features.

    This layer uses message passing to update node representations by aggregating
    information from neighbors, weighted by edge features (e.g., distances).
    """

    lin_node: eqx.nn.Linear
    lin_edge: eqx.nn.Linear
    lin_out: eqx.nn.Linear

    def __init__(self, node_dim, out_dim, key):
        keys = jr.split(key, 3)
        self.lin_node = eqx.nn.Linear(node_dim, out_dim, key=keys[0])
        self.lin_edge = eqx.nn.Linear(
            1, out_dim, key=keys[1]
        )  # 1D edge feature: distance
        self.lin_out = eqx.nn.Linear(out_dim, out_dim, key=keys[2])

    def __call__(
        self, x: jnp.ndarray, adj: jnp.ndarray, edge_features: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Forward pass of the GNN layer.

        Args:
            x: Node features of shape (n_nodes, node_dim).
            adj: Adjacency matrix of shape (n_nodes, n_nodes).
            edge_features: Edge features of shape (n_nodes, n_nodes, 1).

        Returns:
            Updated node features of shape (n_nodes, out_dim).
        """
        h = jax.vmap(self.lin_node)(x)

        # Message passing with edge features
        # We can apply a linear transformation to edge features and use them as weights
        # or add them to the node messages.
        # Here: m_i = sum_j (adj_ij * (h_j + lin_edge(e_ij)))

        # Project edge features
        e_proj = jax.vmap(jax.vmap(self.lin_edge))(edge_features)  # (n, n, out_dim)

        # Combine messages
        # h[None, :, :] has shape (1, n, out_dim)
        # e_proj has shape (n, n, out_dim)
        messages = (h[None, :, :] + e_proj) * adj[:, :, None]

        m = jnp.sum(messages, axis=1)

        # Update
        out = jax.nn.relu(h + m)
        return out


class FineTunerGNN(eqx.Module):
    """
    The main Graph Neural Network model for structural refinement.

    It consists of several GNN layers followed by an output head that predicts
    delta dihedrals for each residue.
    """

    layers: list
    output_head: eqx.nn.Linear

    def __init__(self, node_dim, hidden_dim, out_dim, n_layers, key):
        keys = jr.split(key, n_layers + 1)
        self.layers = []
        curr_dim = node_dim
        for i in range(n_layers):
            self.layers.append(GNNLayer(curr_dim, hidden_dim, key=keys[i]))
            curr_dim = hidden_dim

        self.output_head = eqx.nn.Linear(hidden_dim, out_dim, key=keys[-1])

    def __call__(
        self, x: jnp.ndarray, adj: jnp.ndarray, edge_features: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Predict structural adjustments.

        Args:
            x: Input node features (one-hot residue types).
            adj: Adjacency matrix (sequential + spatial).
            edge_features: Edge features (normalized distances).

        Returns:
            Predicted delta dihedrals (delta_phi, delta_psi) for each residue.
        """
        for layer in self.layers:
            x = layer(x, adj, edge_features)

        # Predict delta dihedrals
        out = jax.vmap(self.output_head)(x)
        return out
