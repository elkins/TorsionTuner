import jax
import jax.numpy as jnp
import jax.random as jr
from torsiontuner.model import FineTunerGNN

def test_model_forward():
    key = jr.PRNGKey(0)
    n_nodes = 10
    node_dim = 20
    hidden_dim = 32
    out_dim = 2
    n_layers = 2
    
    model = FineTunerGNN(node_dim, hidden_dim, out_dim, n_layers, key)
    
    x = jr.normal(key, (n_nodes, node_dim))
    adj = jnp.ones((n_nodes, n_nodes))
    edge_features = jr.normal(key, (n_nodes, n_nodes, 1))
    
    out = model(x, adj, edge_features)
    assert out.shape == (n_nodes, out_dim)
    assert not jnp.any(jnp.isnan(out))

if __name__ == "__main__":
    test_model_forward()
    print("Model tests passed!")
