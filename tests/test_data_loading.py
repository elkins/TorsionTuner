from torsiontuner.data import get_graph_features, load_pdb


def test_load_pdb():
    data = load_pdb("test_helix.pdb")
    assert "coords" in data
    assert "res_indices" in data
    assert "elements" in data
    assert len(data["res_indices"]) == 20
    assert data["coords"].shape == (60, 3)  # 20 residues * 3 atoms
    assert len(data["elements"]) == 60


def test_graph_features():
    data = load_pdb("test_helix.pdb")
    node_features, adj, edge_features = get_graph_features(data)

    n_residues = 20
    assert node_features.shape == (n_residues, 20)
    assert adj.shape == (n_residues, n_residues)
    assert edge_features.shape == (n_residues, n_residues, 1)

    # Check spatial adjacency
    # In a helix, residue i and i+4 should be close (approx 5.4A)
    # 10A cutoff should definitely include i and i+1, i+2, i+3, i+4
    assert adj[0, 1] == 1.0
    assert adj[0, 4] == 1.0


if __name__ == "__main__":
    test_load_pdb()
    test_graph_features()
    print("Data loading tests passed!")
