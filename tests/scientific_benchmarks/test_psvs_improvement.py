from torsiontuner.psvs_parser import parse_psvs_summary

# Mock PSVS summary for raw AlphaFold model (2KHD)
MOCK_PSVS_BEFORE = """
PSVS 1.5 Analysis for AF-Q9KL30-v4
Verify3D (expected > -0.1): -0.15
PROCHECK G-factor (phi/psi) (expected > -0.5): -0.85
PROCHECK G-factor (all-atom) (expected > -0.5): -1.10
MolProbity clashscore (expected < 10): 12.4
"""

# Mock PSVS summary for TorsionTuner refined model
MOCK_PSVS_AFTER = """
PSVS 1.5 Analysis for refined_2khd
Verify3D (expected > -0.1): 0.22
PROCHECK G-factor (phi/psi) (expected > -0.5): -0.12
PROCHECK G-factor (all-atom) (expected > -0.5): -0.35
MolProbity clashscore (expected < 10): 4.1
"""


def test_psvs_metrics_improvement():
    """
    Verify that our parser correctly extracts metrics and can identify
    the expected improvements in a refined model.
    """
    before = parse_psvs_summary(MOCK_PSVS_BEFORE)
    after = parse_psvs_summary(MOCK_PSVS_AFTER)

    # Assertions for Verify3D (Higher is better)
    assert after["verify3d"] > before["verify3d"]
    assert after["verify3d"] > -0.1  # Now in favored range

    # Assertions for PROCHECK (Higher is better, usually > -0.5)
    assert after["procheck_phi_psi"] > before["procheck_phi_psi"]
    assert after["procheck_phi_psi"] > -0.5

    # Assertions for Clashscore (Lower is better)
    assert after["clashscore"] < before["clashscore"]
    assert after["clashscore"] < 10


def test_parser_partial_content():
    """Ensure the parser handles missing metrics gracefully."""
    partial_text = "Verify3D (expected > -0.1): 0.5"
    results = parse_psvs_summary(partial_text)
    assert results["verify3d"] == 0.5
    assert "clashscore" not in results
