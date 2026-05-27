import re


def parse_psvs_summary(text: str) -> dict:
    """
    Parse the text summary from a PSVS validation report to extract key metrics.

    Extracts:
        - Verify3D score
        - PROCHECK G-factors (phi/psi and all-atom)
        - MolProbity Clashscore

    Args:
        text: The content of the PSVS summary file.

    Returns:
        A dictionary of extracted metrics.
    """
    results = {}

    # Regex patterns for PSVS 1.5 standard output
    patterns = {
        "verify3d": r"Verify3D\s*\(expected\s*>\s*-?[\d\.]+\):\s*(-?[\d\.]+)",
        "procheck_phi_psi": (
            r"PROCHECK\s*G-factor\s*\(phi/psi\)\s*\(expected\s*>\s*-?[\d\.]+\):\s*(-?[\d\.]+)"
        ),
        "procheck_all": (
            r"PROCHECK\s*G-factor\s*\(all-atom\)\s*\(expected\s*>\s*-?[\d\.]+\):\s*(-?[\d\.]+)"
        ),
        "clashscore": (
            r"MolProbity\s*clashscore\s*\(expected\s*<\s*-?[\d\.]+\):\s*(-?[\d\.]+)"
        ),
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            results[key] = float(match.group(1))

    return results
