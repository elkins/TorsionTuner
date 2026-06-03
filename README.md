# 🧬 TorsionTuner: GNN-Based Protein Structure Refinement

[![Tests](https://github.com/elkins/TorsionTuner/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/TorsionTuner/actions/workflows/test.yml)
[![Lint](https://github.com/elkins/TorsionTuner/actions/workflows/lint.yml/badge.svg)](https://github.com/elkins/TorsionTuner/actions/workflows/lint.yml)
[![Docs](https://github.com/elkins/TorsionTuner/actions/workflows/docs.yml/badge.svg)](https://github.com/elkins/TorsionTuner/actions/workflows/docs.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](https://mypy-lang.org/)

**Documentation:** [https://elkins.github.io/TorsionTuner/](https://elkins.github.io/TorsionTuner/)

---

## 🌟 Why TorsionTuner?

Static snapshots of proteins, such as those from AlphaFold, often miss the subtle dynamic nuances of molecules in their natural, solution-state environments.  **TorsionTuner** bridges this gap. It is a specialized machine learning engine that "nudges" idealized structures into better agreement with real-world experimental data.

### The Problem
Traditional refinement often breaks the laws of chemistry -— bond lengths stretch and angles distort -— just to fit noisy data.

### The TorsionTuner Solution
By operating exclusively in **torsional space** ($\phi/\psi$ angles), we ensure the laws of physics are respected. Our differentiable kinematics layer allows gradients to flow from the experimental loss (SAXS/NMR) directly back into the GNN weights, creating a chemically valid, evidence-based refinement.

---

## ✨ Core Features

*   **🛰️ Differentiable Kinematics**: Powered by a JAX-native implementation of the Natural Extension Reference Frame (NeRF) algorithm.
*   **🧠 Geometric GNN**: An Equinox-based Graph Neural Network that captures both sequential (backbone) and spatial (3D contact) relationships.
*   **⚖️ Multi-Objective Optimization**: Simultaneously fits SAXS profiles, backbone chemical shifts, and structural geometry (Ramachandran regularization).
*   **🧪 Evidence-Based**: Rooted in refinement strategies pioneered by the **Montelione Group**.

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/elkins/TorsionTuner.git
cd TorsionTuner
pip install -e .
```

### 2. Refine a Structure
```bash
# Generate a sample helix using synth-pdb
synth-pdb --length 20 --conformation alpha --output test_helix.pdb

# Run the refinement pipeline
python -m torsiontuner.train
```

---

## 🛠 Software Architecture

The project is structured for transparency and modularity:

*   **`torsiontuner/data.py`**: PDB loading and graph construction.
*   **`torsiontuner/model.py`**: The GNN architecture and message-passing layers.
*   **`torsiontuner/kinematics.py`**: The differentiable bridge between angles and 3D space.
*   **`torsiontuner/montelione_utils.py`**: Chemical shift losses and quality proxies.
*   **`torsiontuner/train.py`**: The training orchestration and optimization loop.

---

## 🏛 Research Integration

This project implements refinement strategies for integrating AI-predicted structures with NMR data.

*   **Chemical Shift Driven Refinement:** Using $C_\alpha$ shift prediction to improve AlphaFold model accuracy.
*   **Scientific Validation:** Rigorous benchmarking against the **NESG "Blind" Dataset**. See our [Scientific Validation Roadmap](docs/SCIENTIFIC_VALIDATION.md) for more details.

---

## 📚 Glossary

*   **Dihedral Angles ($\phi, \psi$):** The rotation angles of the protein backbone that define its overall 3D fold.
*   **NeRF (Natural Extension Reference Frame):** An algorithm used to convert internal coordinates (angles/lengths) into 3D Cartesian coordinates.
*   **CSRMSD:** Chemical Shift Root-Mean-Square Deviation—a measure of how well a structure fits experimental NMR data.
*   **Ramachandran Regularization:** A soft potential penalizing phi/psi values outside the favored backbone geometry regions (alpha-helix, beta-strand, left-handed alpha). Used as a training-time regularizer; analogous to the Ramachandran terms in CNS and Rosetta. For post-refinement backbone quality assessment, see PROCHECK or MolProbity. For NMR-specific structure accuracy validation, see ANSURR (Fowler et al. 2020, *Nature Commun.*).
*   **SAXS (Small-Angle X-ray Scattering):** A technique that provides information on the overall shape, size, and dynamics of proteins in solution.

---

## 📖 Key References
*   **RPF Scores:** Huang, Y. J., et al. (2005). *J. Am. Chem. Soc.*, 127(5), 1665–1674.
*   **Rosetta Refinement:** Mao, B., et al. (2014). *J. Am. Chem. Soc.*, 136(5), 1893–1906.
*   **AlphaFold-NMR Assessment:** Li, E. H., et al. (2023). *J. Magn. Reson.*, 352, 107481.
*   **Debye Formula for SAXS:** Debye, P. (1915). *Annalen der Physik*, 351(6), 809-876.

---

## 🤝 Contributing & Support

We welcome contributions from both the Machine Learning and Structural Biology communities!
*   **Bugs/Features:** Please open an issue.
*   **Questions:** Visit our [Documentation](https://elkins.github.io/TorsionTuner/) or reach out via GitHub Discussions.

---

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Related Projects

TorsionTuner depends on and integrates with:

- [diff-biophys](https://github.com/elkins/diff-biophys) — JAX differentiable kernels for SAXS and NMR losses
- [synth-pdb](https://github.com/elkins/synth-pdb) — Synthetic PDB generation for training and validation
- [synth-nmr](https://github.com/elkins/synth-nmr) — Chemical shift prediction and NMR observables
- [synth-saxs](https://github.com/elkins/synth-saxs) — SAXS profile simulation for loss computation
- [diff-ensemble](https://github.com/elkins/diff-ensemble) — Ensemble-level counterpart using a VAE architecture

---

## 📖 Citation

```bibtex
@software{torsiontuner,
  author  = {Elkins, George},
  title   = {TorsionTuner: GNN-based protein structure refinement in torsional space},
  year    = {2024},
  url     = {https://github.com/elkins/TorsionTuner},
  version = {0.1.0}
}
```
