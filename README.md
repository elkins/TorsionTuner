# Experimental Fine-Tuner: GNN-Based Protein Structure Refinement

The **Experimental Fine-Tuner** is a specialized machine learning tool designed to bridge the gap between "idealized" static protein structures (like those from AlphaFold) and "dynamic" solution-state experimental data (SAXS/NMR). 

By utilizing a JAX-based Graph Neural Network (GNN) and a differentiable kinematics layer, the model applies subtle adjustments to the backbone dihedral angles ($\phi/\psi$) to minimize the discrepancy between the predicted structure and real experimental observables.

---

## 🔬 Theoretical Foundations

### 1. The Torsional Prediction Strategy
Instead of predicting 3D coordinates ($x, y, z$) directly—which can easily break chemical constraints like bond lengths—this model predicts **torsional deltas** ($\Delta\phi, \Delta\psi$). 
*   **Advantage:** By keeping bond lengths and bond angles fixed to their idealized values, we ensure the resulting structure is always chemically valid.
*   **Input:** Initial backbone angles from an AlphaFold PDB.
*   **Output:** Small angular adjustments that "nudge" the protein into a conformation that fits the data.

### 2. Differentiable Kinematics (NeRF)
To calculate a loss against experimental data, we convert the predicted angles back into 3D coordinates using the **Natural Extension Reference Frame (NeRF)** algorithm, implemented in JAX.
*   **Forward Pass:** Angles $\rightarrow$ 3D Coordinates $\rightarrow$ Simulated Observables (SAXS/NMR).
*   **Backward Pass:** Gradients flow from the experimental loss back through the kinematics layer to the GNN weights.

### 3. Multi-Objective Loss Function
The model optimizes a weighted combination of multiple experimental and structural constraints:
$$\mathcal{L}_{total} = w_{saxs}\mathcal{L}_{saxs} + w_{nmr}\mathcal{L}_{nmr} + w_{quality}\mathcal{L}_{quality} + w_{reg}\mathcal{L}_{reg}$$

*   **SAXS Loss:** Uses the Debye formula to fit Small-Angle X-ray Scattering profiles.
*   **NMR Loss:** Fits experimental $C_\alpha$ chemical shifts using a differentiable predictor.
*   **Quality Loss (ANSURR Proxy):** A soft Ramachandran penalty that ensures $\phi/\psi$ angles remain in favored regions.
*   **Regularization:** Prevents over-fitting by keeping the angular "nudges" small.

---

## 🛠 Software Architecture

*   **`src/data.py`**: PDB loading and graph construction using `Biotite`.
*   **`src/model.py`**: An **Equinox** (JAX-native) GNN with message-passing layers.
*   **`src/kinematics.py`**: The JAX-differentiable bridge between angles and 3D space.
*   **`src/montelione_utils.py`**: Implementation of chemical shift losses and structural quality proxies.
*   **`src/train.py`**: The orchestration layer using **Optax** for optimization.

---

## 🚀 Getting Started

### 1. Installation
```bash
# Install core dependencies
pip install -r requirements.txt
```

### 2. Running a Refinement
```bash
# 1. Generate a synthetic "target" helix
python generate_test_pdb.py

# 2. Run the multi-objective fine-tuning optimization
PYTHONPATH=. python src/train.py
```

---

## 🏛 Research Integration (The Montelione Group Approach)
This project implements refinement strategies pioneered by the **Montelione Group** (RPI/Rutgers) for integrating AI-predicted structures with NMR data.

### 1. Chemical Shift Driven Refinement
Research shows that AlphaFold models often rival the accuracy of experimental NMR structures but can be further improved by fitting to backbone chemical shifts.
*   **Implementation:** We use differentiable $C_\alpha$ shift prediction to "nudge" models towards experimental accuracy.
*   **Reference:** *Tejero R, et al. (2022). "AlphaFold models of small proteins rival the accuracy of solution NMR structures." Frontiers in Molecular Biosciences.*

### 2. Validation with RPF-DP and ANSURR
The Montelione lab advocates for rigorous validation of refined models using "goodness-of-fit" metrics.
*   **RPF-DP:** Measures how well the 3D model fits unassigned NOESY peak lists (Recall, Precision, F-measure, and Discrimination Power).
*   **ANSURR:** Validates accuracy by comparing flexibility predicted from chemical shifts (RCI) with flexibility from the 3D structure.
*   **Reference:** *Huang YJ, et al. (2012). "RPF: a quality assessment tool for protein NMR structures." Nucleic Acids Research.*
*   **Reference:** *Fowler NJ, et al. (2020). "A method for validating the accuracy of NMR protein structures." Nature Communications.*

### 3. PSVS Suite
Refined structures should be validated using the **Protein Structure Validation Software (PSVS)** suite.
*   **Utility:** Confirms that the refinement has improved the "NMR-quality" of the prediction.
*   **Link:** [PSVS Server](http://psvs-1_5.nesg.org/)

---

## 📖 Key References
*   **RPF Scores:** Huang, Y. J., et al. (2005). *J. Am. Chem. Soc.*, 127(5), 1665–1674.
*   **Rosetta Refinement:** Mao, B., et al. (2014). *J. Am. Chem. Soc.*, 136(5), 1893–1906.
*   **AlphaFold-NMR Assessment:** Li, E. H., et al. (2023). *J. Magn. Reson.*, 352, 107481.
*   **Debye Formula for SAXS:** Debye, P. (1915). *Annalen der Physik*, 351(6), 809-876.
