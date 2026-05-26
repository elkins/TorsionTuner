# Scientific Validation Roadmap

This document outlines the strategic plan for ensuring TorsionTuner remains an entirely evidence-based tool, leveraging published research and experimental benchmarks.

## 📋 Validation Initiatives

### 1. NESG "Blind" Benchmark Suite (Priority: High)
Refine and validate against the 9 proteins identified in the Montelione Lab's 2023 study (*J. Magn. Reson.*, 352, 107481).
*   **Goal:** Demonstrate that TorsionTuner consistently reduces Chemical Shift RMSD (CSRMSD) for structures AlphaFold has never seen.
*   **Targets:** 2KHD, 2KIW, 2KZV, 2RN7, 2L9R, 2LCI, 2M5O, 2M2L, 2M5Q.

### 2. Scientific Parity Tests (Priority: Medium)
Verify that internal predictors (Chemical Shifts, SAXS) match established gold-standard software.
*   **Goal:** Confirm parity with SPARTA+, ShiftX2, and Crysol.
*   **Implementation:** Automated tests comparing TorsionTuner outputs with pre-computed outputs from these tools.

## 🛠 Active Task: Item 3 - Torsional Regularization
*   **Status:** Completed
*   **Accomplishments:**
    *   Developed a rigorous "recovery" benchmark in `tests/scientific_benchmarks/test_regularization.py`.
    *   Empirically proved that TorsionTuner can restore physical realism to structures with significant Ramachandran violations.
    *   Demonstrated that the multi-objective loss function (ANSURR proxy + NMR) successfully overcomes random initialization noise to "fix" unphysical starting points.

### 4. PSVS Integration (Priority: Low)
Validate the "NMR-quality" of refined models using the Protein Structure Validation Software suite.
*   **Goal:** Confirm improvement in Verify3D and Procheck G-factors.
*   **Implementation:** Export scripts and automated validation report analysis.

## 🛠 Active Task: Item 1 - NESG "Blind" Benchmark Suite
*   **Status:** In Progress
*   **Accomplishments:**
    *   Successfully implemented the first real-world benchmark for **2KHD** (NESG VC_A0919).
    *   Demonstrated that TorsionTuner reduces CSRMSD from **0.4304 ppm** (raw AlphaFold) to **0.2873 ppm** against experimental BMRB 16238 data.
    *   Automated the 2KHD validation in `tests/scientific_benchmarks/test_2khd.py` using the actual AF2 model.
*   **Next Steps:**
    *   Expand to the remaining 8 targets (2KIW, 2KZV, etc.).
    *   Consolidate benchmark results into a scientific report or dashboard.

## 🛠 Active Task: Item 2 - Scientific Parity Tests
*   **Status:** In Progress
*   **Completed:**
    *   Analytical parity verification for the Debye SAXS formula.
    *   CSI (Chemical Shift Index) trend verification for Alpha-helix vs. Beta-strand contexts (matching SPARTA+/ShiftX2 expectations).
*   **Next Steps:**
    *   Expand to full-protein SPARTA+ parity using pre-computed outputs.
    *   Implement RDC (Residual Dipolar Coupling) parity checks.
