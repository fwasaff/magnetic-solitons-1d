# Magnetic Solitons in 1D Heisenberg Chains: Non-Rigid Dynamics

[![DOI](https://img.shields.io/badge/DOI-pending-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Research Article:** *Beyond the Rigid-Particle Model: Mobility Sign Change of Chiral Solitons in a 1D Anisotropic Heisenberg Chain*  
> **Authors:** Felipe Wasaff, [Mentor Name]  
> **Affiliation:** Departamento de Física, Universidad de Chile  
> **Status:** Under review at Physica B: Condensed Matter

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Reproducing Results](#reproducing-results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 🔬 Overview

This repository contains the complete computational framework for studying **chiral magnetic solitons** (1D skyrmions) in classical Heisenberg spin chains with Dzyaloshinskii-Moriya interaction (DMI) and easy-axis anisotropy.

### Research Highlights

We systematically investigate:
1. **Static phase diagram** resolution (H/SL/FM phases)
2. **Soliton mobility** as a function of Gilbert damping (α)
3. **Non-monotonic dynamics** including mobility sign change
4. **Failure of rigid-particle models** (Thiele equation)

### Physical System

**Hamiltonian:**
```
H = -J Σᵢ Sᵢ·Sᵢ₊₁ + D Σᵢ (Sᵢ × Sᵢ₊₁)·ẑ + Dₐ Σᵢ (Sᵢᶻ)²
```

**Key Parameters:**
- Exchange: J = 1.0 (energy unit)
- DMI: D/J = 0.1 - 1.0
- Anisotropy: Dₐ/J = 0.0 to -0.5
- System size: N = 200 spins (periodic BC)

---

## 🎯 Key Findings

### 1. Phase Diagram Clarification

![Phase Diagram](manuscript/figures/fig1_phase_diagram.png)

We resolve literature ambiguity by systematic energy minimization:
- **Helicoidal (H):** High DMI, low anisotropy
- **Soliton Lattice (SL):** Intermediate regime (true ground state)
- **Ferromagnetic (FM):** High anisotropy (metastable in most regime)

### 2. Mobility Sign Change

![Mobility vs Damping](manuscript/figures/fig4_mobility.png)

**Novel result:** Soliton mobility μ = dv/dhz exhibits:
- Positive mobility for α < 0.04
- **Sign change** near α ≈ 0.04
- Negative mobility for 0.04 < α < 0.16
- Large fluctuations for α > 0.16

**Physical interpretation:** Non-rigid soliton dynamics where damping induces internal deformations affecting both gyrotropic and dissipative forces.

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- NumPy, SciPy, Matplotlib
- (Optional) Jupyter for notebooks
- (Optional) FFmpeg for animations

### Option 1: Using conda (Recommended)
```bash
# Clone repository
git clone https://github.com/fwasaff/magnetic-solitons-1d.git
cd magnetic-solitons-1d

# Create environment
conda env create -f environment.yml
conda activate mag-solitons

# Verify installation
python scripts/tests/test_installation.py
```

### Option 2: Using pip
```bash
# Clone repository
git clone https://github.com/fwasaff/magnetic-solitons-1d.git
cd magnetic-solitons-1d

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/tests/test_installation.py
```

---

## ⚡ Quick Start

### Example 1: Phase Diagram Calculation
```python
from scripts.core.phase_diagram import compute_phase_diagram

# Compute 20x20 grid
phases, energies = compute_phase_diagram(
    D_range=(0.1, 1.0),
    Da_range=(0.0, -0.5),
    n_points=20
)

# Visualize
from scripts.visualization.plot_phase_diagram import plot_phase_diagram
plot_phase_diagram(phases, save_path='results/my_phase_diagram.png')
```

### Example 2: Single Soliton Dynamics
```python
from scripts.core.soliton_dynamics import simulate_soliton

# Parameters
params = {
    'D': 0.25,
    'Da': -0.10,
    'alpha': 0.05,
    'hz': -0.010
}

# Run simulation
trajectory = simulate_soliton(**params)

# Extract velocity
from scripts.analysis.extract_velocity import fit_velocity
velocity, error = fit_velocity(trajectory, fit_window=(30, 150))
print(f"Velocity: {velocity:.3f} ± {error:.3f} sites/(J⁻¹ℏ)")
```

### Example 3: Mobility Calculation
```bash
# Full mobility analysis for one damping value
python scripts/analysis/calculate_mobility.py \
    --alpha 0.05 \
    --hz-range -0.02 0.02 \
    --n-fields 5 \
    --output results/mobility_alpha0.05.npz
```

---

## 📁 Repository Structure
```
magnetic-solitons-1d/
├── scripts/
│   ├── core/                    # Core simulation engines
│   │   ├── hamiltonian.py       # Energy and field calculations
│   │   ├── llg_integrator.py    # LLG equation solver
│   │   ├── phase_diagram.py     # Ground state finder
│   │   └── soliton_dynamics.py  # Dynamics simulation
│   ├── analysis/                # Analysis tools
│   │   ├── extract_velocity.py  # Velocity measurement
│   │   ├── calculate_mobility.py # Mobility computation
│   │   ├── transient_analysis.py # [CRITICAL 1]
│   │   ├── error_propagation.py  # [CRITICAL 2]
│   │   └── coherence_metrics.py  # [CRITICAL 2]
│   ├── visualization/           # Plotting utilities
│   │   ├── plot_phase_diagram.py
│   │   ├── plot_configurations.py
│   │   ├── plot_spatiotemporal.py
│   │   └── plot_mobility.py
│   └── tests/                   # Unit tests
│       ├── test_installation.py
│       ├── test_hamiltonian.py
│       └── test_llg_solver.py
├── notebooks/
│   ├── 01_phase_diagram_tutorial.ipynb
│   ├── 02_soliton_dynamics_demo.ipynb
│   ├── 03_mobility_analysis.ipynb
│   └── 04_figure_reproduction.ipynb
├── data/
│   ├── raw/                     # Raw simulation output
│   ├── processed/               # Analyzed data
│   └── figures/                 # Generated figures
├── manuscript/
│   ├── main.tex                 # LaTeX source
│   ├── figures/                 # Paper figures
│   └── supplementary/           # Supplementary material
├── docs/
│   ├── METHODS.md               # Detailed methodology
│   ├── REPRODUCIBILITY.md       # Reproduction guide
│   └── API_REFERENCE.md         # Code documentation
├── results/                     # Final results
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
└── CITATION.cff                 # Citation metadata
```

---

## 🔄 Reproducing Results

### Full Reproduction Pipeline
```bash
# 1. Compute phase diagram (Fig. 1, ~2 hours)
python scripts/core/phase_diagram.py --config configs/phase_diagram.yaml

# 2. Generate ground state configurations (Fig. 2, ~10 min)
python scripts/analysis/extract_configurations.py

# 3. Run mobility analysis (Fig. 3-4, ~24 hours on 8 cores)
bash scripts/run_full_mobility_scan.sh

# 4. Generate all figures
python scripts/visualization/generate_all_figures.py

# 5. Compile manuscript
cd manuscript && pdflatex main.tex && bibtex main && pdflatex main.tex
```

### Computational Requirements

**Minimal (testing):**
- 4 CPU cores
- 8 GB RAM
- ~2 GB disk space
- Time: ~4 hours

**Full reproduction:**
- 8+ CPU cores (recommended)
- 16 GB RAM
- ~50 GB disk space
- Time: ~30 hours

### Pre-computed Data

For quick figure generation without running full simulations:
```bash
# Download pre-computed dataset (~5 GB)
wget https://zenodo.org/record/XXXXX/magnetic_solitons_data.tar.gz
tar -xzf magnetic_solitons_data.tar.gz -C data/processed/

# Generate figures from pre-computed data
python scripts/visualization/generate_all_figures.py --use-precomputed
```

---

## 📊 Data Availability

**Raw simulation data** and **processed results** will be deposited at Zenodo upon publication:
- Phase diagram data
- Complete mobility dataset (19 α × 5 hz × 5 runs = 475 simulations)
- Spatiotemporal trajectories
- Analysis outputs

**Estimated dataset size:** ~5 GB compressed

---

## 📖 Citation

If you use this code or data in your research, please cite:
```bibtex
@article{Wasaff2025Solitons,
  title={Beyond the Rigid-Particle Model: Mobility Sign Change of Chiral Solitons in a 1D Anisotropic Heisenberg Chain},
  author={Wasaff, Felipe and [Mentor Name]},
  journal={Physica B: Condensed Matter},
  year={2025},
  volume={XXX},
  pages={XXX},
  doi={10.1016/j.physb.2025.XXXXX}
}
```

**Software citation:**
```bibtex
@software{Wasaff2025Code,
  author={Wasaff, Felipe},
  title={magnetic-solitons-1d: Computational Framework for 1D Magnetic Solitons},
  year={2025},
  publisher={GitHub},
  url={https://github.com/fwasaff/magnetic-solitons-1d},
  version={1.0.0}
}
```

---

## 🤝 Contributing

This is research code associated with a specific publication. While we welcome:
- Bug reports (use GitHub Issues)
- Clarification questions (use Discussions)
- Suggestions for improvements

Please note that substantial modifications would constitute derivative work. If you're interested in extending this research, please contact us.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Note:** The manuscript content in `manuscript/` is © 2025 by the authors and has separate copyright restrictions.

---

## 👤 Contact

**Felipe Wasaff**
- 🏛️ Physics Coordinator, Universidad de Chile
- 📧 felipe.wasaff@uchile.cl
- 🐙 GitHub: [@fwasaff](https://github.com/fwasaff)
- 💼 LinkedIn: [felipe-wasaff](https://linkedin.com/in/felipe-wasaff)

**Supervisor:** [Mentor Name]
- 📧 [email@uchile.cl]

---

## 🙏 Acknowledgments

- **Funding:** [FADOP 2025, ANID, etc.]
- **Computational Resources:** [If used cluster]
- **Theoretical Discussions:** [Collaborators]
- **Software:** This project uses NumPy, SciPy, Matplotlib

---

## 📚 Related Publications

1. Wasaff et al., "Previous related work", Journal (2024)
2. [Other relevant papers from your group]

---

## 🔗 Useful Links

- [Manuscript preprint (arXiv)](https://arxiv.org/)
- [Supplementary Material](docs/SUPPLEMENTARY.md)
- [Dataset (Zenodo)](https://zenodo.org/)
- [Research Group Website](https://fisica.uchile.cl/)

---

*Last updated: December 2025*
