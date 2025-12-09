# Manuscript Figures

Publication-ready figures for Physical Review B submission.

## Main Figures

### Figure 1: Phase Diagram
**File:** `fig1_phase_diagram.pdf`
**Source:** `figure2_phase_diagram_PRB.pdf`
**Description:** Zero-field ground state phase diagram (Da/J vs D/J)
- Blue: Helicoidal (H) phase
- Orange: Soliton Lattice (SL) phase  
- Green: Ferromagnetic (FM) phase
- Red star: Parameter point for dynamics study (D/J=0.25, Da/J=-0.10)

### Figure 2: Spin Configurations
**File:** `fig2_configurations.pdf`
**Source:** `figure3_phase_configurations_PRB.pdf`
**Description:** Representative ground state spin configurations
- **(a) Helicoidal:** Uniform spiral modulation, λ ≈ 9 sites
- **(b) Soliton Lattice:** FM domains separated by chiral walls
- **(c) Ferromagnetic:** Strong alignment ⟨Sz⟩ ≈ 0.98

### Figure 3: Methodology Validation
**File:** `fig3_methodology.pdf`
**Source:** `figure4_methodology_validation_PRB.pdf`
**Description:** Soliton mobility measurement methodology
- **(a) Spatiotemporal map:** Sz(x,t) showing steady-state propagation
  - Black line: soliton trajectory
  - Yellow region: fitting window (30-150 J⁻¹ℏ)
  - Linear fit: v = -0.274 sites/(J⁻¹ℏ)
- **(b) Velocity vs field:** v(hz) for three α values
  - Blue: α=0.02, μ = +3.86 ± 0.25 (positive mobility)
  - Orange: α=0.05, μ = -2.09 ± 0.23 (negative mobility)
  - Red: α=0.16, μ = -6.84 ± 0.73

### Figure 4: Mobility vs Damping (Main Result)
**File:** `fig4_mobility.pdf`
**Source:** `figure_5_FINAL_COMPOSITE_3-panel.pdf`
**Description:** ⭐ Key finding - non-monotonic mobility
- **(a) Intrinsic velocity:** vint(α) at hz=0
  - Abrupt transition near α ≈ 0.04
- **(b) Mobility:** μ(α) = dv/dhz
  - **Positive** for α < 0.04
  - **Sign change** at α ≈ 0.04
  - **Negative** for 0.04 < α < 0.16
  - Large fluctuations for α > 0.16
- **(c) Parametric plot:** μ vs vint
  - Multi-valued relationship
  - Discrete clustering → three dynamical regimes
  - Proof of non-rigid soliton dynamics

## Generation

Figures generated using:
```bash
python scripts/visualization/plot_configurations.py
python scripts/visualization/plot_methodology.py
python scripts/visualization/plot_mobility.py
```

Or regenerate all from manuscript:
```bash
cd manuscript
pdflatex main.tex
```

## Technical Specifications

- **Format:** PDF (vector graphics)
- **Resolution:** 600 DPI for raster elements
- **Size:** 3.4" (single column) or 7" (double column)
- **Font:** Computer Modern (LaTeX compatible)
- **Colormap:** Scientific (cmocean)

## Source Data

All figures are based on simulation data in:
- `data/processed/phase_diagram_data.npz`
- `data/processed/trajectories/*.npz`
- `data/processed/mobility_full_scan.npz`

## Citation

When using these figures, cite:
```
Wasaff, F. et al., "Beyond the Rigid-Particle Model: Mobility Sign 
Change of Chiral Solitons in a 1D Anisotropic Heisenberg Chain", 
Physica B: Condensed Matter (2025)
```

