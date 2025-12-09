# Core Simulation Engines

Fundamental simulation engines for magnetic soliton dynamics in 1D Heisenberg chains.

## Overview

This module provides the computational core for all simulations in the project:
- Landau-Lifshitz-Gilbert (LLG) equation integration
- Ground state phase diagram computation
- Single soliton dynamics simulation

## Files

### `llg_engine.py`
**Purpose:** Time evolution of classical spin systems

**Key Components:**
```python
class LLGSystem:
    def __init__(self, N, J, D, Da, alpha, gamma=1.0)
    def calculate_effective_field(self, spins)
    def integrate(self, initial_spins, t_span, hz=0.0)
```

**Features:**
- Adaptive RK45 integration (scipy.integrate.solve_ivp)
- Automatic spin normalization (|S| = 1)
- Exchange + DMI + Anisotropy interactions
- Periodic boundary conditions
- External field application

**Performance:**
- Single trajectory (N=200, T=200): ~30 seconds
- Memory: ~500 MB
- Parallelizable: Yes (multiple trajectories)

**Example:**
```python
from scripts.core.llg_engine import LLGSystem

system = LLGSystem(N=200, D=0.25, Da=-0.10, alpha=0.05)
spins_FM = system.ferromagnetic_state()
trajectory = system.integrate(spins_FM, t_span=(0, 200), hz=-0.01)
```

---

### `phase_diagram.py`
**Purpose:** Ground state phase diagram computation

**Algorithm:**
1. Initialize random spin configuration
2. Compute B_eff = -∂H/∂S
3. Update: dS/dt = S × B_eff (no damping)
4. Normalize: |S| = 1
5. Repeat until |ΔE/E| < 10⁻⁸

**Key Functions:**
```python
def compute_phase_diagram(D_range, Da_range, n_points=20)
def classify_phase(spins, energy)
def save_phase_data(phases, energies, filename)
```

**Output:**
- Phase classification map (H/SL/FM)
- Ground state energies
- Representative spin configurations

**Performance:**
- 20×20 grid: ~2 hours (8 cores parallel)
- Memory: ~4 GB
- Typical convergence: 500-2000 iterations

**Example:**
```python
from scripts.core.phase_diagram import compute_phase_diagram

phases, energies = compute_phase_diagram(
    D_range=(0.1, 1.0),
    Da_range=(0.0, -0.5),
    n_points=20
)
```

---

### `soliton_dynamics.py`
**Purpose:** Single soliton dynamics from metastable FM state

**Workflow:**
1. Initialize ferromagnetic state (Si = ẑ)
2. Apply Gaussian pulse: H_pulse(i,t) = H₀ exp[-...] x̂
3. Nucleate single soliton
4. Evolve under DC field hz
5. Track position vs time

**Pulse Parameters:**
- Amplitude: H₀ = -10.0 J
- Spatial width: σ = 3.0 sites
- Temporal width: τ = 0.5 J⁻¹ℏ
- Center: i₀ = N/2, t₀ = 2.0 J⁻¹ℏ

**Key Functions:**
```python
def simulate_soliton(D, Da, alpha, hz, N=200, T_max=200.0)
def apply_nucleation_pulse(spins, t, params)
def track_soliton_core(trajectory)
```

**Example:**
```python
from scripts.core.soliton_dynamics import simulate_soliton

result = simulate_soliton(
    D=0.25, Da=-0.10, 
    alpha=0.05, hz=-0.010
)
# Returns: trajectory with Sx, Sy, Sz vs (site, time)
```

---

## Physics Background

### Hamiltonian
```
H = -J Σᵢ Sᵢ·Sᵢ₊₁                    (Exchange)
  + D Σᵢ (Sᵢ × Sᵢ₊₁)·ẑ               (DMI)
  + Dₐ Σᵢ (Sᵢᶻ)²                     (Anisotropy)
  - Σᵢ hₑₓₜ · Sᵢ                     (External field)
```

### LLG Equation
```
dSᵢ/dt = -γ Sᵢ × Bₑff,ᵢ - αγ Sᵢ × (Sᵢ × Bₑff,ᵢ)
```

Where:
- γ: Gyromagnetic ratio (set to 1.0)
- α: Gilbert damping parameter
- Bₑff = -∂H/∂S: Effective field

### Phase Diagram Regions

**Helicoidal (H):** High DMI, low |Da|
- Uniform spiral: λ ≈ 2πJ/D
- ⟨Sˣ⟩, ⟨Sʸ⟩ ≠ 0, ⟨Sᶻ⟩ ≈ 0

**Soliton Lattice (SL):** Intermediate regime
- FM domains: Sᶻ ≈ ±1
- Chiral walls: width ~5 sites
- Periodic spacing: λ ~ 65 sites (for D/J=0.25)

**Ferromagnetic (FM):** High |Da|, low D
- Strong alignment: ⟨Sᶻ⟩ ≈ 0.98
- Metastable in SL region

---

## Units and Conventions

**Energy:** J (exchange constant)
**Time:** J⁻¹ℏ
**Length:** Lattice sites
**Spin:** Dimensionless, |S| = 1

**Typical Values:**
- J = 1.0 (energy unit)
- D/J = 0.1 - 1.0
- Da/J = 0.0 to -0.5
- α = 0.02 - 0.20

---

## Dependencies
```python
numpy >= 1.21      # Array operations
scipy >= 1.7       # ODE integration (solve_ivp)
matplotlib >= 3.4  # Visualization (optional)
```

---

## Testing
```bash
# Test LLG engine
python -m pytest scripts/tests/test_llg_engine.py

# Test phase diagram
python scripts/core/phase_diagram.py --test

# Quick simulation test
python scripts/core/soliton_dynamics.py --D 0.25 --Da -0.10 --alpha 0.05
```

---

## References

1. **LLG Equation:**
   - Landau & Lifshitz, Phys. Z. Sowjetunion 8, 153 (1935)
   - Gilbert, IEEE Trans. Magn. 40, 3443 (2004)

2. **Phase Diagram:**
   - Dmitriev & Krivnov, PRB 81, 054408 (2010)
   - Masaki & Stamps, PRB 95, 024418 (2017)

3. **Soliton Dynamics:**
   - Kim, J. Phys.: Condens. Matter 35, 345801 (2023)
   - Kishine & Ovchinnikov, Solid State Physics 66, 1 (2015)

---

## Notes

- All simulations use periodic boundary conditions
- Spin normalization enforced: |Sᵢ| = 1.0 ± 10⁻⁶
- Default integrator: RK45 (adaptive step, rtol=1e-6)
- Energy conservation checked in phase diagram
- Soliton core defined as region where Sᶻ < 0

---

*Last updated: December 2025*
