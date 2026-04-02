# CLAUDE.md — Contexto del Proyecto para Sesiones Futuras

Este archivo preserva el estado, las decisiones y el plan de trabajo del proyecto
para que no se pierda contexto entre sesiones de Claude Code.

---

## Descripción del Proyecto

**Título:** "Beyond the Rigid-Particle Model: Mobility Sign Change of Chiral Solitons
in a 1D Anisotropic Heisenberg Chain"

**Autor:** Felipe Wasaff — Departamento de Física, Universidad de Chile
**Magíster:** Simulación Computacional (primer semestre)
**Asignatura relacionada:** Programación Avanzada

**Hamiltoniano del sistema:**
```
H = -J Σ Sᵢ·Sᵢ₊₁  +  D Σ (Sᵢ×Sᵢ₊₁)·ẑ  +  Dₐ Σ (Sᵢᶻ)²
```
- J = 1.0 (intercambio ferromagnético, unidad de energía)
- D = 0.25 J (Dzyaloshinskii-Moriya)
- Da = -0.10 J (anisotropía de eje fácil)
- N = 200 sitios (cadena 1D con condiciones de frontera periódicas)

**Resultado principal:** La movilidad del solitón μ = dv/dh_z cambia de signo
al aumentar el amortiguamiento α, lo cual es incompatible con el modelo rígido
de Thiele. Esto evidencia deformaciones internas del solitón inducidas por la
disipación.

---

## Rama de Desarrollo

```
claude/peer-review-publication-3ZtPu
```

Todos los cambios van a esta rama. No pushear a main sin revisión.

---

## Arquitectura del Código (estado actual)

```
scripts/
├── core/
│   ├── __init__.py          ← exports de todo el módulo core
│   ├── exceptions.py        ← jerarquía de excepciones personalizadas
│   ├── decorators.py        ← @timer, @validate_spins, @log_simulation
│   ├── fields.py            ← ExternalField(ABC), GaussianPulse, ConstantField,
│   │                           CombinedField, ScaledField, nucleation_field()
│   └── llg_engine.py        ← HeisenbergChain (@dataclass), LLGSimulator (@dataclass)
│
├── runners/
│   └── run_mobility_scan.py ← barrido alpha×h_dc×realizaciones (generadores)
│
├── analysis/
│   ├── calculate_mobility.py ← SolitonTracker + pipeline de generadores
│   └── extract_velocity.py   ← análisis de velocidad (versión legacy)
│
└── visualization/
    ├── plot_mobility.py
    ├── plot_configurations.py
    ├── plot_dynamics.py
    └── plot_phase_examples.py
```

### Conceptos de Programación Avanzada implementados

| Concepto | Dónde | Para qué |
|---|---|---|
| `abc.ABC` + `@abstractmethod` | `fields.py` | Contrato de interfaz para campos externos |
| `@dataclass` + `__post_init__` | `llg_engine.py` | Genera __init__/__repr__, valida parámetros |
| `frozen=True, slots=True` | `GaussianPulse`, `ConstantField` | Inmutabilidad + eficiencia de memoria |
| Sobrecarga `__add__`, `__mul__` | `ExternalField` | Composición algebraica de campos |
| Excepciones personalizadas | `exceptions.py` | Errores de dominio físico |
| `@timer`, `@validate_spins` | `decorators.py` | Validación y logging automático |
| Generadores (`yield`) | `run_mobility_scan.py`, `calculate_mobility.py` | Lazy loading, pipeline de datos |
| Metaprogramación | Todo lo anterior | Código robusto para investigación |

---

## Críticas del Árbitro y Estado

Revista objetivo: **New Journal of Physics** (IOP, acceso abierto, gratuito bajo acuerdo ANID-IOP)

| ID | Crítica | Severidad | Estado |
|----|---------|-----------|--------|
| C1 | Sin análisis de tamaño finito (solo N=200) | CRÍTICA | Pendiente FASE 2 |
| C2 | Clasificación de fases ad hoc (umbrales sin justificar) | CRÍTICA | **Parcialmente saneada** — `classify_phase()` ahora usa factor de estructura S(k) + ratio AC/DC |
| C3 | Protocolo de nucleación no justificado | CRÍTICA | **Parcialmente saneada** — `GaussianPulse` ABC explícita, falta estudio de sensibilidad |
| C4 | Estadística insuficiente en zona crítica α≈0.04 | CRÍTICA | **Saneada en FASE 0** — 20 realizaciones en α∈[0.02,0.08] vs 5 en resto |
| C5 | Sin comparación con predicción de Thiele | CRÍTICA | Pendiente FASE 5 |
| C6 | Solo un punto en espacio de parámetros | MENOR | Pendiente FASE 6 |

---

## Plan de Fases (COMPROMETIDO)

### FASE 0 — Preparación ✅ COMPLETADA
- [x] `classify_phase()` con factor de estructura S(k) y ratio AC/DC
- [x] `compute_structure_factor()` y `phase_diagnostics()` en HeisenbergChain
- [x] `initial_fm_state_with_noise()` en LLGSimulator
- [x] `run_mobility_scan.py` con `n_realizations` adaptativo (5 general, 20 crítica)
- [x] `calculate_mobility.py` promedia realizaciones y propaga error → μ ± σ
- [x] `CLAUDE.md` creado

### FASE 1 — Reproducción del resultado base
**Objetivo:** Verificar que el resultado original (cambio de signo de μ) se reproduce
con el código refactorizado antes de ampliar la investigación.

**Scripts a ejecutar:**
```bash
python scripts/runners/run_mobility_scan.py
python scripts/analysis/calculate_mobility.py
```

**Criterio de éxito:** Curva μ(α) muestra cambio de signo en α ≈ 0.04,
con barras de error que no cruzan μ=0 antes del cruce real.

**Parámetros fijos:** N=200, D/J=0.25, Da/J=-0.10

**Archivos de salida:** `datos_barrido_mu/datos_a*.npz`, `mu_vs_alpha_data.npz`

### FASE 2 — Análisis de tamaño finito (responde C1)
**Objetivo:** Mostrar que μ(α) converge al aumentar N.

**Script nuevo a crear:** `scripts/runners/run_size_scan.py`

**Qué hace:**
- Corre el mismo barrido alpha×h_dc para N = 200, 400, 800
- Guarda en directorios separados: `datos_N200/`, `datos_N400/`, `datos_N800/`
- `calculate_mobility.py` se ejecuta tres veces (o acepta `data_dir` como argumento)

**Figura nueva:** μ(α) para los 3 tamaños en el mismo panel → convergencia visual

**Criterio de éxito:** Las curvas de N=400 y N=800 coinciden dentro de barras de error.

### FASE 3 — Robustez de nucleación (responde C3)
**Objetivo:** Mostrar que μ(α) no depende de h0 ni σ del pulso de nucleación.

**Script nuevo a crear:** `scripts/runners/run_nucleation_test.py`

**Qué hace:**
- Fija α=0.05 (en la zona del cambio de signo)
- Barre: h0 ∈ {-5J, -10J, -20J} y σ ∈ {2, 3, 5} sitios
- Mide μ para cada combinación

**Figura nueva:** Tabla/gráfico de μ vs parámetros del pulso → μ es robusto

**Criterio de éxito:** Variación de μ < 10% para los distintos parámetros del pulso.

### FASE 4 — Validación de clasificación de fases (responde C2)
**Objetivo:** Mostrar el factor de estructura S(k) para cada fase y comparar k_peak
con la predicción teórica del vector de onda helicoidal k_H = arctan(D/J).

**Script nuevo a crear:** `scripts/visualization/plot_structure_factor.py`

**Qué muestra:**
- S(k) para un representante de cada fase (H, SL, FM)
- Línea vertical en k_H teórico para comparación
- Texto con los diagnostics de `phase_diagnostics()`

**Figura nueva:** Panel 3×2: configuración de espines + S(k) para cada fase

### FASE 5 — Predicción de Thiele (responde C5)
**Objetivo:** Calcular μ_Thiele(α) analíticamente y mostrarlo junto a los datos.

**Script nuevo a crear:** `scripts/analysis/thiele_prediction.py`

**Física:**
La ecuación de Thiele para una partícula rígida predice:
```
μ_Thiele = G / (G² + D_Thiele² α²)
```
donde G es la carga topológica y D_Thiele es el tensor de disipación.
Para el modelo 1D con DMI: G y D_Thiele se calculan como integrales del perfil
del solitón en el estado fundamental.

**Figura modificada:** Fig. 4 con curva teórica de Thiele como línea discontinua

### FASE 6 — Exploración del espacio de parámetros (responde C6)
**Objetivo:** Mostrar que el cambio de signo de μ ocurre en toda la fase SL.

**Script nuevo a crear:** `scripts/runners/run_parameter_space.py`

**Qué hace:**
- Selecciona 4-6 puntos dentro de la fase SL del diagrama de fase
- Corre el barrido alpha×h_dc en cada punto
- Compara curvas μ(α) entre puntos

**Figura nueva:** Panel multi-curva μ(α) para diferentes (D/J, Da/J)

---

## Parámetros Físicos Clave

```python
# Punto de estudio principal (en la fase SL)
N  = 200          # tamaño de la cadena (a ampliar en FASE 2)
J  = 1.0          # energía de intercambio (unidad)
D  = 0.25 * J     # DMI
Da = -0.10 * J    # anisotropía eje fácil

# Barrido de dinámica
ALPHA_VALUES = np.linspace(0.01, 0.20, 20)   # amortiguamiento
HDC_VALUES   = np.linspace(-0.02, 0.02, 5)   # campo DC para μ = dv/dh

# Realizaciones (FASE 0)
N_REALIZATIONS_DEFAULT  = 5
N_REALIZATIONS_CRITICAL = 20    # en α ∈ [0.02, 0.08]
NOISE_AMPLITUDE         = 0.01  # ruido térmico en estado inicial

# Pulso de nucleación (GaussianPulse)
h0    = -10.0 * J
t0    = 2.0
tau   = 0.5
i0    = N // 2
sigma = 3.0

# Integración
T_MAX    = 200.0    # tiempo total (unidades J⁻¹ℏ)
DT_SAVE  = 0.5      # intervalo de muestreo
T_FIT    = (30, 150)  # ventana de ajuste de velocidad
```

---

## Cómo Reproducir el Entorno

```bash
# 1. Activar entorno (requiere numpy, scipy, matplotlib)
pip install numpy scipy matplotlib

# 2. Verificar instalación
python scripts/tests/test_installation.py

# 3. Ejecutar FASE 1
python scripts/runners/run_mobility_scan.py
python scripts/analysis/calculate_mobility.py

# 4. Generar figuras
python scripts/visualization/plot_mobility.py
```

---

## Notas de Asignatura (Programación Avanzada)

El código incorpora todos los conceptos del programa del magíster:

- **OOP:** `HeisenbergChain`, `LLGSimulator`, `SolitonTracker`
- **Excepciones:** jerarquía `MagneticSolitonError` con 7 subclases
- **Decoradores:** `@timer`, `@validate_spins`, `@log_simulation`, `@validate_parameters`
- **Generadores:** `parameter_grid()`, `simulation_jobs()`, `trajectory_loader()`, `velocity_results()`
- **Metaprogramación:** `abc.ABC`, `@dataclass(frozen, slots)`, `__post_init__`, sobrecarga de operadores

Cada concepto tiene comentarios explicativos en el código fuente.

---

## Historial de Commits Relevantes

```
53fc624  refactor: Add metaprogramming — ABC hierarchy and dataclasses
6f6a827  refactor: Apply advanced programming concepts to improve code quality
6a4c010  test: Add installation verification script
```

---

*Última actualización: FASE 0 completada. Próximo paso: ejecutar FASE 1.*
