# --- run_mobility_scan.py ---
# Ejecuta el barrido completo de movilidad: alpha × h_dc × realizaciones.
#
# FASE 0 — Cambio clave respecto a la versión anterior:
#   Ahora soporta n_realizations por punto (alpha, h_dc).
#   Cada realización parte de un estado FM perturbado levemente (ruido térmico),
#   lo que genera estadística real para calcular barras de error en μ(α).
#
#   Estrategia de realizaciones:
#     - n_realizations = 5  en el rango general   (α fuera de [0.02, 0.08])
#     - n_realizations = 20 en la zona crítica     (α ∈ [0.02, 0.08])
#       → responde la crítica C4 del árbitro (estadística insuficiente en el cruce)
#
#   Nombres de archivo: datos_a{alpha}_h{hdc}_r{realization:02d}.npz
#   Ejemplo:           datos_a0p050_h0p010_r03.npz

import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.llg_engine import HeisenbergChain, LLGSimulator
from core.fields import nucleation_field
from core.exceptions import MagneticSolitonError


# =============================================================================
# Parámetros del sistema
# =============================================================================

N     = 200
J     = 1.0
D     = 0.25 * J
Da    = -0.10 * J
GAMMA = 1.0

# Barrido principal
ALPHA_VALUES = np.linspace(0.01, 0.20, 20)
HDC_VALUES   = np.linspace(-0.02, 0.02, 5)

# Realizaciones: más en la zona crítica del cruce de signo (FASE 0 — crítica C4)
N_REALIZATIONS_DEFAULT  = 5
N_REALIZATIONS_CRITICAL = 20
ALPHA_CRITICAL_LOW  = 0.02   # inicio de la zona crítica
ALPHA_CRITICAL_HIGH = 0.08   # fin de la zona crítica

# Amplitud del ruido térmico para las realizaciones independientes
NOISE_AMPLITUDE = 0.01  # 1% del módulo del espín — perturbación pequeña y física

# Simulación
T_MAX      = 200.0
DT_SAVE    = 0.5
OUTPUT_DIR = "datos_barrido_mu"


# =============================================================================
# Funciones auxiliares
# =============================================================================

def n_realizations_for(alpha: float) -> int:
    """
    Devuelve el número de realizaciones para un valor de alpha.
    Usa más realizaciones en la zona crítica donde ocurre el cambio de signo.
    """
    if ALPHA_CRITICAL_LOW <= alpha <= ALPHA_CRITICAL_HIGH:
        return N_REALIZATIONS_CRITICAL
    return N_REALIZATIONS_DEFAULT


def build_filepath(output_dir: str, alpha: float, h_dc: float, realization: int) -> str:
    """Construye el path del archivo incluyendo el índice de realización."""
    alpha_str = f"{alpha:.3f}".replace('.', 'p')
    hdc_str   = f"{h_dc:.3f}".replace('.', 'p').replace('-', 'm')
    return os.path.join(output_dir, f"datos_a{alpha_str}_h{hdc_str}_r{realization:02d}.npz")


# =============================================================================
# Generadores
# =============================================================================

def parameter_grid(alpha_values, hdc_values):
    """
    Generador perezoso de todas las combinaciones (alpha, h_dc).
    No construye la lista completa en memoria — yield entrega un par a la vez.
    """
    for alpha in alpha_values:
        for h_dc in hdc_values:
            yield alpha, h_dc


def simulation_jobs(alpha_values, hdc_values, output_dir, skip_existing=True):
    """
    Generador que produce los trabajos pendientes: (alpha, h_dc, realization).

    Automáticamente asigna más realizaciones en la zona crítica (α ∈ [0.02, 0.08])
    para responder la crítica C4 del árbitro.

    Yields
    ------
    tuple (int, int, float, float, int, str)
        (idx_actual, total, alpha, h_dc, realization, filepath)
    """
    # Pre-calcular el total para el progreso
    total = sum(
        n_realizations_for(a) * len(hdc_values)
        for a in alpha_values
    )

    idx = 0
    for alpha, h_dc in parameter_grid(alpha_values, hdc_values):
        n_real = n_realizations_for(alpha)
        for r in range(n_real):
            idx += 1
            filepath = build_filepath(output_dir, alpha, h_dc, r)
            if skip_existing and os.path.exists(filepath):
                print(f"  [skip] ({idx}/{total}) alpha={alpha:.3f}, h_dc={h_dc:.4f}, r={r:02d}")
                continue
            yield idx, total, alpha, h_dc, r, filepath


# =============================================================================
# Función principal del barrido
# =============================================================================

def run_scan(
    chain: HeisenbergChain,
    alpha_values=ALPHA_VALUES,
    hdc_values=HDC_VALUES,
    output_dir: str = OUTPUT_DIR,
    skip_existing: bool = True,
    noise_amplitude: float = NOISE_AMPLITUDE,
):
    """
    Ejecuta el barrido completo de movilidad con múltiples realizaciones.

    Para cada (alpha, h_dc, realization):
      1. Crea LLGSimulator con ese alpha
      2. Genera estado inicial FM + ruido (realization > 0) o FM puro (r=0)
      3. Aplica campo: GaussianPulse + ConstantField
      4. Integra LLG y guarda trayectoria con índice de realización

    La realización r=0 siempre usa el estado FM puro (reproducibilidad exacta).
    Las realizaciones r>0 usan ruido térmico aleatorio (estadística real).

    Parámetros
    ----------
    chain : HeisenbergChain
    alpha_values, hdc_values : array-like
    output_dir : str
    skip_existing : bool    Permite reanudar un barrido interrumpido.
    noise_amplitude : float Amplitud del ruido térmico (default 0.01).
    """
    os.makedirs(output_dir, exist_ok=True)

    total_sims = sum(
        n_realizations_for(a) * len(hdc_values)
        for a in alpha_values
    )
    n_critical = sum(1 for a in alpha_values
                     if ALPHA_CRITICAL_LOW <= a <= ALPHA_CRITICAL_HIGH)

    print("=" * 65)
    print("  BARRIDO COMPLETO DE MOVILIDAD — FASE 1")
    print(f"  {chain}")
    print(f"  Realizaciones (general):  {N_REALIZATIONS_DEFAULT}")
    print(f"  Realizaciones (α crítica [{ALPHA_CRITICAL_LOW},{ALPHA_CRITICAL_HIGH}]): "
          f"{N_REALIZATIONS_CRITICAL}  ({n_critical} valores de α)")
    print(f"  Total simulaciones:       {total_sims}")
    print(f"  Directorio de salida:     {output_dir}")
    print("=" * 65)

    for idx, total, alpha, h_dc, r, filepath in simulation_jobs(
        alpha_values, hdc_values, output_dir, skip_existing
    ):
        t_start = time.perf_counter()
        zone = " [CRÍTICA]" if ALPHA_CRITICAL_LOW <= alpha <= ALPHA_CRITICAL_HIGH else ""
        print(f"\n  ({idx}/{total}) alpha={alpha:.3f}{zone}, "
              f"h_dc/J={h_dc/J:.4f}, realización r={r:02d}")

        try:
            sim = LLGSimulator(chain, alpha=alpha, gamma=GAMMA)

            # r=0 → estado puro para reproducibilidad; r>0 → con ruido térmico
            if r == 0:
                S0 = sim.initial_fm_state()
            else:
                S0 = sim.initial_fm_state_with_noise(noise_amplitude)

            campo = nucleation_field(N=chain.N, J=chain.J, h_dc=h_dc)

            sol = sim.run(
                S0=S0,
                t_span=(0.0, T_MAX),
                dt_save=DT_SAVE,
                external_field=campo,
            )

            np.savez_compressed(
                filepath,
                S_history=sol.y,
                time_points=sol.t,
                alpha=alpha,
                h_dc=h_dc,
                realization=r,
            )

            elapsed = time.perf_counter() - t_start
            print(f"  → Guardado en {os.path.basename(filepath)} ({elapsed:.1f} s)")

        except MagneticSolitonError as exc:
            print(f"  → [AVISO FÍSICO] {type(exc).__name__}: {exc}")
        except Exception as exc:
            print(f"  → [ERROR] {type(exc).__name__}: {exc}")

    print("\n" + "=" * 65)
    print("  BARRIDO FINALIZADO")
    print("=" * 65)


# =============================================================================
# Punto de entrada
# =============================================================================

if __name__ == "__main__":
    chain = HeisenbergChain(N=N, J=J, D=D, Da=Da)
    run_scan(chain)
