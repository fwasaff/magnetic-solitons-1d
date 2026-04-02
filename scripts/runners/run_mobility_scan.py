# --- run_mobility_scan.py ---
# Ejecuta el barrido completo de movilidad: alpha × h_dc.
#
# Ahora usa la jerarquía ExternalField para construir los campos:
#   nucleation_field(N, J, h_dc) → GaussianPulse + ConstantField
#
# Generadores: parameter_grid() y simulation_jobs() entregan combinaciones
# de parámetros de forma perezosa, sin construir listas en memoria.

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

ALPHA_VALUES = np.linspace(0.01, 0.20, 20)
HDC_VALUES   = np.linspace(-0.02, 0.02, 5)

T_MAX      = 200.0
DT_SAVE    = 0.5
OUTPUT_DIR = "datos_barrido_mu"


# =============================================================================
# Generadores
# =============================================================================

def parameter_grid(alpha_values, hdc_values):
    """
    Generador perezoso de todas las combinaciones (alpha, h_dc).
    No construye la lista completa: entrega un par a la vez con yield.
    """
    for alpha in alpha_values:
        for h_dc in hdc_values:
            yield alpha, h_dc


def simulation_jobs(alpha_values, hdc_values, output_dir, skip_existing=True):
    """
    Generador que filtra y produce sólo los trabajos pendientes.
    Si skip_existing=True, omite los archivos ya guardados (permite reanudar).

    Yields: (índice, total, alpha, h_dc, filepath)
    """
    total = len(alpha_values) * len(hdc_values)
    idx = 0
    for alpha, h_dc in parameter_grid(alpha_values, hdc_values):
        idx += 1
        filepath = _build_filepath(output_dir, alpha, h_dc)
        if skip_existing and os.path.exists(filepath):
            print(f"  [skip] ({idx}/{total}) alpha={alpha:.3f}, h_dc={h_dc:.3f}")
            continue
        yield idx, total, alpha, h_dc, filepath


# =============================================================================
# Auxiliares
# =============================================================================

def _build_filepath(output_dir: str, alpha: float, h_dc: float) -> str:
    alpha_str = f"{alpha:.3f}".replace('.', 'p')
    hdc_str   = f"{h_dc:.3f}".replace('.', 'p').replace('-', 'm')
    return os.path.join(output_dir, f"datos_a{alpha_str}_h{hdc_str}.npz")


# =============================================================================
# Función principal del barrido
# =============================================================================

def run_scan(
    chain: HeisenbergChain,
    alpha_values=ALPHA_VALUES,
    hdc_values=HDC_VALUES,
    output_dir: str = OUTPUT_DIR,
    skip_existing: bool = True,
):
    """
    Ejecuta el barrido completo usando simulation_jobs() como generador.

    Para cada (alpha, h_dc):
      1. Construye el campo externo con nucleation_field() → CombinedField
      2. Crea un LLGSimulator con ese alpha
      3. Integra la LLG y guarda la trayectoria
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(alpha_values) * len(hdc_values)

    print("=" * 60)
    print("  BARRIDO COMPLETO DE MOVILIDAD")
    print(f"  {chain}")
    print(f"  Total de simulaciones: {total}")
    print("=" * 60)

    for idx, total_jobs, alpha, h_dc, filepath in simulation_jobs(
        alpha_values, hdc_values, output_dir, skip_existing
    ):
        t_start = time.perf_counter()
        print(f"\n  Sim ({idx}/{total_jobs}): alpha={alpha:.3f}, h_dc/J={h_dc/J:.3f}")

        try:
            sim = LLGSimulator(chain, alpha=alpha, gamma=GAMMA)

            # Construir campo: GaussianPulse + ConstantField (via __add__)
            campo = nucleation_field(N=chain.N, J=chain.J, h_dc=h_dc)
            print(f"  Campo: {campo}")

            sol = sim.run(
                S0=sim.initial_fm_state(),
                t_span=(0.0, T_MAX),
                dt_save=DT_SAVE,
                external_field=campo,
            )

            np.savez_compressed(filepath, S_history=sol.y, time_points=sol.t)
            elapsed = time.perf_counter() - t_start
            print(f"  → Guardado ({elapsed:.1f} s)")

        except MagneticSolitonError as exc:
            print(f"  → [AVISO FÍSICO] {type(exc).__name__}: {exc}")
        except Exception as exc:
            print(f"  → [ERROR] {type(exc).__name__}: {exc}")

    print("\n" + "=" * 60)
    print("  BARRIDO FINALIZADO")
    print("=" * 60)


# =============================================================================
# Punto de entrada
# =============================================================================

if __name__ == "__main__":
    chain = HeisenbergChain(N=N, J=J, D=D, Da=Da)
    run_scan(chain)
