# --- run_mobility_scan.py ---
# Ejecuta el barrido completo de movilidad: alpha × h_dc.
#
# Refactorización con:
#   - Generadores: parameter_grid() y simulation_jobs() evitan construir
#     listas enormes en memoria — cada combinación se genera bajo demanda.
#   - Clases: usa HeisenbergChain y LLGSimulator del motor físico.
#   - Excepciones: manejo explícito de errores de simulación.

import numpy as np
import os
import sys
import time

# Ajustar path para importar desde scripts/core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.llg_engine import HeisenbergChain, LLGSimulator
from core.exceptions import MagneticSolitonError


# =============================================================================
# Parámetros del sistema
# =============================================================================

# Física
N    = 200
J    = 1.0
D    = 0.25 * J
Da   = -0.10 * J
GAMMA = 1.0

# Pulso de nucleación del solitón
PULSE_PARAMS = {
    'h0':    -10.0 * J,
    't0':    2.0,
    'tau':   0.5,
    'i0':    N // 2,
    'sigma': 3.0,
}

# Barrido de parámetros
ALPHA_VALUES = np.linspace(0.01, 0.20, 20)   # 20 valores de amortiguamiento
HDC_VALUES   = np.linspace(-0.02, 0.02, 5)   # 5 campos DC para calcular dv/dh

# Simulación
T_MAX   = 200.0
DT_SAVE = 0.5
OUTPUT_DIR = "datos_barrido_mu"


# =============================================================================
# Generadores
# =============================================================================

def parameter_grid(alpha_values, hdc_values):
    """
    Generador que produce todas las combinaciones (alpha, h_dc) de forma perezosa.

    Concepto: generador con 'yield' — no construye la lista completa en memoria,
    sino que entrega un par a la vez. Útil cuando el número de combinaciones
    es grande (aquí: 20 × 5 = 100 pares).

    Yields
    ------
    tuple (float, float)
        Par (alpha, h_dc).
    """
    for alpha in alpha_values:
        for h_dc in hdc_values:
            yield alpha, h_dc


def simulation_jobs(alpha_values, hdc_values, output_dir, skip_existing=True):
    """
    Generador que filtra y produce sólo los trabajos que aún no tienen
    resultados guardados en disco.

    Concepto: generador con lógica de filtrado — combina iteración perezosa
    con consulta al sistema de archivos, sin cargar nada en memoria.

    Parámetros
    ----------
    alpha_values, hdc_values : array-like
        Valores del barrido.
    output_dir : str
        Directorio donde se guardan los resultados.
    skip_existing : bool
        Si True, omite simulaciones ya completadas (permite reanudar).

    Yields
    ------
    tuple (int, int, float, float, str)
        (índice_actual, total, alpha, h_dc, filepath)
    """
    total = len(alpha_values) * len(hdc_values)
    idx = 0
    for alpha, h_dc in parameter_grid(alpha_values, hdc_values):
        idx += 1
        filepath = _build_filepath(output_dir, alpha, h_dc)
        if skip_existing and os.path.exists(filepath):
            print(f"  [skip] ({idx}/{total}) alpha={alpha:.3f}, h_dc={h_dc:.3f} — ya existe.")
            continue
        yield idx, total, alpha, h_dc, filepath


# =============================================================================
# Funciones auxiliares
# =============================================================================

def _build_filepath(output_dir: str, alpha: float, h_dc: float) -> str:
    """Construye el path del archivo de salida de forma reproducible."""
    alpha_str = f"{alpha:.3f}".replace('.', 'p')
    hdc_str   = f"{h_dc:.3f}".replace('.', 'p').replace('-', 'm')
    return os.path.join(output_dir, f"datos_a{alpha_str}_h{hdc_str}.npz")


def _build_h_args(h_dc: float) -> dict:
    """Empaqueta los argumentos del campo externo para LLGSimulator.gaussian_pulse."""
    return {**PULSE_PARAMS, 'h_dc': h_dc}


def _initial_state(N: int) -> np.ndarray:
    """Estado inicial: ferromagnético metaestable (todos los espines en +z)."""
    S0 = np.zeros((N, 3))
    S0[:, 2] = 1.0
    return S0


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
    Ejecuta el barrido completo de movilidad usando el generador simulation_jobs().

    Parámetros
    ----------
    chain : HeisenbergChain
        La cadena física a simular.
    alpha_values : array-like
        Valores de amortiguamiento a barrer.
    hdc_values : array-like
        Valores de campo DC a barrer.
    output_dir : str
        Directorio de salida para los archivos .npz.
    skip_existing : bool
        Si True, omite simulaciones ya guardadas (permite reanudar el barrido).
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(alpha_values) * len(hdc_values)

    print("=" * 60)
    print("  BARRIDO COMPLETO DE MOVILIDAD")
    print(f"  {chain}")
    print(f"  Total de simulaciones: {total}")
    print(f"  Directorio de salida:  {output_dir}")
    print("=" * 60)

    # El generador simulation_jobs() produce sólo los trabajos pendientes
    for idx, total_jobs, alpha, h_dc, filepath in simulation_jobs(
        alpha_values, hdc_values, output_dir, skip_existing
    ):
        t_start = time.perf_counter()
        print(f"\n  Sim ({idx}/{total_jobs}): alpha={alpha:.3f}, h_dc/J={h_dc/J:.3f}")

        try:
            # Crear simulador con este valor de alpha
            sim = LLGSimulator(chain, alpha=alpha, gamma=GAMMA)

            # Ejecutar dinámica con campo externo combinado (pulso + DC)
            sol = sim.run(
                S0=_initial_state(chain.N),
                t_span=(0.0, T_MAX),
                dt_save=DT_SAVE,
                h_func=LLGSimulator.gaussian_pulse,
                h_args=_build_h_args(h_dc),
            )

            # Guardar trayectoria
            np.savez_compressed(filepath, S_history=sol.y, time_points=sol.t)

            elapsed = time.perf_counter() - t_start
            print(f"  → Guardado en {filepath} ({elapsed:.1f} s)")

        except MagneticSolitonError as exc:
            # Error de dominio físico esperado: registrar y continuar
            print(f"  → [AVISO FÍSICO] {type(exc).__name__}: {exc}")

        except Exception as exc:
            # Error inesperado: registrar sin abortar el barrido completo
            print(f"  → [ERROR] {type(exc).__name__}: {exc}")
            print(f"     Continuando con el siguiente trabajo...")

    print("\n" + "=" * 60)
    print("  BARRIDO FINALIZADO")
    print("=" * 60)


# =============================================================================
# Punto de entrada
# =============================================================================

if __name__ == "__main__":
    chain = HeisenbergChain(N=N, J=J, D=D, Da=Da)
    run_scan(chain)
