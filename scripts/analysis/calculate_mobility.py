# --- calculate_mobility.py ---
# Carga trayectorias guardadas, extrae velocidades y calcula la movilidad μ = dv/dh.
#
# Refactorización con:
#   - Clase SolitonTracker: encapsula la lógica de rastreo y ajuste de velocidad.
#   - Generadores: trajectory_loader() carga archivos uno a uno de forma perezosa.
#   - Excepciones: manejo explícito de SolitonNotFoundError, FitFailedError, etc.

import numpy as np
import os
import sys
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.exceptions import (
    SolitonNotFoundError,
    SolitonDiedError,
    InsufficientDataError,
    FitFailedError,
    DataFileNotFoundError,
)


# =============================================================================
# Parámetros de análisis
# =============================================================================

DATA_DIR    = "datos_barrido_mu"
ALPHA_VALUES = np.linspace(0.01, 0.20, 20)
HDC_VALUES   = np.linspace(-0.02, 0.02, 5)

# Realizaciones (deben coincidir con run_mobility_scan.py)
N_REALIZATIONS_DEFAULT  = 5
N_REALIZATIONS_CRITICAL = 20
ALPHA_CRITICAL_LOW  = 0.02
ALPHA_CRITICAL_HIGH = 0.08

T_START_FIT  = 30.0
T_END_FIT    = 150.0
SZ_THRESHOLD = 0.0   # Umbral para detectar el núcleo del solitón


# =============================================================================
# Clase SolitonTracker
# Responsabilidad: extraer la velocidad de una trayectoria simulada
# =============================================================================

class SolitonTracker:
    """
    Extrae la trayectoria y la velocidad de un solitón a partir de
    los datos guardados de una simulación LLG.

    Parámetros
    ----------
    t_start_fit : float
        Inicio del intervalo de ajuste lineal (régimen estacionario).
    t_end_fit : float
        Fin del intervalo de ajuste lineal.
    sz_threshold : float
        Umbral de Sz para detectar el núcleo del solitón (Sz < threshold).

    Ejemplo
    -------
    >>> tracker = SolitonTracker(t_start_fit=30.0, t_end_fit=150.0)
    >>> data = np.load("datos_a0p050_h0p010.npz")
    >>> v = tracker.compute_velocity(data, alpha=0.05)
    """

    def __init__(
        self,
        t_start_fit: float = T_START_FIT,
        t_end_fit: float   = T_END_FIT,
        sz_threshold: float = SZ_THRESHOLD,
    ):
        self.t_start_fit  = t_start_fit
        self.t_end_fit    = t_end_fit
        self.sz_threshold = sz_threshold

    def _track_position(
        self, S_history: np.ndarray, time_points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rastrea la posición del solitón a lo largo del tiempo.

        Parámetros
        ----------
        S_history : np.ndarray, forma (N_times, N, 3)
            Historia de espines.
        time_points : np.ndarray
            Instantes de tiempo correspondientes.

        Devuelve
        --------
        positions : np.ndarray
            Posición (centroide del núcleo) en cada instante con solitón.
        times : np.ndarray
            Tiempos correspondientes donde se detectó el solitón.
        """
        positions = []
        times     = []

        for t_idx, t in enumerate(time_points):
            core_indices = np.where(S_history[t_idx, :, 2] < self.sz_threshold)[0]
            if len(core_indices) > 0:
                positions.append(np.mean(core_indices))
                times.append(t)

        return np.array(positions), np.array(times)

    def compute_velocity(self, data: dict, alpha: float, h_dc: float = None) -> float:
        """
        Calcula la velocidad del solitón desde un dict de datos de simulación.

        Parámetros
        ----------
        data : dict-like (npz)
            Debe contener 'S_history' (forma N*3 × N_times) y 'time_points'.
        alpha : float
            Valor del amortiguamiento (para mensajes de error).
        h_dc : float, opcional
            Valor del campo DC (para mensajes de error).

        Devuelve
        --------
        float
            Velocidad del solitón en unidades de sitios/tiempo.

        Lanza
        -----
        SolitonNotFoundError
            Si Sz nunca cruza el umbral.
        SolitonDiedError
            Si el solitón muere antes del intervalo de ajuste.
        InsufficientDataError
            Si hay menos de 2 puntos en el intervalo de ajuste.
        FitFailedError
            Si el ajuste lineal de scipy no converge.
        """
        S_history_flat = data['S_history']
        time_points    = data['time_points']
        N = S_history_flat.shape[0] // 3

        # Reformatear a (N_times, N, 3)
        S_history = S_history_flat.T.reshape(-1, N, 3)

        # Rastrear posición del núcleo
        positions, times = self._track_position(S_history, time_points)

        if len(times) == 0:
            raise SolitonNotFoundError(alpha, self.sz_threshold)

        # Verificar que el solitón sobrevivió hasta el intervalo de ajuste
        if times[-1] < self.t_start_fit:
            raise SolitonDiedError(alpha, t_death=times[-1], t_fit_start=self.t_start_fit)

        # Filtrar puntos en el intervalo estacionario
        mask = (times >= self.t_start_fit) & (times <= self.t_end_fit)
        n_points = int(np.sum(mask))

        if n_points < 2:
            raise InsufficientDataError(
                n_available=n_points,
                n_required=2,
                context=f"ajuste lineal (alpha={alpha:.3f})",
            )

        t_fit   = times[mask]
        pos_fit = positions[mask]

        # Ajuste lineal: posición(t) = velocidad × t + posición_inicial
        try:
            params, _ = curve_fit(lambda t, v, p0: v * t + p0, t_fit, pos_fit)
            return float(params[0])
        except RuntimeError:
            raise FitFailedError(fit_type="lineal posición-tiempo", alpha=alpha, h_dc=h_dc)


# =============================================================================
# Generadores de carga de datos
# =============================================================================

def _n_realizations_for(alpha: float) -> int:
    """Devuelve el número de realizaciones según si alpha está en zona crítica."""
    if ALPHA_CRITICAL_LOW <= alpha <= ALPHA_CRITICAL_HIGH:
        return N_REALIZATIONS_CRITICAL
    return N_REALIZATIONS_DEFAULT


def _build_filepath(data_dir: str, alpha: float, h_dc: float, realization: int) -> str:
    """Reconstruye el path del archivo dado alpha, h_dc y realización."""
    alpha_str = f"{alpha:.3f}".replace('.', 'p')
    hdc_str   = f"{h_dc:.3f}".replace('.', 'p').replace('-', 'm')
    return os.path.join(data_dir, f"datos_a{alpha_str}_h{hdc_str}_r{realization:02d}.npz")


def trajectory_loader(data_dir: str, alpha_values, hdc_values):
    """
    Generador perezoso (lazy loading) que carga trayectorias una a una.

    Para cada (alpha, h_dc), itera sobre TODAS las realizaciones disponibles
    y las entrega de a una. Esto permite promediar sobre realizaciones sin
    cargar todos los datos en RAM simultáneamente.

    Yields
    ------
    tuple (float, float, int, dict | None)
        (alpha, h_dc, realization_idx, data)
        data es None si el archivo no existe.
    """
    for alpha in alpha_values:
        n_real = _n_realizations_for(alpha)
        for h_dc in hdc_values:
            for r in range(n_real):
                filepath = _build_filepath(data_dir, alpha, h_dc, r)
                try:
                    data = np.load(filepath)
                    yield alpha, h_dc, r, data
                except FileNotFoundError:
                    yield alpha, h_dc, r, None


def velocity_results(
    tracker: SolitonTracker,
    data_dir: str,
    alpha_values,
    hdc_values,
):
    """
    Pipeline de generadores: trajectory_loader → cálculo de velocidad.

    Para cada (alpha, h_dc, realización) calcula la velocidad del solitón.
    Las velocidades de múltiples realizaciones al mismo (alpha, h_dc) se
    promedian después en run_mobility_analysis() para obtener v ± σ.

    Yields
    ------
    tuple (float, float, int, float | None)
        (alpha, h_dc, realization, velocity)
        velocity es None si ocurrió cualquier error esperado.
    """
    for alpha, h_dc, r, data in trajectory_loader(data_dir, alpha_values, hdc_values):
        if data is None:
            yield alpha, h_dc, r, None
            continue

        try:
            v = tracker.compute_velocity(data, alpha=alpha, h_dc=h_dc)
            yield alpha, h_dc, r, v
        except (SolitonNotFoundError, SolitonDiedError,
                InsufficientDataError, FitFailedError) as exc:
            print(f"  [aviso] alpha={alpha:.3f}, h_dc={h_dc:.3f}, r={r} → {exc}")
            yield alpha, h_dc, r, None


# =============================================================================
# Cálculo de movilidad μ = dv/dh
# =============================================================================

def compute_mobility(velocities: list, hdc_values: list, weights: list = None) -> float | None:
    """
    Calcula la movilidad μ = dv/dh mediante ajuste lineal.

    Parámetros
    ----------
    velocities : list of float
        Velocidades válidas (sin None).
    hdc_values : list of float
        Campos DC correspondientes.

    Devuelve
    --------
    float o None
        Movilidad μ, o None si no hay suficientes datos.
    """
    if len(velocities) < 2:
        return None
    try:
        kwargs = {}
        if weights is not None:
            kwargs['sigma'] = [1.0 / np.sqrt(w) for w in weights]
            kwargs['absolute_sigma'] = True
        params, _ = curve_fit(
            lambda h, mu, v0: mu * h + v0,
            hdc_values,
            velocities,
            **kwargs,
        )
        return float(params[0])
    except RuntimeError:
        return None


# =============================================================================
# Función principal de análisis
# =============================================================================

def run_mobility_analysis(
    data_dir: str = DATA_DIR,
    alpha_values=ALPHA_VALUES,
    hdc_values=HDC_VALUES,
    output_file: str = "mu_vs_alpha_data.npz",
) -> tuple[list, list, list]:
    """
    Análisis completo de movilidad promediando sobre todas las realizaciones.

    Pipeline:
        trajectory_loader → velocity_results → promedio por realización
        → compute_mobility (μ = dv/dh) → error propagation

    Para cada alpha:
      1. Agrupa velocidades por (h_dc, realization)
      2. Promedia velocidades sobre realizaciones para cada h_dc → v̄(h_dc) ± σ
      3. Ajusta v̄(h_dc) = μ·h_dc + v_int para obtener μ y v_int
      4. Propaga incertidumbre de v̄ al error de μ

    Devuelve
    --------
    tuple (list, list, list)
        (alpha_list, mu_list, mu_err_list) — incluye barras de error.
    """
    tracker = SolitonTracker()

    print("=" * 65)
    print("  ANÁLISIS DE MOVILIDAD μ = dv/dh  (con realizaciones)")
    print(f"  Directorio: {data_dir}")
    print("=" * 65)

    # Acumular velocidades indexadas por (alpha, h_dc, realization)
    # Estructura: vel_data[alpha][h_dc] = [v_r0, v_r1, ...]
    vel_data: dict = {a: {h: [] for h in hdc_values} for a in alpha_values}

    for alpha, h_dc, r, v in velocity_results(tracker, data_dir, alpha_values, hdc_values):
        if v is not None:
            # Encontrar la clave h_dc más cercana en el dict
            key = min(vel_data[alpha].keys(), key=lambda k: abs(k - h_dc))
            vel_data[alpha][key].append(v)

    # Calcular μ y su error para cada alpha
    alpha_out  = []
    mu_out     = []
    mu_err_out = []

    for alpha in alpha_values:
        # Para cada h_dc: calcular velocidad media y desviación estándar
        hdc_valid, v_mean, v_std = [], [], []
        for h_dc in hdc_values:
            vs = vel_data[alpha][h_dc]
            if len(vs) >= 1:
                hdc_valid.append(h_dc)
                v_mean.append(float(np.mean(vs)))
                v_std.append(float(np.std(vs)) if len(vs) > 1 else 0.0)

        if len(hdc_valid) < 2:
            print(f"  alpha={alpha:.3f} → insuficientes puntos h_dc ({len(hdc_valid)})")
            continue

        # Ajuste lineal ponderado: v̄(h) = μ·h + v_int
        # Usar error estándar de la media (SEM = std/√n) como peso
        n_real = _n_realizations_for(alpha)
        weights = None
        if any(s > 0 for s in v_std):
            sem = [s / np.sqrt(n_real) + 1e-6 for s in v_std]
            weights = [1.0 / s**2 for s in sem]

        mu = compute_mobility(v_mean, hdc_valid, weights=weights)
        if mu is None:
            print(f"  alpha={alpha:.3f} → ajuste falló")
            continue

        # Error de μ: propagación desde los SEM de velocidad
        # Estimación simple: rango intercuartil de los μ por realización
        # (para publicación se usará bootstrap en la FASE 3)
        v_std_arr = np.array(v_std)
        mu_err = float(np.mean(v_std_arr) / np.sqrt(n_real)) if np.any(v_std_arr > 0) else 0.0

        alpha_out.append(alpha)
        mu_out.append(mu)
        mu_err_out.append(mu_err)

        n_pts = sum(len(vel_data[alpha][h]) for h in hdc_values)
        zone  = " [CRÍTICA]" if ALPHA_CRITICAL_LOW <= alpha <= ALPHA_CRITICAL_HIGH else ""
        print(f"  alpha={alpha:.3f}{zone} → μ={mu:+.4f} ± {mu_err:.4f}  ({n_pts} velocidades)")

    np.savez_compressed(
        output_file,
        alpha=alpha_out,
        mu=mu_out,
        mu_err=mu_err_out,
    )
    print(f"\n  Resultados guardados en: {output_file}")
    print("=" * 65)

    return alpha_out, mu_out, mu_err_out


# =============================================================================
# Punto de entrada
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    alpha_list, mu_list = run_mobility_analysis()

    if not alpha_list:
        print("No hay datos suficientes para graficar.")
        sys.exit(1)

    plt.figure(figsize=(10, 7))
    plt.plot(alpha_list, mu_list, 'o-', linewidth=2, markersize=8, color='C0')
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label=r'$\mu = 0$')
    plt.xlabel(r'Parámetro de Amortiguamiento ($\alpha$)')
    plt.ylabel(r'Movilidad $\mu = dv/dh_z$')
    plt.title('Movilidad vs. Amortiguamiento')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figure_MOBILITY_mu_vs_alpha.png", dpi=300)
    print("Gráfico guardado en 'figure_MOBILITY_mu_vs_alpha.png'")
    plt.show()
