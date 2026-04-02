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

T_START_FIT = 30.0
T_END_FIT   = 150.0
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

def _build_filepath(data_dir: str, alpha: float, h_dc: float) -> str:
    """Reconstruye el path del archivo dado alpha y h_dc."""
    alpha_str = f"{alpha:.3f}".replace('.', 'p')
    hdc_str   = f"{h_dc:.3f}".replace('.', 'p').replace('-', 'm')
    return os.path.join(data_dir, f"datos_a{alpha_str}_h{hdc_str}.npz")


def trajectory_loader(data_dir: str, alpha_values, hdc_values):
    """
    Generador que carga archivos de trayectoria de forma perezosa (lazy loading).

    En lugar de cargar todos los archivos en memoria de una vez, este generador
    lee UN archivo a la vez, lo entrega al llamador, y sólo entonces carga el
    siguiente. Esto es crucial para barridos grandes donde los datos no caben
    en RAM.

    Concepto: generador con 'yield' — la ejecución se pausa en cada yield
    y se reanuda cuando el llamador pide el siguiente elemento.

    Yields
    ------
    tuple (float, float, dict | None)
        (alpha, h_dc, data)
        data es None si el archivo no existe (error no fatal).
    """
    for alpha in alpha_values:
        for h_dc in hdc_values:
            filepath = _build_filepath(data_dir, alpha, h_dc)
            try:
                data = np.load(filepath)
                yield alpha, h_dc, data
            except FileNotFoundError:
                # Emitir None permite al llamador decidir qué hacer
                yield alpha, h_dc, None


def velocity_results(
    tracker: SolitonTracker,
    data_dir: str,
    alpha_values,
    hdc_values,
):
    """
    Generador que combina trajectory_loader con SolitonTracker para producir
    velocidades calculadas.

    Encadena dos generadores: trajectory_loader → cálculo de velocidad.
    Este patrón se llama 'generator pipeline' y es muy eficiente en memoria.

    Yields
    ------
    tuple (float, float, float | None)
        (alpha, h_dc, velocity)
        velocity es None si ocurrió cualquier error esperado.
    """
    for alpha, h_dc, data in trajectory_loader(data_dir, alpha_values, hdc_values):
        if data is None:
            print(f"  [missing] alpha={alpha:.3f}, h_dc={h_dc:.3f} — archivo no encontrado")
            yield alpha, h_dc, None
            continue

        try:
            v = tracker.compute_velocity(data, alpha=alpha, h_dc=h_dc)
            yield alpha, h_dc, v
        except (SolitonNotFoundError, SolitonDiedError,
                InsufficientDataError, FitFailedError) as exc:
            print(f"  [aviso] alpha={alpha:.3f}, h_dc={h_dc:.3f} → {exc}")
            yield alpha, h_dc, None


# =============================================================================
# Cálculo de movilidad μ = dv/dh
# =============================================================================

def compute_mobility(velocities: list, hdc_values: list) -> float | None:
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
        params, _ = curve_fit(
            lambda h, mu, v0: mu * h + v0,
            hdc_values,
            velocities,
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
) -> tuple[list, list]:
    """
    Ejecuta el análisis completo de movilidad para todos los valores de alpha.

    Usa el pipeline de generadores:
        trajectory_loader → velocity_results → compute_mobility

    Parámetros
    ----------
    data_dir : str
        Directorio con los archivos .npz de trayectorias.
    alpha_values, hdc_values : array-like
        Valores del barrido.
    output_file : str
        Archivo de salida con los resultados.

    Devuelve
    --------
    tuple (list, list)
        (alpha_list, mu_list) con los puntos válidos.
    """
    tracker = SolitonTracker()

    print("=" * 60)
    print("  ANÁLISIS DE MOVILIDAD μ = dv/dh")
    print(f"  Directorio de datos: {data_dir}")
    print("=" * 60)

    # Agrupar resultados por alpha usando el generador velocity_results
    results_by_alpha: dict[float, list] = {a: [] for a in alpha_values}
    hdc_by_alpha: dict[float, list] = {a: [] for a in alpha_values}

    for alpha, h_dc, v in velocity_results(tracker, data_dir, alpha_values, hdc_values):
        if v is not None:
            results_by_alpha[alpha].append(v)
            hdc_by_alpha[alpha].append(h_dc)

    # Calcular μ para cada alpha
    alpha_out = []
    mu_out    = []

    for alpha in alpha_values:
        vs  = results_by_alpha[alpha]
        hds = hdc_by_alpha[alpha]

        mu = compute_mobility(vs, hds)
        if mu is not None:
            alpha_out.append(alpha)
            mu_out.append(mu)
            print(f"  alpha={alpha:.3f} → μ = {mu:+.4f}  ({len(vs)} puntos)")
        else:
            print(f"  alpha={alpha:.3f} → μ no calculable ({len(vs)} puntos disponibles)")

    # Guardar resultados
    np.savez_compressed(output_file, alpha=alpha_out, mu=mu_out)
    print(f"\n  Resultados guardados en: {output_file}")
    print("=" * 60)

    return alpha_out, mu_out


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
