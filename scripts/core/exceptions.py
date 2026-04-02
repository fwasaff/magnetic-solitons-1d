# --- exceptions.py ---
# Jerarquía de excepciones personalizadas para el proyecto de solitones magnéticos.
#
# Concepto de Programación Avanzada: Excepciones personalizadas
# Permiten distinguir errores de DOMINIO FÍSICO de errores genéricos de Python,
# haciendo el código más legible y el manejo de errores más preciso.
#
#   Ejemplo de uso:
#       try:
#           v = tracker.compute_velocity(traj)
#       except SolitonNotFoundError:
#           # manejar ausencia de solitón, no un error genérico
#           v = np.nan


class MagneticSolitonError(Exception):
    """
    Clase base para todos los errores del proyecto.
    Heredar de Exception (no de BaseException) es la práctica estándar.
    """
    pass


# --- Errores de simulación / dinámica ---

class ConvergenceError(MagneticSolitonError):
    """
    El algoritmo de relajación no convergió dentro del número máximo de pasos.
    Se lanza desde find_ground_state() cuando se alcanza max_steps sin
    que el cambio sea menor a la tolerancia.
    """
    def __init__(self, steps: int, final_change: float, tolerance: float):
        self.steps = steps
        self.final_change = final_change
        self.tolerance = tolerance
        super().__init__(
            f"No convergió en {steps} pasos. "
            f"Cambio final: {final_change:.2e} > tolerancia: {tolerance:.2e}. "
            f"Considera aumentar max_steps o dt_relax."
        )


class SolitonNotFoundError(MagneticSolitonError):
    """
    No se detectó ningún solitón en la trayectoria.
    Ocurre cuando Sz nunca cae por debajo del umbral durante la simulación.
    Causas comunes: pulso de nucleación muy débil, amortiguamiento muy alto.
    """
    def __init__(self, alpha: float, threshold: float = 0.0):
        self.alpha = alpha
        self.threshold = threshold
        super().__init__(
            f"No se detectó solitón para alpha={alpha:.3f}. "
            f"Sz nunca cruzó el umbral {threshold}. "
            f"Intenta aumentar la amplitud del pulso de nucleación (h0)."
        )


class SolitonDiedError(MagneticSolitonError):
    """
    El solitón desapareció antes del intervalo de ajuste.
    Diferente a SolitonNotFoundError: el solitón existió pero no sobrevivió
    lo suficiente para medir su velocidad en régimen estacionario.
    """
    def __init__(self, alpha: float, t_death: float, t_fit_start: float):
        self.alpha = alpha
        self.t_death = t_death
        self.t_fit_start = t_fit_start
        super().__init__(
            f"El solitón murió en t={t_death:.1f} para alpha={alpha:.3f}, "
            f"antes del inicio del ajuste (t={t_fit_start:.1f}). "
            f"Considera reducir el campo DC o el amortiguamiento."
        )


# --- Errores de análisis numérico ---

class InsufficientDataError(MagneticSolitonError):
    """
    No hay suficientes puntos de datos para realizar un ajuste estadístico.
    Se lanza cuando se necesitan al menos N puntos y se tienen menos.
    """
    def __init__(self, n_available: int, n_required: int, context: str = ""):
        self.n_available = n_available
        self.n_required = n_required
        super().__init__(
            f"Datos insuficientes para ajuste{': ' + context if context else ''}. "
            f"Se requieren {n_required} puntos, hay {n_available}."
        )


class FitFailedError(MagneticSolitonError):
    """
    El ajuste de curva (curve_fit) no pudo converger.
    Envuelve RuntimeError de scipy con contexto adicional del dominio.
    """
    def __init__(self, fit_type: str, alpha: float, h_dc: float = None):
        self.fit_type = fit_type
        self.alpha = alpha
        self.h_dc = h_dc
        h_info = f", h_dc={h_dc:.3f}" if h_dc is not None else ""
        super().__init__(
            f"Fallo en ajuste '{fit_type}' para alpha={alpha:.3f}{h_info}. "
            f"Verifica que los datos tengan suficiente variación."
        )


# --- Errores de parámetros / configuración ---

class InvalidParameterError(MagneticSolitonError):
    """
    Un parámetro físico tiene un valor fuera del rango válido o esperado.
    Por ejemplo: alpha < 0, N <= 0, J == 0.
    """
    def __init__(self, param_name: str, value, reason: str = ""):
        self.param_name = param_name
        self.value = value
        super().__init__(
            f"Parámetro inválido '{param_name}' = {value}. "
            f"{reason}"
        )


class DataFileNotFoundError(MagneticSolitonError):
    """
    No se encontró un archivo de datos de simulación esperado.
    Más informativo que FileNotFoundError estándar: indica qué simulación falta.
    """
    def __init__(self, filepath: str, alpha: float = None, h_dc: float = None):
        self.filepath = filepath
        self.alpha = alpha
        self.h_dc = h_dc
        params = []
        if alpha is not None:
            params.append(f"alpha={alpha:.3f}")
        if h_dc is not None:
            params.append(f"h_dc={h_dc:.3f}")
        param_str = f" [{', '.join(params)}]" if params else ""
        super().__init__(
            f"Archivo de datos no encontrado{param_str}: {filepath}. "
            f"Ejecuta primero run_mobility_scan.py para generar los datos."
        )
