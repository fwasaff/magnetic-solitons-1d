# --- decorators.py ---
# Decoradores reutilizables para el proyecto de solitones magnéticos.
#
# Concepto de Programación Avanzada: Decoradores
# Un decorador es una función que envuelve a otra función para agregar
# comportamiento ANTES o DESPUÉS de su ejecución, sin modificar su código.
#
# Patrón general:
#   def mi_decorador(func):
#       @functools.wraps(func)  # preserva nombre y docstring del original
#       def wrapper(*args, **kwargs):
#           # ... código antes ...
#           resultado = func(*args, **kwargs)
#           # ... código después ...
#           return resultado
#       return wrapper

import time
import functools
import numpy as np

from .exceptions import InvalidParameterError


# =============================================================================
# @timer — mide y reporta el tiempo de ejecución de cualquier función
# =============================================================================

def timer(func):
    """
    Decorador que mide el tiempo de ejecución de una función e imprime el
    resultado formateado.

    Uso:
        @timer
        def mi_funcion_lenta(...):
            ...

    Salida de ejemplo:
        [find_ground_state] completado en 4.32 s
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        # Formatear según magnitud
        if elapsed < 1.0:
            time_str = f"{elapsed * 1000:.1f} ms"
        elif elapsed < 60.0:
            time_str = f"{elapsed:.2f} s"
        else:
            time_str = f"{elapsed / 60:.1f} min"
        print(f"  [timer] {func.__name__}() → {time_str}")
        return result
    return wrapper


# =============================================================================
# @validate_spins — verifica normalización de espines antes de operar
# =============================================================================

def validate_spins(func):
    """
    Decorador para métodos de clase que reciben un array S de forma (N, 3).
    Verifica que todos los espines estén normalizados (|S_i| ≈ 1) antes de
    ejecutar la función. Lanza ValueError si detecta espines mal normalizados.

    Se asume que S es el PRIMER argumento posicional después de self.

    Uso:
        class HeisenbergChain:
            @validate_spins
            def effective_field(self, S, ...):
                ...
    """
    @functools.wraps(func)
    def wrapper(self, S, *args, **kwargs):
        norms = np.linalg.norm(S, axis=1)
        tol = 1e-4
        if not np.allclose(norms, 1.0, atol=tol):
            bad_idx = np.where(np.abs(norms - 1.0) > tol)[0]
            raise ValueError(
                f"[{func.__name__}] Espines no normalizados en {len(bad_idx)} sitios. "
                f"Rango de normas: [{norms.min():.6f}, {norms.max():.6f}]. "
                f"Primeros sitios con error: {bad_idx[:5]}."
            )
        return func(self, S, *args, **kwargs)
    return wrapper


# =============================================================================
# @log_simulation — registra parámetros de entrada y estado de salida
# =============================================================================

def log_simulation(func):
    """
    Decorador para métodos de simulación. Imprime los parámetros clave
    al inicio y un resumen del resultado al final.
    Útil para `LLGSimulator.run()` durante barridos largos.

    Uso:
        class LLGSimulator:
            @log_simulation
            def run(self, S0, t_span, ...):
                ...
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Intentar extraer parámetros del objeto self si existen
        alpha = getattr(self, 'alpha', '?')
        chain = getattr(self, 'chain', None)
        n = getattr(chain, 'N', '?') if chain else '?'
        print(f"  [sim] Iniciando {func.__name__}() | N={n}, alpha={alpha}")
        try:
            result = func(self, *args, **kwargs)
            status = getattr(result, 'status', 'ok') if result is not None else 'ok'
            print(f"  [sim] {func.__name__}() finalizado → status={status}")
            return result
        except Exception as exc:
            print(f"  [sim] {func.__name__}() falló → {type(exc).__name__}: {exc}")
            raise
    return wrapper


# =============================================================================
# @validate_parameters — valida rangos de parámetros físicos en __init__
# =============================================================================

def validate_parameters(*rules):
    """
    Decorador de fábrica (decorator factory) que valida parámetros de un
    método __init__. Cada regla es una tupla (nombre, condición, mensaje).

    Concepto avanzado: decorador con argumentos → devuelve un decorador.

    Uso:
        @validate_parameters(
            ('N',     lambda v: v > 0,   "debe ser un entero positivo"),
            ('alpha', lambda v: 0 < v < 1, "debe estar en (0, 1)"),
        )
        def __init__(self, N, alpha, ...):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Construir un dict nombre→valor inspeccionando la firma
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())[1:]  # excluir 'self'
            bound = dict(zip(params, args))
            bound.update(kwargs)

            for param_name, condition, reason in rules:
                if param_name in bound:
                    value = bound[param_name]
                    if not condition(value):
                        raise InvalidParameterError(param_name, value, reason)

            return func(self, *args, **kwargs)
        return wrapper
    return decorator
