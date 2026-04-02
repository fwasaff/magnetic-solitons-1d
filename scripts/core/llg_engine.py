# --- llg_engine.py ---
# Motor físico del modelo de Heisenberg 1D con DMI y anisotropía.
#
# Refactorización con Programación Orientada a Objetos:
#
#   HeisenbergChain — encapsula los parámetros físicos y la física estática
#   LLGSimulator    — encapsula la dinámica LLG para una cadena dada
#
# Cada clase tiene responsabilidades claras y únicas (principio SRP).
# Los decoradores de decorators.py se usan para validación y logging.

import numpy as np
from scipy.integrate import solve_ivp

from .exceptions import ConvergenceError, InvalidParameterError
from .decorators import timer, validate_spins, validate_parameters, log_simulation


# =============================================================================
# Clase HeisenbergChain
# Responsabilidad: física ESTÁTICA — parámetros, campo efectivo, estado fundamental
# =============================================================================

class HeisenbergChain:
    """
    Cadena de Heisenberg 1D con interacción DMI y anisotropía de eje fácil.

    Hamiltoniano:
        H = -J Σ Sᵢ·Sᵢ₊₁  +  D Σ (Sᵢ×Sᵢ₊₁)·ẑ  +  Dₐ Σ (Sᵢᶻ)²

    Parámetros
    ----------
    N : int
        Número de sitios en la cadena.
    J : float
        Constante de intercambio de Heisenberg (J > 0 → ferromagnético).
    D : float
        Intensidad de la interacción Dzyaloshinskii-Moriya (en unidades de J).
    Da : float
        Constante de anisotropía (Da < 0 → eje fácil en z).

    Ejemplo
    -------
    >>> chain = HeisenbergChain(N=200, J=1.0, D=0.25, Da=-0.10)
    >>> S0 = chain.find_ground_state()
    >>> phase = chain.classify_phase(S0)
    """

    @validate_parameters(
        ('N',  lambda v: isinstance(v, int) and v > 0, "debe ser un entero positivo"),
        ('J',  lambda v: v != 0,                       "no puede ser cero"),
        ('Da', lambda v: v <= 0,                       "debe ser ≤ 0 para eje fácil"),
    )
    def __init__(self, N: int, J: float = 1.0, D: float = 0.25, Da: float = -0.10):
        self.N = N
        self.J = J
        self.D = D
        self.Da = Da

    def __repr__(self):
        return (
            f"HeisenbergChain(N={self.N}, J={self.J}, "
            f"D/J={self.D/self.J:.3f}, Da/J={self.Da/self.J:.3f})"
        )

    # ------------------------------------------------------------------
    # Física del campo efectivo
    # ------------------------------------------------------------------

    @validate_spins
    def effective_field(self, S: np.ndarray, h_external=0.0) -> np.ndarray:
        """
        Calcula el campo efectivo B_eff en cada sitio de la cadena.

        Parámetros
        ----------
        S : np.ndarray, forma (N, 3)
            Configuración de espines (deben estar normalizados).
        h_external : float o np.ndarray, forma (N, 3)
            Campo externo aplicado.

        Devuelve
        --------
        np.ndarray, forma (N, 3)
            Campo efectivo total en cada sitio.
        """
        # Condiciones de contorno periódicas
        s_next = np.roll(S, -1, axis=0)
        s_prev = np.roll(S,  1, axis=0)

        # 1. Término de Heisenberg
        b_heisenberg = self.J * (s_prev + s_next)

        # 2. Término DMI: D = D·ẑ  →  B_DMI = D (ẑ × ΔS)
        diff = s_next - s_prev
        b_dmi = np.zeros_like(S)
        b_dmi[:, 0] = -self.D * diff[:, 1]
        b_dmi[:, 1] =  self.D * diff[:, 0]

        # 3. Término de anisotropía de eje fácil
        b_anisotropy = np.zeros_like(S)
        b_anisotropy[:, 2] = -2.0 * self.Da * S[:, 2]

        return b_heisenberg + b_dmi + b_anisotropy + h_external

    # ------------------------------------------------------------------
    # Estado fundamental
    # ------------------------------------------------------------------

    @timer
    def find_ground_state(
        self,
        max_steps: int = 20000,
        tolerance: float = 1e-8,
        dt_relax: float = 0.05,
        raise_on_no_convergence: bool = False,
    ) -> np.ndarray:
        """
        Encuentra el estado fundamental por relajación (descenso de gradiente
        en la esfera unitaria).

        Parámetros
        ----------
        max_steps : int
            Número máximo de pasos de relajación.
        tolerance : float
            Criterio de convergencia (cambio máximo en espines).
        dt_relax : float
            Paso de tiempo de la relajación.
        raise_on_no_convergence : bool
            Si True, lanza ConvergenceError al agotar los pasos.
            Si False (por defecto), devuelve el mejor estado encontrado.

        Devuelve
        --------
        np.ndarray, forma (N, 3)
            Configuración de espines del estado fundamental (o el mejor encontrado).

        Lanza
        -----
        ConvergenceError
            Si raise_on_no_convergence=True y no se alcanzó la tolerancia.
        """
        # Estado inicial aleatorio en la esfera unitaria
        S = np.random.rand(self.N, 3) - 0.5
        S /= np.linalg.norm(S, axis=1, keepdims=True)

        final_change = np.inf

        for step in range(max_steps):
            B_eff = self.effective_field(S, h_external=0)
            S_old = S.copy()

            # Paso de descenso de gradiente en la esfera
            S = S + dt_relax * B_eff
            S /= np.linalg.norm(S, axis=1, keepdims=True)

            # Verificar convergencia cada 500 pasos
            if step % 500 == 0:
                final_change = np.max(np.abs(S - S_old))
                if final_change < tolerance:
                    return S  # Convergió exitosamente

        # Se agotaron los pasos
        if raise_on_no_convergence:
            raise ConvergenceError(
                steps=max_steps,
                final_change=final_change,
                tolerance=tolerance,
            )
        return S  # Devuelve el mejor estado disponible

    # ------------------------------------------------------------------
    # Clasificación de fase
    # ------------------------------------------------------------------

    @validate_spins
    def classify_phase(self, S: np.ndarray) -> str:
        """
        Clasifica la fase magnética de una configuración de espines.

        Criterios:
          - FM  (Ferromagnético): ⟨Sz⟩ > 0.95  →  espines bien alineados en z
          - H   (Helicoidal):     std(Sz) < 0.45 →  oscilaciones suaves en z
          - SL  (Soliton Lattice): resto         →  paredes de dominio quirales

        Parámetros
        ----------
        S : np.ndarray, forma (N, 3)
            Configuración de espines.

        Devuelve
        --------
        str
            Una de: 'FM', 'H', 'SL'
        """
        sz_mean = np.mean(S[:, 2])
        sz_std  = np.std(S[:, 2])

        if sz_mean > 0.95:
            return "FM"
        elif sz_std < 0.45:
            return "H"
        else:
            return "SL"


# =============================================================================
# Clase LLGSimulator
# Responsabilidad: física DINÁMICA — integración de la ecuación LLG
# =============================================================================

class LLGSimulator:
    """
    Integrador de la ecuación de Landau-Lifshitz-Gilbert (LLG) para
    una HeisenbergChain dada.

    Ecuación LLG:
        dSᵢ/dt = -γ Sᵢ × B_eff  -  α·γ Sᵢ × (Sᵢ × B_eff)

    Parámetros
    ----------
    chain : HeisenbergChain
        La cadena física a simular.
    alpha : float
        Parámetro de amortiguamiento de Gilbert (0 < alpha < 1).
    gamma : float
        Razón giromagnética (por defecto 1.0 en unidades adimensionales).

    Ejemplo
    -------
    >>> chain = HeisenbergChain(N=200, J=1.0, D=0.25, Da=-0.10)
    >>> sim = LLGSimulator(chain, alpha=0.05)
    >>> S0 = np.zeros((200, 3)); S0[:, 2] = 1.0  # Estado FM inicial
    >>> sol = sim.run(S0, t_span=(0, 200))
    """

    @validate_parameters(
        ('alpha', lambda v: 0 < v < 1,  "debe estar en el intervalo (0, 1)"),
        ('gamma', lambda v: v > 0,      "debe ser positivo"),
    )
    def __init__(self, chain: HeisenbergChain, alpha: float, gamma: float = 1.0):
        self.chain = chain
        self.alpha = alpha
        self.gamma = gamma

    def __repr__(self):
        return (
            f"LLGSimulator(chain={self.chain}, "
            f"alpha={self.alpha}, gamma={self.gamma})"
        )

    # ------------------------------------------------------------------
    # Lado derecho de la ecuación LLG (para solve_ivp)
    # ------------------------------------------------------------------

    def _rhs(self, t: float, S_flat: np.ndarray,
             h_func=None, h_args: dict = None) -> np.ndarray:
        """
        Calcula dS/dt para el integrador de scipy.

        Parámetros
        ----------
        t : float
            Tiempo actual.
        S_flat : np.ndarray, forma (N*3,)
            Estado actual del sistema como array plano.
        h_func : callable, opcional
            Función h_func(t, N, **h_args) → np.ndarray (N, 3).
        h_args : dict, opcional
            Argumentos adicionales para h_func.

        Devuelve
        --------
        np.ndarray, forma (N*3,)
            Derivada temporal aplanada.
        """
        N = self.chain.N
        S = S_flat.reshape((N, 3))

        # Re-normalizar en cada paso para estabilidad numérica
        norms = np.linalg.norm(S, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        S = S / norms

        # Campo externo en este instante
        if h_func is not None:
            h_ext = h_func(t, N, **(h_args or {}))
        else:
            h_ext = 0.0

        B_eff = self.chain.effective_field(S, h_ext)

        # Términos de la LLG
        precession = -self.gamma * np.cross(S, B_eff)
        damping    = -self.alpha * self.gamma * np.cross(S, np.cross(S, B_eff))

        return (precession + damping).flatten()

    # ------------------------------------------------------------------
    # Ejecutar simulación completa
    # ------------------------------------------------------------------

    @log_simulation
    def run(
        self,
        S0: np.ndarray,
        t_span: tuple,
        dt_save: float = 0.5,
        h_func=None,
        h_args: dict = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ):
        """
        Integra la ecuación LLG desde S0 en el intervalo t_span.

        Parámetros
        ----------
        S0 : np.ndarray, forma (N, 3)
            Condición inicial de los espines.
        t_span : tuple (t0, tf)
            Intervalo de integración.
        dt_save : float
            Intervalo de tiempo entre puntos guardados.
        h_func : callable, opcional
            Campo externo dependiente del tiempo.
        h_args : dict, opcional
            Argumentos para h_func.
        rtol, atol : float
            Tolerancias del integrador RK45.

        Devuelve
        --------
        scipy.integrate.OdeResult
            Objeto con atributos .t (tiempos) y .y (estados).
            Acceder a la solución: sol.y.T.reshape(-1, N, 3)
        """
        t0, tf = t_span
        t_eval = np.arange(t0, tf, dt_save)

        return solve_ivp(
            fun=self._rhs,
            t_span=t_span,
            y0=S0.flatten(),
            method='RK45',
            t_eval=t_eval,
            args=(h_func, h_args),
            rtol=rtol,
            atol=atol,
        )

    # ------------------------------------------------------------------
    # Pulso de nucleación gaussiano
    # ------------------------------------------------------------------

    @staticmethod
    def gaussian_pulse(
        t: float,
        N: int,
        h0: float,
        t0: float,
        tau: float,
        i0: int,
        sigma: float,
        h_dc: float = 0.0,
    ) -> np.ndarray:
        """
        Campo externo combinado: pulso Gaussiano en X (nucleación) + campo DC en Z.

        Parámetros
        ----------
        t : float      Tiempo actual.
        N : int        Número de sitios.
        h0 : float     Amplitud del pulso.
        t0 : float     Centro temporal del pulso.
        tau : float    Ancho temporal del pulso.
        i0 : int       Centro espacial del pulso (sitio).
        sigma : float  Ancho espacial del pulso (sitios).
        h_dc : float   Campo DC constante en la dirección z.

        Devuelve
        --------
        np.ndarray, forma (N, 3)
        """
        time_profile  = h0 * np.exp(-((t - t0) ** 2) / (2 * tau ** 2))
        space_profile = np.exp(-((np.arange(N) - i0) ** 2) / (2 * sigma ** 2))

        field = np.zeros((N, 3))
        field[:, 0] = time_profile * space_profile  # Pulso en X
        field[:, 2] = h_dc                          # Campo DC en Z
        return field
