# --- llg_engine.py ---
# Motor físico del modelo de Heisenberg 1D con DMI y anisotropía.
#
# Conceptos de Programación Avanzada aplicados:
#
#   1. @dataclass — metaprogramación que genera __init__, __repr__, __eq__
#      automáticamente a partir de las anotaciones de tipo de la clase.
#      Se usa __post_init__ para validación de parámetros físicos.
#
#   2. OOP con responsabilidades separadas:
#      - HeisenbergChain : parámetros físicos + física estática (campo efectivo,
#                          estado fundamental, clasificación de fase)
#      - LLGSimulator    : física dinámica (integración LLG, condición inicial)
#
#   3. Decoradores de decorators.py:
#      - @timer         : mide tiempo de ejecución
#      - @validate_spins: verifica normalización antes de operar
#      - @log_simulation: loguea inicio/fin de simulaciones
#
#   4. ExternalField (ABC) de fields.py:
#      LLGSimulator ahora recibe objetos ExternalField en vez de funciones
#      sueltas con dicts de argumentos — más seguro y autodocumentado.

import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
from typing import Optional

from .exceptions import ConvergenceError, InvalidParameterError
from .decorators import timer, validate_spins, log_simulation
from .fields import ExternalField


# =============================================================================
# HeisenbergChain — física estática
# =============================================================================

@dataclass
class HeisenbergChain:
    """
    Cadena de Heisenberg 1D con interacción DMI y anisotropía de eje fácil.

    Hamiltoniano:
        H = -J Σ Sᵢ·Sᵢ₊₁  +  D Σ (Sᵢ×Sᵢ₊₁)·ẑ  +  Dₐ Σ (Sᵢᶻ)²

    Metaprogramación: @dataclass genera automáticamente:
        __init__(self, N, J=1.0, D=0.25, Da=-0.10)
        __repr__  → "HeisenbergChain(N=200, J=1.0, D=0.25, Da=-0.1)"
        __eq__    → comparación por valor de atributos

    La validación se hace en __post_init__, que @dataclass llama al final
    de su __init__ generado.

    Parámetros
    ----------
    N : int    Número de sitios.
    J : float  Intercambio de Heisenberg (J > 0 → ferromagnético).
    D : float  Intensidad DMI (en unidades de J).
    Da : float Anisotropía (Da < 0 → eje fácil en z).

    Ejemplo
    -------
    >>> chain = HeisenbergChain(N=200, J=1.0, D=0.25, Da=-0.10)
    >>> S0 = chain.find_ground_state()
    >>> chain.classify_phase(S0)
    'SL'
    """
    N:  int
    J:  float = 1.0
    D:  float = 0.25
    Da: float = -0.10

    def __post_init__(self):
        """
        Validación de parámetros físicos.

        __post_init__ es el mecanismo de @dataclass para ejecutar código
        personalizado DESPUÉS de que su __init__ generado asigne los atributos.
        Equivale a lo que haría un __init__ manual después de self.X = X.
        """
        if not isinstance(self.N, int) or self.N <= 0:
            raise InvalidParameterError(
                'N', self.N, "debe ser un entero positivo (número de sitios)"
            )
        if self.J == 0:
            raise InvalidParameterError(
                'J', self.J, "no puede ser cero (energía de intercambio)"
            )
        if self.Da > 0:
            raise InvalidParameterError(
                'Da', self.Da, "debe ser ≤ 0 para anisotropía de eje fácil"
            )

    # ------------------------------------------------------------------
    # Propiedades derivadas — calculadas a partir de los atributos base
    # ------------------------------------------------------------------

    @property
    def d_ratio(self) -> float:
        """D/J — relación adimensional de la DMI."""
        return self.D / self.J

    @property
    def da_ratio(self) -> float:
        """Da/J — relación adimensional de la anisotropía."""
        return self.Da / self.J

    # ------------------------------------------------------------------
    # Campo efectivo
    # ------------------------------------------------------------------

    @validate_spins
    def effective_field(
        self,
        S: np.ndarray,
        h_external: "np.ndarray | float" = 0.0,
    ) -> np.ndarray:
        """
        Calcula el campo efectivo B_eff = -∂H/∂S en cada sitio.

        Parámetros
        ----------
        S : np.ndarray (N, 3)
            Configuración de espines (normalizados, validados por @validate_spins).
        h_external : float o np.ndarray (N, 3)
            Campo externo aplicado.

        Devuelve
        --------
        np.ndarray (N, 3)
        """
        s_next = np.roll(S, -1, axis=0)  # vecino derecho (CBC periódicas)
        s_prev = np.roll(S,  1, axis=0)  # vecino izquierdo

        # Heisenberg: B = J(S_{i-1} + S_{i+1})
        b_heisenberg = self.J * (s_prev + s_next)

        # DMI: B_x = -D·ΔS_y,  B_y = +D·ΔS_x  (D orientado en ẑ)
        diff = s_next - s_prev
        b_dmi = np.zeros_like(S)
        b_dmi[:, 0] = -self.D * diff[:, 1]
        b_dmi[:, 1] =  self.D * diff[:, 0]

        # Anisotropía de eje fácil: B_z = -2 Da S_z
        b_aniso = np.zeros_like(S)
        b_aniso[:, 2] = -2.0 * self.Da * S[:, 2]

        return b_heisenberg + b_dmi + b_aniso + h_external

    # ------------------------------------------------------------------
    # Estado fundamental por relajación
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
        Encuentra el estado fundamental por descenso de gradiente en la esfera.

        Parámetros
        ----------
        max_steps : int      Máximo de iteraciones.
        tolerance : float    Criterio de parada (cambio máximo en S).
        dt_relax : float     Paso de la relajación.
        raise_on_no_convergence : bool
            Si True, lanza ConvergenceError al agotar los pasos.

        Devuelve
        --------
        np.ndarray (N, 3) — estado fundamental (o el mejor encontrado).
        """
        S = np.random.rand(self.N, 3) - 0.5
        S /= np.linalg.norm(S, axis=1, keepdims=True)

        final_change = np.inf
        for step in range(max_steps):
            B_eff = self.effective_field(S, h_external=0)
            S_old = S.copy()
            S = S + dt_relax * B_eff
            S /= np.linalg.norm(S, axis=1, keepdims=True)

            if step % 500 == 0:
                final_change = np.max(np.abs(S - S_old))
                if final_change < tolerance:
                    return S

        if raise_on_no_convergence:
            raise ConvergenceError(max_steps, final_change, tolerance)
        return S

    # ------------------------------------------------------------------
    # Clasificación de fase
    # ------------------------------------------------------------------

    def compute_structure_factor(self, S: np.ndarray) -> tuple:
        """
        Calcula el factor de estructura de la componente Sz:
            S(k) = |FFT(Sz)|² / N

        Cada fase tiene una firma característica en S(k):
          FM  : pico dominante en k=0 (orden uniforme)
          H   : pico en k_helix ≈ arctan(D/J)/π (espiral uniforme)
          SL  : pico en k_SL < k_helix (red con periodo mayor)

        Devuelve
        --------
        k_vals : np.ndarray  Vectores de onda en unidades de 2π/a.
        S_k    : np.ndarray  Potencia espectral normalizada.
        k_peak : float       k del pico dominante (excluyendo k=0).
        """
        sz = S[:, 2]
        fft_vals = np.fft.rfft(sz)
        S_k    = np.abs(fft_vals) ** 2 / self.N
        k_vals = np.fft.rfftfreq(self.N) * 2 * np.pi  # unidades: 2π/a

        # Pico dominante excluyendo el modo DC (k=0)
        S_k_ac = S_k[1:]
        k_peak = k_vals[1:][np.argmax(S_k_ac)]

        return k_vals, S_k, k_peak

    @validate_spins
    def classify_phase(self, S: np.ndarray) -> str:
        """
        Clasifica la fase magnética: 'FM', 'H' o 'SL'.

        Criterio combinado: estadísticos de Sz + factor de estructura S(k).

        Lógica:
          1. FM  : ⟨Sz⟩ > 0.95  →  espines bien alineados en z.
          2. H   : S(k=0) domina sobre el espectro AC  →  Sz ≈ 0,
                   sin estructura periódica detectada.
          3. SL  : pico AC significativo en S(k≠0)  →  paredes de dominio.

        El ratio potencia_AC / potencia_DC es el discriminante más robusto
        que los umbrales puros de media/desviación, ya que no depende de
        la convención de signo del estado fundamental.

        Devuelve
        --------
        str : 'FM', 'H' o 'SL'
        """
        sz_mean = np.mean(S[:, 2])
        _, S_k, _ = self.compute_structure_factor(S)

        # Potencia en DC (k=0) vs máximo del espectro AC (k≠0)
        power_dc = S_k[0]
        power_ac_max = S_k[1:].max()
        # Ratio: cuánto peso tiene la estructura periódica respecto al uniforme
        ac_dc_ratio = power_ac_max / (power_dc + 1e-12)

        if sz_mean > 0.95:
            return "FM"
        elif ac_dc_ratio < 0.05:
            # El espectro AC es < 5% del DC → sin estructura periódica → Helicoidal
            return "H"
        else:
            return "SL"

    def phase_diagnostics(self, S: np.ndarray) -> dict:
        """
        Devuelve un diccionario con todos los indicadores usados para
        clasificar la fase. Útil para validar la clasificación y para
        las figuras de diagnóstico del árbitro.

        Devuelve
        --------
        dict con: phase, sz_mean, sz_std, k_peak, ac_dc_ratio
        """
        sz = S[:, 2]
        _, S_k, k_peak = self.compute_structure_factor(S)
        power_dc    = S_k[0]
        power_ac    = S_k[1:].max()
        ac_dc_ratio = power_ac / (power_dc + 1e-12)

        phase = self.classify_phase(S)

        # Vector de onda helicoidal teórico: k_H = arctan(D/J)
        k_helix_theory = np.arctan(self.D / self.J) / np.pi * 2 * np.pi

        return {
            'phase':          phase,
            'sz_mean':        float(np.mean(sz)),
            'sz_std':         float(np.std(sz)),
            'k_peak':         float(k_peak),
            'k_helix_theory': float(k_helix_theory),
            'ac_dc_ratio':    float(ac_dc_ratio),
        }


# =============================================================================
# LLGSimulator — física dinámica
# =============================================================================

@dataclass
class LLGSimulator:
    """
    Integrador de la ecuación de Landau-Lifshitz-Gilbert (LLG).

    dSᵢ/dt = -γ Sᵢ × B_eff  −  α·γ Sᵢ × (Sᵢ × B_eff)

    Ahora recibe objetos ExternalField (ABC) en vez de funciones sueltas,
    lo que garantiza la interfaz correcta en tiempo de instanciación.

    Metaprogramación: @dataclass + __post_init__ (igual que HeisenbergChain).
    field() de dataclasses evita el antipatrón de mutable default argument.

    Parámetros
    ----------
    chain : HeisenbergChain   La cadena física a simular.
    alpha : float             Parámetro de amortiguamiento de Gilbert.
    gamma : float             Razón giromagnética (adimensional = 1.0).

    Ejemplo
    -------
    >>> chain = HeisenbergChain(N=200, J=1.0, D=0.25, Da=-0.10)
    >>> sim   = LLGSimulator(chain, alpha=0.05)
    >>> pulse = GaussianPulse(h0=-10.0, t0=2.0, tau=0.5, i0=100, sigma=3.0)
    >>> dc    = ConstantField(h_dc=0.01)
    >>> sol   = sim.run(S0, t_span=(0, 200), external_field=pulse + dc)
    """
    chain: HeisenbergChain
    alpha: float
    gamma: float = 1.0

    def __post_init__(self):
        if not isinstance(self.chain, HeisenbergChain):
            raise TypeError(
                f"'chain' debe ser HeisenbergChain, no {type(self.chain).__name__}"
            )
        if not (0 < self.alpha < 1):
            raise InvalidParameterError(
                'alpha', self.alpha, "debe estar en el intervalo abierto (0, 1)"
            )
        if self.gamma <= 0:
            raise InvalidParameterError(
                'gamma', self.gamma, "debe ser positivo"
            )

    # ------------------------------------------------------------------
    # Lado derecho de la LLG (para solve_ivp)
    # ------------------------------------------------------------------

    def _rhs(
        self,
        t: float,
        S_flat: np.ndarray,
        external_field: Optional[ExternalField],
    ) -> np.ndarray:
        """
        Calcula dS/dt para el integrador de scipy.

        Parámetros
        ----------
        t : float
            Tiempo actual.
        S_flat : np.ndarray (N*3,)
            Estado del sistema aplanado.
        external_field : ExternalField | None
            Campo externo (objeto callable con interfaz garantizada por ABC).

        Devuelve
        --------
        np.ndarray (N*3,)
        """
        N = self.chain.N
        S = S_flat.reshape((N, 3))

        # Re-normalizar para estabilidad numérica
        norms = np.linalg.norm(S, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        S = S / norms

        # Evaluar campo externo
        h_ext = external_field(t, N) if external_field is not None else 0.0

        B_eff = self.chain.effective_field(S, h_ext)

        precession = -self.gamma * np.cross(S, B_eff)
        damping    = -self.alpha * self.gamma * np.cross(S, np.cross(S, B_eff))

        return (precession + damping).flatten()

    # ------------------------------------------------------------------
    # Integración completa
    # ------------------------------------------------------------------

    @log_simulation
    def run(
        self,
        S0: np.ndarray,
        t_span: tuple,
        dt_save: float = 0.5,
        external_field: Optional[ExternalField] = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ):
        """
        Integra la LLG desde S0 en el intervalo t_span.

        Parámetros
        ----------
        S0 : np.ndarray (N, 3)       Condición inicial.
        t_span : tuple (t0, tf)      Intervalo de integración.
        dt_save : float              Paso de muestreo de la solución.
        external_field : ExternalField | None
            Campo externo (objeto de jerarquía ExternalField).
            Admite cualquier subclase: GaussianPulse, ConstantField,
            CombinedField, ScaledField, etc.
        rtol, atol : float           Tolerancias de RK45.

        Devuelve
        --------
        scipy.integrate.OdeResult
            .t  → tiempos guardados
            .y  → estados (N*3 × N_tiempos), reformatear con .T.reshape(-1,N,3)
        """
        t_eval = np.arange(t_span[0], t_span[1], dt_save)

        return solve_ivp(
            fun=self._rhs,
            t_span=t_span,
            y0=S0.flatten(),
            method='RK45',
            t_eval=t_eval,
            args=(external_field,),
            rtol=rtol,
            atol=atol,
        )

    # ------------------------------------------------------------------
    # Estado inicial estándar: ferromagnético metaestable
    # ------------------------------------------------------------------

    def initial_fm_state(self) -> np.ndarray:
        """
        Devuelve el estado ferromagnético metaestable: todos los espines en +z.
        Punto de partida estándar para la nucleación de solitones.
        """
        S0 = np.zeros((self.chain.N, 3))
        S0[:, 2] = 1.0
        return S0

    def initial_fm_state_with_noise(self, noise_amplitude: float = 0.01) -> np.ndarray:
        """
        Estado FM metaestable con pequeñas fluctuaciones térmicas aleatorias.

        Motivación física: un sistema real a temperatura finita no parte de
        un estado perfectamente polarizado. Las fluctuaciones modifican
        ligeramente la trayectoria del solitón, generando realizaciones
        estadísticamente independientes.

        Este es el mecanismo correcto para calcular barras de error en la
        velocidad y la movilidad (respuesta al árbitro C4).

        Parámetros
        ----------
        noise_amplitude : float
            Amplitud de la perturbación aleatoria (en unidades de |S|=1).
            Por defecto 0.01 → ≈ 1% del módulo del espín.

        Devuelve
        --------
        np.ndarray (N, 3), normalizado (|S_i| = 1).
        """
        S0 = np.zeros((self.chain.N, 3))
        S0[:, 2] = 1.0
        S0 += noise_amplitude * (np.random.rand(self.chain.N, 3) - 0.5)
        # Re-normalizar para mantener |S_i| = 1
        S0 /= np.linalg.norm(S0, axis=1, keepdims=True)
        return S0
