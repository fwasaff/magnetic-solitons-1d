# --- fields.py ---
# Jerarquía de campos magnéticos externos usando metaprogramación.
#
# Conceptos de Programación Avanzada aplicados:
#
#   1. abc.ABC + @abstractmethod
#      ExternalField define un CONTRATO: cualquier campo debe implementar
#      __call__(t, N). El intérprete lo verifica en tiempo de instanciación.
#
#   2. @dataclass(frozen=True, slots=True)
#      - frozen=True  → instancia inmutable (como una constante física, no debería
#                       cambiar durante la simulación). Habilita hashing.
#      - slots=True   → Python 3.10+. Reemplaza el __dict__ interno por un layout
#                       fijo en memoria. Más rápido en acceso y usa menos RAM.
#                       Importante en barridos con miles de objetos campo.
#
#   3. Sobrecarga de operador __add__
#      Permite componer campos con la sintaxis natural:
#          campo = GaussianPulse(...) + ConstantField(h_dc=0.01)
#      Devuelve un CombinedField automáticamente.
#
#   Ventaja científica: cada tipo de protocolo experimental queda como
#   una clase independiente, fácil de extender sin tocar el simulador.

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


# =============================================================================
# Clase base abstracta — define el contrato de todos los campos externos
# =============================================================================

class ExternalField(ABC):
    """
    Clase base abstracta para campos magnéticos externos.

    Cualquier subclase DEBE implementar __call__(t, N).
    Si no lo hace, Python lanzará TypeError al intentar instanciarla.

    Metaprogramación: abc.ABCMeta (base de ABC) intercepta la creación de la
    clase y registra qué métodos abstractos existen. Al instanciar, verifica
    automáticamente que todos estén implementados.
    """

    @abstractmethod
    def __call__(self, t: float, N: int) -> np.ndarray:
        """
        Evalúa el campo en el tiempo t para una cadena de N sitios.

        Parámetros
        ----------
        t : float
            Tiempo actual de la simulación.
        N : int
            Número de sitios en la cadena.

        Devuelve
        --------
        np.ndarray, forma (N, 3)
            Campo externo en cada sitio [Bx, By, Bz].
        """
        ...

    def __add__(self, other: 'ExternalField') -> 'CombinedField':
        """
        Sobrecarga del operador + para componer campos.

        Permite escribir:
            campo_total = GaussianPulse(...) + ConstantField(h_dc=0.01)

        En vez de:
            campo_total = CombinedField(GaussianPulse(...), ConstantField(0.01))
        """
        if not isinstance(other, ExternalField):
            return NotImplemented
        return CombinedField(self, other)

    def __mul__(self, factor: float) -> 'ScaledField':
        """Escala la amplitud del campo por un factor: campo_mitad = campo * 0.5"""
        return ScaledField(self, factor)

    def __rmul__(self, factor: float) -> 'ScaledField':
        """Permite escribir: campo_doble = 2.0 * campo"""
        return ScaledField(self, factor)


# =============================================================================
# Campos concretos — implementan el contrato de ExternalField
# =============================================================================

@dataclass(frozen=True, slots=True)
class GaussianPulse(ExternalField):
    """
    Pulso gaussiano en la dirección X para nucleación de solitones.

    Perfil: B_x(i, t) = h0 · exp(-(t-t0)²/2τ²) · exp(-(i-i0)²/2σ²)

    Parámetros
    ----------
    h0 : float    Amplitud del pulso (negativa para orientar el solitón).
    t0 : float    Centro temporal del pulso (en unidades de J⁻¹ℏ).
    tau : float   Ancho temporal (dispersión gaussiana temporal).
    i0 : int      Sitio central del pulso espacial.
    sigma : float Ancho espacial del pulso (sitios).

    Metaprogramación: @dataclass(frozen=True, slots=True)
    - frozen: los parámetros del pulso no cambian durante la simulación.
    - slots: layout fijo en memoria, acceso O(1) a atributos.
    - __init__, __repr__, __eq__, __hash__ generados automáticamente.
    """
    h0:    float
    t0:    float
    tau:   float
    i0:    int
    sigma: float

    def __call__(self, t: float, N: int) -> np.ndarray:
        time_profile  = self.h0 * np.exp(-((t - self.t0) ** 2) / (2 * self.tau ** 2))
        space_profile = np.exp(-((np.arange(N) - self.i0) ** 2) / (2 * self.sigma ** 2))
        field = np.zeros((N, 3))
        field[:, 0] = time_profile * space_profile  # Pulso en eje X
        return field


@dataclass(frozen=True, slots=True)
class ConstantField(ExternalField):
    """
    Campo DC constante en la dirección Z: B_z(i, t) = h_dc.

    Usado para medir la movilidad del solitón: μ = dv/dh_dc.

    Parámetros
    ----------
    h_dc : float   Intensidad del campo constante en z (en unidades de J).
    """
    h_dc: float

    def __call__(self, t: float, N: int) -> np.ndarray:
        field = np.zeros((N, 3))
        field[:, 2] = self.h_dc
        return field


@dataclass(frozen=True, slots=True)
class ScaledField(ExternalField):
    """
    Campo escalado: B_scaled = factor × B_original.
    Creado automáticamente por los operadores * y *=.
    """
    base_field: ExternalField
    factor: float

    def __call__(self, t: float, N: int) -> np.ndarray:
        return self.factor * self.base_field(t, N)


class CombinedField(ExternalField):
    """
    Superposición lineal de múltiples campos externos.
    Creado automáticamente por el operador + entre ExternalField.

    Ejemplo
    -------
    >>> pulse = GaussianPulse(h0=-10.0, t0=2.0, tau=0.5, i0=100, sigma=3.0)
    >>> dc    = ConstantField(h_dc=0.01)
    >>> total = pulse + dc          # usa ExternalField.__add__
    >>> B     = total(t=5.0, N=200) # suma los dos campos
    """

    def __init__(self, *fields: ExternalField):
        # Validar que todos los campos sean ExternalField
        for i, f in enumerate(fields):
            if not isinstance(f, ExternalField):
                raise TypeError(
                    f"El campo #{i} no es un ExternalField: {type(f).__name__}"
                )
        self._fields = tuple(fields)

    def __call__(self, t: float, N: int) -> np.ndarray:
        result = np.zeros((N, 3))
        for field in self._fields:
            result += field(t, N)
        return result

    def __repr__(self) -> str:
        components = " + ".join(repr(f) for f in self._fields)
        return f"CombinedField({components})"

    # CombinedField no puede ser frozen/dataclass porque acepta número variable
    # de campos. Implementamos __eq__ y __hash__ manualmente para consistencia.
    def __eq__(self, other) -> bool:
        return isinstance(other, CombinedField) and self._fields == other._fields

    def __hash__(self) -> int:
        return hash(self._fields)


# =============================================================================
# Función de fábrica — crea el campo estándar del experimento
# =============================================================================

def nucleation_field(
    N: int,
    J: float = 1.0,
    h_dc: float = 0.0,
    h0: float = -10.0,
    t0: float = 2.0,
    tau: float = 0.5,
    sigma: float = 3.0,
) -> CombinedField:
    """
    Crea el campo externo estándar del experimento: pulso de nucleación + DC.

    Encapsula el conocimiento del protocolo experimental en un solo lugar,
    en lugar de repetir los parámetros en cada script.

    Parámetros
    ----------
    N : int        Número de sitios (para calcular i0 = N//2).
    J : float      Constante de intercambio (para escalar h0).
    h_dc : float   Campo DC en z para medir movilidad.
    h0, t0, tau, sigma : parámetros del pulso gaussiano.

    Devuelve
    --------
    CombinedField
        pulse + dc_field
    """
    pulse = GaussianPulse(
        h0=h0 * J,
        t0=t0,
        tau=tau,
        i0=N // 2,
        sigma=sigma,
    )
    dc = ConstantField(h_dc=h_dc)
    return pulse + dc  # usa el operador __add__ sobrecargado
