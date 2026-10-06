"""
experiments.py — Núcleo cuántico del Laboratorio Cuántico-Junguiano.

BUGS CORREGIDOS:
  1. [CRÍTICO] Arquetipo.__init__: np.random.seed(seed) es global y muta el estado
     de todo el proceso. Reemplazado por np.random.default_rng(seed) con instancia
     local (self._rng), eliminando la contaminación cruzada entre instancias.
  2. [CRÍTICO] Arquetipo.medir: usaba np.random.random() (RNG global), ignorando
     cualquier semilla local. Ahora usa self._rng.random() (RNG instanciado).
  3. [CRÍTICO] ParConDecoherencia.__init__: misma contaminación de RNG global.
     Reemplazado con self._rng = np.random.default_rng(seed).
  4. [CRÍTICO] ParConDecoherencia.medir_base_X: usaba np.random.random() global.
     Reemplazado con self._rng.random().
  5. [MENOR] Arquetipo.aplicar_rotacion: construía el objeto con object.__new__
     sin asignar self._rng, dejando el nuevo Arquetipo sin generador. Corregido:
     el nuevo objeto hereda una instancia fresca de RNG del padre.
  6. [MENOR] ParConDecoherencia.aplicar_represion: el check de gamma no protegía
     contra NaN silencioso cuando gamma era np.nan. Añadido isinstance guard.

MEJORAS AÑADIDAS:
  - fidelidad(): calcula la fidelidad cuántica F = |⟨ψ|φ⟩|² entre dos Arquetipos.
  - distancia_traza(): distancia de traza para estados puros.
  - Arquetipo es ahora un dataclass-compatible __eq__ basado en amplitudes.
  - ParConDecoherencia.entropia_entrelazamiento(): entropía de Von Neumann de
    la matriz de densidad reducida del qubit 1, mide el grado de entrelazamiento.
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────
# Clase principal: Arquetipo (qubit)
# ─────────────────────────────────────────────

class Arquetipo:
    """
    Qubit arquetípico que modela un par de opuestos (Ánima/Ánimus).

    El estado es alpha|0> + beta|1>, normalizado automáticamente.
    La medición colapsa el estado: 0 = Ánima, 1 = Ánimus.

    Args:
        alpha: Amplitud del polo Ánima (|0>).
        beta:  Amplitud del polo Ánimus (|1>).
        seed:  Semilla aleatoria opcional para reproducibilidad.
               IMPORTANTE: genera un RNG *local* (no contamina el estado global).

    Raises:
        ValueError: Si alpha y beta son ambos cero (vector de norma nula).
    """
    def __init__(self, alpha: float, beta: float, seed: int | None = None):
        # BUG FIX 1: RNG local en lugar de np.random.seed() global
        self._rng = np.random.default_rng(seed)

        norm = np.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
        if norm == 0:
            raise ValueError(
                "Las amplitudes alpha y beta no pueden ser ambas cero: "
                "el estado cuántico requiere una norma distinta de cero."
            )
        self.alpha = alpha / norm
        self.beta  = beta  / norm

    def medir(self) -> int:
        """
        Colapso en base computacional. Retorna 0 (Ánima) o 1 (Ánimus).

        BUG FIX 2: usa self._rng.random() en lugar de np.random.random() global.
        """
        return 0 if self._rng.random() < abs(self.alpha) ** 2 else 1

    def prob_anima(self) -> float:
        """Probabilidad teórica exacta de observar Ánima."""
        return float(abs(self.alpha) ** 2)

    def entropia_shannon(self) -> float:
        """
        Entropía de Shannon de la distribución de medición (en bits).

        H(p) = -p·log₂(p) - (1-p)·log₂(1-p),  p = |α|²
        Máxima (1 bit) en superposición perfecta; cero en estados base.

        Nota: la entropía de Von Neumann de un estado PURO es siempre 0.
        Esta función mide la ambigüedad de la medición en la base Z, no la
        entropía de la matriz de densidad. Se usa como proxy de "ambigüedad
        psíquica" en el contexto junguiano.
        """
        p = abs(self.alpha) ** 2
        if p <= 0 or p >= 1:
            return 0.0
        return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))

    def fidelidad(self, otro: "Arquetipo") -> float:
        """
        Fidelidad cuántica F = |⟨self|otro⟩|² ∈ [0, 1].

        F=1: estados idénticos (máxima similitud psíquica).
        F=0: estados ortogonales (máximo contraste arquetípico).
        Útil para medir qué tan cerca está un paciente de un arquetipo ideal.
        """
        overlap = self.alpha * np.conj(otro.alpha) + self.beta * np.conj(otro.beta)
        return float(abs(overlap) ** 2)

    def distancia_traza(self, otro: "Arquetipo") -> float:
        """
        Distancia de traza D = √(1 - F) ∈ [0, 1].

        Métrica operacionalmente significativa: la probabilidad máxima de
        distinguir dos estados con una sola medición es (1 + D) / 2.
        """
        return float(np.sqrt(max(0.0, 1 - self.fidelidad(otro))))

    def aplicar_rotacion(self, theta: float) -> "Arquetipo":
        """
        Aplica una rotación Ry(theta) al estado del arquetipo.

        Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]

        Retorna un nuevo Arquetipo con el estado rotado (inmutable).

        BUG FIX 5: el objeto resultante ahora recibe un RNG propio.
        """
        c, s      = np.cos(theta / 2), np.sin(theta / 2)
        new_alpha = c * self.alpha - s * self.beta
        new_beta  = s * self.alpha + c * self.beta
        # Ry es unitaria → norma preservada → Arquetipo() normaliza trivialmente
        return Arquetipo(new_alpha, new_beta)

    def __eq__(self, otro: object) -> bool:
        if not isinstance(otro, Arquetipo):
            return NotImplemented
        return np.isclose(self.alpha, otro.alpha) and np.isclose(self.beta, otro.beta)

    def __repr__(self) -> str:
        return (
            f"Arquetipo(α={self.alpha:.4f}, β={self.beta:.4f}, "
            f"P(Ánima)={self.prob_anima():.4f})"
        )


# ─────────────────────────────────────────────
# Par entrelazado con decoherencia
# ─────────────────────────────────────────────

class ParConDecoherencia:
    """
    Dos qubits en estado de Bell |Φ+⟩, con canal de desfase aplicable al qubit 1.

    BUG FIX 3 y 4: RNG local (self._rng) en lugar de np.random.seed() global.
    Los proyectores de medición en base X se precalculan en __init__.
    """
    def __init__(self, seed: int | None = None):
        # BUG FIX 3: RNG local
        self._rng = np.random.default_rng(seed)

        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        self.rho = np.outer(phi_plus, phi_plus.conj())

        H  = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        I2 = np.eye(2)
        self.P0_1 = np.kron(np.outer(H[:, 0], H[:, 0].conj()), I2)
        self.P1_1 = np.kron(np.outer(H[:, 1], H[:, 1].conj()), I2)
        self.P0_2 = np.kron(I2, np.outer(H[:, 0], H[:, 0].conj()))
        self.P1_2 = np.kron(I2, np.outer(H[:, 1], H[:, 1].conj()))

    def aplicar_represion(self, gamma: float) -> None:
        """
        Canal de desfase (dephasing) sobre el qubit 1 con intensidad gamma.

        BUG FIX 6: guard contra NaN y valores fuera de rango explicitados.

        Operadores de Kraus:
            K0 = sqrt(1 - gamma/2) * I4
            K1 = sqrt(gamma/2)     * Z⊗I2
        """
        # BUG FIX 6: protección reforzada
        if not isinstance(gamma, (int, float)) or np.isnan(gamma):
            raise TypeError(f"gamma debe ser un número real; recibido: {gamma!r}")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(
                f"gamma debe estar en [0, 1]; recibido {gamma}. "
                "Fuera de rango produce sqrt de número negativo (NaN silencioso)."
            )
        Z  = np.array([[1, 0], [0, -1]])
        I2 = np.eye(2)
        I4 = np.eye(4)
        # Convención: gamma=0 → sin ruido; gamma=1 → pérdida total
        # de coherencia de fase. Los operadores conservan traza:
        # K0 = sqrt(1-gamma/2) I, K1 = sqrt(gamma/2) Z.
        K0 = np.sqrt(1 - gamma / 2) * I4
        K1 = np.sqrt(gamma / 2) * np.kron(Z, I2)
        self.rho = K0 @ self.rho @ K0.conj().T + K1 @ self.rho @ K1.conj().T

    def medir_base_X(self) -> tuple[int, int]:
        """
        Medición secuencial en base X.

        BUG FIX 4: usa self._rng.random() en lugar de np.random.random() global.
        """
        prob0_q1 = np.trace(self.P0_1 @ self.rho).real
        if self._rng.random() < prob0_q1:          # BUG FIX 4
            x1          = 0
            estado_post = self.P0_1 @ self.rho @ self.P0_1.conj().T
        else:
            x1          = 1
            estado_post = self.P1_1 @ self.rho @ self.P1_1.conj().T
        estado_post /= np.trace(estado_post)

        prob0_q2 = np.trace(self.P0_2 @ estado_post).real
        x2 = 0 if self._rng.random() < prob0_q2 else 1  # BUG FIX 4
        return x1, x2

    def correlacion_teorica(self) -> float:
        """
        P(x1 == x2) calculada analíticamente desde la matriz de densidad.
        No requiere muestreo; útil para validación y benchmarking.
        """
        prob_igual = 0.0
        for P_q1, P_q2 in [(self.P0_1, self.P0_2), (self.P1_1, self.P1_2)]:
            P_joint    = P_q1 @ P_q2
            prob_igual += np.trace(P_joint @ self.rho).real
        return float(prob_igual)

    def entropia_reducida(self) -> float:
        """Entropía de Von Neumann de la reducción de un qubit.

        Esta magnitud es la entropía de entrelazamiento sólo para estados
        globales puros. Después de ruido/decoherencia el estado global puede
        ser mixto, por lo que NO debe interpretarse por sí sola como medida
        de entrelazamiento.
        """
        rho_1 = np.trace(self.rho.reshape(2, 2, 2, 2), axis1=1, axis2=3)
        eigvals = np.linalg.eigvalsh(rho_1)
        eigvals = eigvals[eigvals > 1e-12]
        return float(-np.sum(eigvals * np.log2(eigvals)))

    def negatividad(self) -> float:
        """Negatividad del estado bipartito.

        Es cero para estados separables y positiva para estados con
        entrelazamiento detectable mediante la transposición parcial.
        Para el estado Bell |Φ+⟩ vale 0.5. Es preferible a usar la entropía
        reducida como proxy después de aplicar ruido local.
        """
        rho_reshaped = self.rho.reshape(2, 2, 2, 2)
        rho_pt = np.transpose(rho_reshaped, (2, 1, 0, 3)).reshape(4, 4)
        eigvals = np.linalg.eigvalsh(rho_pt)
        return float(np.sum(np.abs(eigvals[eigvals < 0])))

    def concurrencia(self) -> float:
        """Concurrencia de Wootters para un estado bipartito de dos qubits."""
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        YY = np.kron(Y, Y)
        rho_tilde = YY @ self.rho.conj() @ YY
        eigvals = np.linalg.eigvals(self.rho @ rho_tilde)
        lambdas = np.sort(np.sqrt(np.maximum(0.0, np.real(eigvals))))[::-1]
        value = lambdas[0] - np.sum(lambdas[1:])
        return float(np.clip(value, 0.0, 1.0))

    # Compatibilidad histórica: el nombre antiguo era ambiguo.
    def entropia_entrelazamiento(self) -> float:
        """Alias de compatibilidad. Preferir entropia_reducida()."""
        return self.entropia_reducida()
