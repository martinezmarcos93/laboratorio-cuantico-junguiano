"""
lindblad.py — Canal de Lindblad generalizado para represión compleja.

Propuesto en INFORME_ANALISIS.md (MEJORA 7).

Extiende ParConDecoherencia con dos mecanismos distintos de ruido. Las lecturas
"represión", "olvido" o "insight" son analogías de modelado, no equivalencias
físicas ni clínicas.

  γ₁ (relajación / T1):   olvido activo — el contenido se disipa al inconsciente
                           Operador: L1 = √γ₁ · (σ₋ ⊗ I)   donde σ₋|1⟩ = |0⟩
                           Junguiano: contenido que "cae" al inconsciente sin dejar huella.

  γ₂ (desfase puro / T2): interferencia — el contenido permanece pero pierde coherencia
                           Operador: L2 = √(γ₂/2) · (σz ⊗ I)
                           Junguiano: represión que no borra el contenido sino que impide
                           el "insight" — el arquetipo existe pero no puede volverse consciente.

Ecuación maestra de Lindblad, dρ/dt = Σ_k [Lk ρ Lk† − ½{Lk†Lk, ρ}], resuelta de
forma exacta con la exponencial del superoperador: ρ(t) = exp(𝓛 t) ρ(0).

Solución analítica para |Φ+⟩ (verificada en tests/test_lindblad.py), con t = dt:
    ρ[00,00] = 1/2          ρ[11,11] = e^{−γ₁t}/2      ρ[01,01] = (1 − e^{−γ₁t})/2
    ρ[00,11] = ρ[11,00] = ½ · e^{−γ₁t/2} · e^{−γ₂t}

Relación con el canal de Kraus de `ParConDecoherencia.aplicar_represion(γ)`:
ese canal multiplica la coherencia por (1 − γ); el desfase de Lindblad la
multiplica por e^{−γ₂t}. NO son el mismo canal con γ₂ = γ: coinciden cuando
γ₂·t = −ln(1 − γ). En particular γ = 1 (decoherencia total) exige γ₂·t → ∞.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from .experiments import ParConDecoherencia


class ParConLindblad(ParConDecoherencia):
    """
    Par entrelazado con canal de Lindblad de dos parámetros.

    Extiende ParConDecoherencia con el canal completo. El canal de Kraus
    original equivale a γ₁ = 0 y γ₂·dt = −ln(1 − gamma), no a γ₂ = gamma.
    """

    def aplicar_represion_lindblad(
        self,
        gamma1: float,
        gamma2: float,
        dt: float = 1.0,
        pasos: int = 10,
    ) -> None:
        """
        Canal de Lindblad con relajación (γ₁) y desfase puro (γ₂).

        Args:
            gamma1: tasa de relajación [0, 1]  — olvido activo (T1)
            gamma2: tasa de desfase puro [0, 1] — interferencia bloqueada (T2)
            dt:     tiempo total de evolución (>= 0)
            pasos:  sin efecto sobre el resultado. Se conserva y se valida por
                    compatibilidad con la versión que integraba por Euler; la
                    evolución actual es exacta e independiente de este valor.
        """
        for nombre, val in (("gamma1", gamma1), ("gamma2", gamma2)):
            if not isinstance(val, (int, float, np.integer, np.floating)) or not np.isfinite(val):
                raise TypeError(f"{nombre} debe ser un número real finito; recibido: {val!r}")
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{nombre} debe estar en [0, 1]; recibido {val}.")
        if not isinstance(dt, (int, float, np.integer, np.floating)) or not np.isfinite(dt):
            raise TypeError(f"dt debe ser un número real finito; recibido: {dt!r}")
        if dt < 0:
            raise ValueError(f"dt debe ser >= 0; recibido {dt}.")
        if not isinstance(pasos, (int, np.integer)) or pasos < 1:
            raise ValueError(f"pasos debe ser un entero >= 1; recibido {pasos}.")
        if dt == 0 or (gamma1 == 0 and gamma2 == 0):
            return

        I2        = np.eye(2)
        sig_minus = np.array([[0, 1], [0, 0]])   # |0><1|: |1> → |0>
        sig_z     = np.array([[1, 0], [0, -1]])

        L1 = np.sqrt(gamma1)       * np.kron(sig_minus, I2)
        L2 = np.sqrt(gamma2 / 2.0) * np.kron(sig_z, I2)
        operadores = [L for L in (L1, L2) if np.any(L != 0)]

        # Evolución exacta para un generador constante:
        # vec(ρ(t)) = exp(L_super * t) vec(ρ(0)).
        # Esto evita los artefactos de positividad que puede introducir Euler
        # para pasos grandes y hace que "pasos" sea sólo un parámetro de
        # compatibilidad/precisión histórica, no parte de la física simulada.
        dim = self.rho.shape[0]
        superoperador = np.zeros((dim * dim, dim * dim), dtype=complex)
        identidad = np.eye(dim, dtype=complex)
        for L in operadores:
            A = L.conj().T @ L
            superoperador += (
                np.kron(L.conj(), L)
                - 0.5 * np.kron(identidad, A)
                - 0.5 * np.kron(A.T, identidad)
            )

        rho_vec = self.rho.reshape(-1, order="F")
        rho_vec = expm(superoperador * float(dt)) @ rho_vec
        self.rho = rho_vec.reshape((dim, dim), order="F")

        # Corrección únicamente numérica de hermiticidad/traza.
        self.rho = (self.rho + self.rho.conj().T) / 2.0
        tr = float(np.trace(self.rho).real)
        if tr <= 1e-12:
            raise RuntimeError("La evolución de Lindblad produjo una traza numéricamente nula.")
        self.rho /= tr

    def metricas(self) -> dict:
        """Estado resumido del par: entropía de entrelazamiento + correlación."""
        return {
            "entropia_reducida": round(self.entropia_reducida(), 4),
            "negatividad":        round(self.negatividad(), 4),
            "concurrencia":       round(self.concurrencia(), 4),
            "correlacion_teorica": round(self.correlacion_teorica(), 4),
        }


# ─────────────────────────────────────────────
# Escaneo del espacio de represión (γ₁, γ₂)
# ─────────────────────────────────────────────

def escanear_espacio_lindblad(
    n_puntos: int = 11,
    n_trials: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera una grilla (γ₁ × γ₂) de correlaciones observadas en base X.

    Permite visualizar cómo interactúan el olvido activo y el desfase puro
    para degradar la sincronicidad arquetípica.

    Returns:
        (gamma1_vals, gamma2_vals, mat_correlacion) donde mat[i, j] es la
        correlación promedio para (gamma1_vals[i], gamma2_vals[j]).
    """
    g_vals = np.linspace(0, 1, n_puntos)
    mat    = np.zeros((n_puntos, n_puntos))
    rng    = np.random.default_rng(seed)

    for i, g1 in enumerate(g_vals):
        for j, g2 in enumerate(g_vals):
            par = ParConLindblad(seed=int(rng.integers(0, 99999)))
            par.aplicar_represion_lindblad(float(g1), float(g2))
            iguales = sum(
                1 for _ in range(n_trials)
                if (lambda r: r[0] == r[1])(par.medir_base_X())
            )
            mat[i, j] = iguales / n_trials

    return g_vals, g_vals, mat


def graficar_espacio_lindblad(
    g1_vals: np.ndarray,
    g2_vals: np.ndarray,
    mat: np.ndarray,
    ax=None,
) -> None:
    """Heatmap del espacio de represión Lindblad: correlación vs (γ₁, γ₂)."""
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        mat,
        origin="lower",
        cmap="RdYlGn",
        extent=[float(g2_vals[0]), float(g2_vals[-1]),
                float(g1_vals[0]), float(g1_vals[-1])],
        vmin=0, vmax=1,
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Correlación en base X")
    ax.set_xlabel("γ₂ — desfase puro (T2 / interferencia)")
    ax.set_ylabel("γ₁ — relajación (T1 / olvido activo)")
    ax.set_title("Mapa de represión Lindblad — sincronicidad residual")

    if standalone:
        plt.tight_layout()
        plt.show()
        plt.close()


def comparar_canales(
    gamma_vals: list[float] | None = None,
    seed: int = 42,
) -> None:
    """
    Compara los tres regímenes de represión en la misma gráfica:
      — Canal de Pauli-Z puro (γ₁=0, γ₂=γ)
      — Relajación pura        (γ₁=γ, γ₂=0)
      — Canal mixto            (γ₁=γ/2, γ₂=γ/2)
    """
    if gamma_vals is None:
        gamma_vals = np.linspace(0, 1, 21).tolist()

    rng   = np.random.default_rng(seed)
    n_rep = 150

    corr_z   = []
    corr_t1  = []
    corr_mix = []

    for g in gamma_vals:
        seed_p = int(rng.integers(0, 99999))

        # Canal Z puro (método original)
        par = ParConDecoherencia(seed=seed_p)
        par.aplicar_represion(float(g))
        corr_z.append(sum(
            1 for _ in range(n_rep) if (lambda r: r[0] == r[1])(par.medir_base_X())
        ) / n_rep)

        # Relajación pura T1
        par = ParConLindblad(seed=seed_p)
        par.aplicar_represion_lindblad(float(g), 0.0)
        corr_t1.append(sum(
            1 for _ in range(n_rep) if (lambda r: r[0] == r[1])(par.medir_base_X())
        ) / n_rep)

        # Canal mixto
        par = ParConLindblad(seed=seed_p)
        par.aplicar_represion_lindblad(float(g) / 2, float(g) / 2)
        corr_mix.append(sum(
            1 for _ in range(n_rep) if (lambda r: r[0] == r[1])(par.medir_base_X())
        ) / n_rep)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(gamma_vals, corr_z,   "o-", color="#89b4fa", lw=2, ms=5, label="Desfase puro Z (T2)")
    ax.plot(gamma_vals, corr_t1,  "s-", color="#f38ba8", lw=2, ms=5, label="Relajación T1")
    ax.plot(gamma_vals, corr_mix, "^-", color="#a6e3a1", lw=2, ms=5, label="Canal mixto (T1/2 + T2/2)")
    ax.plot(gamma_vals, [1 - g / 2 for g in gamma_vals],
            "k:", lw=1.5, label="Teórico Z: 1 − γ/2")
    ax.set_xlabel("γ (intensidad de represión)")
    ax.set_ylabel("Correlación en base X")
    ax.set_title("Comparación de canales de represión Lindblad")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    print("Comparando canales de represión...")
    comparar_canales()

    print("\nEscaneando espacio (γ₁, γ₂)... (puede tardar ~30s)")
    g1, g2, mat = escanear_espacio_lindblad(n_puntos=8, n_trials=200)
    graficar_espacio_lindblad(g1, g2, mat)
