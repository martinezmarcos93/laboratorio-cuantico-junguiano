"""
lindblad.py — Canal de Lindblad generalizado para represión compleja.

Propuesto en INFORME_ANALISIS.md (MEJORA 7).

Extiende ParConDecoherencia con dos mecanismos distintos de represión:

  γ₁ (relajación / T1):   olvido activo — el contenido se disipa al inconsciente
                           Operador: L1 = √γ₁ · (σ₋ ⊗ I)   donde σ₋|1⟩ = |0⟩
                           Junguiano: contenido que "cae" al inconsciente sin dejar huella.

  γ₂ (desfase puro / T2): interferencia — el contenido permanece pero pierde coherencia
                           Operador: L2 = √(γ₂/2) · (σz ⊗ I)
                           Junguiano: represión que no borra el contenido sino que impide
                           el "insight" — el arquetipo existe pero no puede volverse consciente.

La ecuación maestra de Lindblad integrada con Euler:
    ρ(t + dt) = ρ(t) + dt · Σ_k [Lk ρ Lk† − ½{Lk†Lk, ρ}]
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .experiments import ParConDecoherencia


class ParConLindblad(ParConDecoherencia):
    """
    Par entrelazado con canal de Lindblad de dos parámetros.

    Drop-in replacement de ParConDecoherencia que añade el canal completo.
    El canal original (solo desfase) equivale a γ₁=0, γ₂=gamma.
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
            dt:     tiempo total de evolución
            pasos:  subdivisiones del intervalo (más pasos = integración más precisa)
        """
        for nombre, val in (("gamma1", gamma1), ("gamma2", gamma2)):
            if not isinstance(val, (int, float)) or np.isnan(val):
                raise TypeError(f"{nombre} debe ser un número real; recibido: {val!r}")
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{nombre} debe estar en [0, 1]; recibido {val}.")

        I2        = np.eye(2)
        sig_minus = np.array([[0, 1], [0, 0]])   # |0><1|: |1> → |0>
        sig_z     = np.array([[1, 0], [0, -1]])

        L1 = np.sqrt(gamma1)       * np.kron(sig_minus, I2)
        L2 = np.sqrt(gamma2 / 2.0) * np.kron(sig_z, I2)
        operadores = [L for L in (L1, L2) if np.any(L != 0)]

        dt_paso = dt / pasos
        for _ in range(pasos):
            drho = sum(
                L @ self.rho @ L.conj().T
                - 0.5 * (L.conj().T @ L @ self.rho + self.rho @ L.conj().T @ L)
                for L in operadores
            )
            self.rho = self.rho + drho * dt_paso
            # Re-normalizar para corregir errores acumulados del método de Euler
            tr = float(np.trace(self.rho).real)
            if tr > 1e-12:
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
    ax.plot(gamma_vals, [1 - g for g in gamma_vals],
            "k:", lw=1.5, label="Teórico Z: 1 − γ")
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
