"""
qst.py — Tomografía de Estado Cuántico (QST) para arquetipos.

La tomografía cuántica es la contraparte computacional de la observación clínica:
a partir de múltiples mediciones en distintas bases, reconstruye el estado
psíquico con máxima precisión estadística.

Protocolo de 3 bases para un qubit:
  Base Z: {|0⟩, |1⟩}         → rz = 2·P(0|Z) − 1  (polarización Ánima/Ánimus)
  Base X: {|+⟩, |−⟩}         → rx = 2·P(+|X) − 1  (coherencia entre polos)
  Base Y: {|+i⟩, |−i⟩}       → ry = 2·P(+i|Y) − 1 (fase compleja)

La densidad reconstruida: ρ = (I + rx·X + ry·Y + rz·Z) / 2

Para el Arquetipo con amplitudes REALES: ry ≈ 0 siempre (no hay fase compleja).
La diagonal de ρ da directamente P(Ánima) = (1 + rz) / 2.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from core.experiments import Arquetipo


# ─────────────────────────────────────────────
# Estimación de α
# ─────────────────────────────────────────────

def tomografia_z(observaciones_z: list[int]) -> dict:
    """
    Estimación por máxima verosimilitud (MLE) de p = |α|² solo con base Z.

    La MLE para Bernoulli es simplemente p̂ = k/n (sin prior).
    Converge al valor real con n → ∞.

    Args:
        observaciones_z: lista de 0s (Ánima) y 1s (Ánimus).

    Returns:
        dict con p_mle, alpha_mle, stderr_p, n_obs.
    """
    if not observaciones_z:
        raise ValueError("Se requiere al menos una observación.")

    n     = len(observaciones_z)
    k     = sum(1 for x in observaciones_z if x == 0)
    p_mle = k / n
    err   = float(np.sqrt(p_mle * (1.0 - p_mle) / n)) if n > 0 else 1.0

    return {
        "p_mle":     round(float(p_mle), 4),
        "alpha_mle": round(float(np.sqrt(max(0.0, p_mle))), 4),
        "stderr_p":  round(err, 4),
        "n_obs":     n,
    }


def tomografia_bloch(
    obs_z: list[int],
    obs_x: list[int],
    obs_y: list[int] | None = None,
) -> dict:
    """
    Tomografía completa del qubit: reconstruye el vector de Bloch (rx, ry, rz).

    Para estados con amplitudes reales (Arquetipo), ry ≈ 0.
    obs_y puede ser None — se asume ry = 0 en ese caso.

    Returns:
        dict con rx, ry, rz, r_norm, pureza, rho_rec y conteos.
    """
    def _p0(obs: list[int]) -> float:
        if not obs:
            return 0.5
        return sum(1 for x in obs if x == 0) / len(obs)

    rz = 2.0 * _p0(obs_z) - 1.0
    rx = 2.0 * _p0(obs_x) - 1.0
    ry = 2.0 * _p0(obs_y) - 1.0 if obs_y is not None else 0.0

    # Matriz de densidad reconstruida ρ = (I + r·σ) / 2
    I_ = np.eye(2, dtype=complex)
    X_ = np.array([[0, 1], [1, 0]], dtype=complex)
    Y_ = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z_ = np.array([[1, 0], [0, -1]], dtype=complex)
    rho_rec = (I_ + rx * X_ + ry * Y_ + rz * Z_) / 2.0

    # Proyectar al interior de la esfera si r > 1 (artefacto estadístico)
    r = float(np.sqrt(rx**2 + ry**2 + rz**2))
    if r > 1.0:
        rx /= r; ry /= r; rz /= r
        rho_rec = (I_ + rx * X_ + ry * Y_ + rz * Z_) / 2.0

    pureza = float(np.trace(rho_rec @ rho_rec).real)

    return {
        "rx":      round(float(rx), 4),
        "ry":      round(float(ry), 4),
        "rz":      round(float(rz), 4),
        "r_norm":  round(min(1.0, r), 4),
        "pureza":  round(float(np.clip(pureza, 0.0, 1.0)), 4),
        "rho_rec": rho_rec,
        "n_z":     len(obs_z),
        "n_x":     len(obs_x),
        "n_y":     len(obs_y) if obs_y else 0,
    }


def reconstruir_arquetipo(resultado_bloch: dict) -> Arquetipo:
    """
    Proyección deliberada de una matriz de densidad a un Arquetipo puro.

    Una matriz mixta no puede representarse exactamente mediante dos amplitudes.
    Para análisis de estados mixtos debe utilizarse rho_rec directamente.
    """
    rho = resultado_bloch["rho_rec"]
    p_anima = float(np.clip(np.real(rho[0, 0]), 0.0, 1.0))
    alpha = float(np.sqrt(p_anima))
    beta = float(np.sqrt(1.0 - p_anima))
    return Arquetipo(alpha if alpha > 1e-9 else 1e-9,
                     beta if beta > 1e-9 else 1e-9)


def fidelidad_densidad(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """Fidelidad de Uhlmann entre dos matrices de densidad de un qubit."""
    def sqrtm_psd(rho: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(rho)
        vals = np.clip(vals.real, 0.0, None)
        return (vecs * np.sqrt(vals)) @ vecs.conj().T

    root_a = sqrtm_psd(rho_a)
    middle = root_a @ rho_b @ root_a
    root_middle = sqrtm_psd(middle)
    return float(np.clip(np.trace(root_middle).real ** 2, 0.0, 1.0))


# ─────────────────────────────────────────────
# Simuladores de medición en bases X e Y
# ─────────────────────────────────────────────

def medir_base_z(arq: Arquetipo, n: int, seed: int | None = None) -> list[int]:
    """Simula mediciones Z con RNG local, haciendo reproducible toda la QST."""
    rng = np.random.default_rng(seed)
    p_anima = arq.prob_anima()
    return [0 if rng.random() < p_anima else 1 for _ in range(n)]


def medir_base_x(arq: Arquetipo, n: int, seed: int | None = None) -> list[int]:
    """
    Simula n mediciones del arquetipo en la base X = {|+⟩, |−⟩}.

    P(+) = |⟨+|ψ⟩|² = (α + β)² / 2
    """
    rng   = np.random.default_rng(seed)
    p_mas = (arq.alpha + arq.beta) ** 2 / 2.0
    return [0 if rng.random() < p_mas else 1 for _ in range(n)]


def medir_base_y(arq: Arquetipo, n: int, seed: int | None = None) -> list[int]:
    """
    Simula n mediciones en la base Y = {|+i⟩, |−i⟩}.

    Para amplitudes reales: P(+i) = 0.5 → ry ≈ 0.
    Incluido por completitud del protocolo estándar de tomografía.
    """
    rng = np.random.default_rng(seed)
    return [0 if rng.random() < 0.5 else 1 for _ in range(n)]


# ─────────────────────────────────────────────
# Tomografía de un RegistroCuantico completo
# ─────────────────────────────────────────────

def tomografia_registro(
    registro,
    n_obs: int = 200,
    seed: int | None = None,
) -> dict[str, dict]:
    """
    Aplica tomografía_bloch a todos los componentes del RegistroCuantico.

    Args:
        registro: instancia de RegistroCuantico.
        n_obs:    mediciones por base por componente.
        seed:     semilla para reproducibilidad.

    Returns:
        {nombre_componente: resultado_tomografia_bloch}
    """
    rng = np.random.default_rng(seed)
    resultados = {}
    for nombre, qubit in zip(registro.COMPONENTES, registro.qubits):
        s_z = int(rng.integers(0, 99999))
        s_x = int(rng.integers(0, 99999))
        s_y = int(rng.integers(0, 99999))
        obs_z = medir_base_z(qubit, n_obs, seed=s_z)
        obs_x = medir_base_x(qubit, n_obs, seed=s_x)
        obs_y = medir_base_y(qubit, n_obs, seed=s_y)
        resultados[nombre] = tomografia_bloch(obs_z, obs_x, obs_y)
    return resultados


# ─────────────────────────────────────────────
# Visualización: Esfera de Bloch 2D
# ─────────────────────────────────────────────

def graficar_esfera_bloch_2d(
    estados: list[dict],
    etiquetas: list[str] | None = None,
    ax=None,
) -> None:
    """
    Visualización 2D del plano ecuatorial de la esfera de Bloch (rx, rz).

    eje rz: polarización Ánima (arriba = |0⟩) / Ánimus (abajo = |1⟩)
    eje rx: coherencia de fase entre polos (superposición)

    Estados puros caen sobre el círculo unitario.
    Estados mixtos (decoherentes) caen dentro.
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(6, 6))

    # Círculo de la esfera de Bloch
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="gray",
            lw=1, ls="--", alpha=0.4, label="Esfera de Bloch")

    # Ejes y anotaciones de polos
    ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.3)
    ax.text(0,  1.08, "|0⟩ Ánima",  ha="center", fontsize=9, color="#6c63ff")
    ax.text(0, -1.12, "|1⟩ Ánimus", ha="center", fontsize=9, color="#f38ba8")
    ax.text( 1.05, 0, "|+⟩",        ha="left",   fontsize=9, color="#a6e3a1")
    ax.text(-1.10, 0, "|−⟩",        ha="right",  fontsize=9, color="gray")

    colors = plt.cm.plasma(np.linspace(0.2, 0.85, max(1, len(estados))))
    for idx, (est, color) in enumerate(zip(estados, colors)):
        rx = est.get("rx", 0.0)
        rz = est.get("rz", 0.0)
        lbl = (etiquetas[idx] if etiquetas and idx < len(etiquetas)
               else f"Estado {idx}")
        ax.scatter([rx], [rz], s=130, color=color, zorder=5)
        ax.annotate(lbl, (rx + 0.04, rz + 0.04), fontsize=8, color=color)

    ax.set_xlim(-1.35, 1.45)
    ax.set_ylim(-1.30, 1.25)
    ax.set_xlabel("rx  (coherencia / superposición)")
    ax.set_ylabel("rz  (polarización Ánima − Ánimus)")
    ax.set_title("Esfera de Bloch 2D — estados arquetípicos")
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)

    if standalone:
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == "__main__":
    print("=== TOMOGRAFÍA DE ESTADO CUÁNTICO ===\n")

    arq = Arquetipo(0.70, float(np.sqrt(1.0 - 0.49)), seed=42)
    n   = 300

    obs_z = [arq.medir()            for _ in range(n)]
    obs_x = medir_base_x(arq, n, seed=7)
    obs_y = medir_base_y(arq, n, seed=13)

    res_z = tomografia_z(obs_z)
    print(f"Tomografía Z sola:")
    print(f"  α real = 0.7000   α_MLE = {res_z['alpha_mle']}  ± {res_z['stderr_p']}")

    res_bloch = tomografia_bloch(obs_z, obs_x, obs_y)
    print(f"\nTomografía completa (Bloch):")
    print(f"  rx={res_bloch['rx']:+.4f}  ry={res_bloch['ry']:+.4f}  "
          f"rz={res_bloch['rz']:+.4f}")
    print(f"  pureza = {res_bloch['pureza']:.4f}  (1.0 = estado puro)")

    arq_rec = reconstruir_arquetipo(res_bloch)
    print(f"\n  Arquetipo reconstruido: {arq_rec}")
    print(f"  Arquetipo original:     {arq}")

    # Visualización: estado real vs reconstruido
    rz_real = 2.0 * arq.prob_anima() - 1.0
    rx_real = 2.0 * arq.alpha * arq.beta
    graficar_esfera_bloch_2d(
        [{"rx": res_bloch["rx"], "rz": res_bloch["rz"]},
         {"rx": rx_real, "rz": rz_real}],
        etiquetas=["QST reconstruido", "Estado real"],
    )
