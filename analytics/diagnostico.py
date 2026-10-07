"""
diagnostico.py — Diagnóstico Arquetipal Bayesiano.

Propuesto en INFORME_ANALISIS.md (MEJORA 8).

Infiere la distribución posterior sobre α dado un conjunto de observaciones
conductuales (mediciones binarias: 0 = Ánima, 1 = Ánimus).

Modelo estadístico:
    obs_i ~ Bernoulli(p)     donde p = |α|²
    Prior:     p ~ Beta(a, b)          (a=b=1 → uniforme)
    Posterior: p | datos ~ Beta(a + n_anima, b + n_animus)  [analítica]

La posterior analítica hace innecesario cualquier MCMC y garantiza
convergencia exacta a la distribución de muestreo con n → ∞.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.archetypes import RegistroCuantico
    from core.interventions import SesionTerapeutica


# ─────────────────────────────────────────────
# Inferencia bayesiana del parámetro α
# ─────────────────────────────────────────────

def inferir_alpha(
    observaciones: list[int],
    prior_a: float = 1.0,
    prior_b: float = 1.0,
) -> dict:
    """
    Inferencia bayesiana del parámetro p = |α|² (y por ende de α).

    Args:
        observaciones: lista de 0s (Ánima) y 1s (Ánimus) observados.
        prior_a, prior_b: parámetros del prior Beta (default=1 → uniforme).

    Returns:
        dict con:
            p_media    — media posterior de p = |α|²
            p_moda     — moda posterior (MAP) de p
            alpha_MAP  — √(p_moda). Es la raíz de la moda de p, no la moda
                         posterior de α (la moda no es invariante ante el
                         cambio de variable); ambas coinciden para n grande
            IC_95_p    — intervalo de credibilidad al 95% de p
            IC_95_alpha— IC 95% convertido a α
            n_obs      — número de observaciones usadas
    """
    if not observaciones:
        raise ValueError("Se requiere al menos una observación.")
    if any(x not in (0, 1) for x in observaciones):
        # Cualquier valor distinto de 0 se contaba en silencio como Ánimus.
        raise ValueError("Las observaciones sólo pueden ser 0 (Ánima) o 1 (Ánimus).")
    if not (np.isfinite(prior_a) and np.isfinite(prior_b)) or prior_a <= 0 or prior_b <= 0:
        raise ValueError("prior_a y prior_b deben ser finitos y positivos.")

    n_anima  = sum(1 for x in observaciones if x == 0)
    n_animus = len(observaciones) - n_anima
    a_post   = prior_a + n_anima
    b_post   = prior_b + n_animus
    dist     = beta_dist(a_post, b_post)

    p_media = float(dist.mean())
    if a_post > 1 and b_post > 1:
        p_moda = float((a_post - 1) / (a_post + b_post - 2))
    else:
        p_moda = p_media  # moda no definida en Beta(1, x) o Beta(x, 1)

    ic_lo, ic_hi = dist.interval(0.95)

    return {
        "p_media":     round(p_media, 4),
        "p_moda":      round(p_moda, 4),
        "alpha_MAP":   round(float(np.sqrt(max(0.0, p_moda))), 4),
        "IC_95_p":     (round(float(ic_lo), 4), round(float(ic_hi), 4)),
        "IC_95_alpha": (
            round(float(np.sqrt(max(0.0, ic_lo))), 4),
            round(float(np.sqrt(max(0.0, ic_hi))), 4),
        ),
        "n_obs": len(observaciones),
    }


def diagnosticar_registro(
    registro: "RegistroCuantico",
    n_obs: int = 200,
) -> dict[str, dict]:
    """
    Diagnóstico bayesiano de todos los componentes de un RegistroCuantico.

    Genera n_obs mediciones de cada qubit y corre inferencia bayesiana.
    Retorna {nombre_componente: resultado_inferir_alpha}.
    """
    return {
        nombre: inferir_alpha([qubit.medir() for _ in range(n_obs)])
        for nombre, qubit in zip(registro.COMPONENTES, registro.qubits)
    }


def diagnosticar_sesion(sesion: "SesionTerapeutica") -> dict:
    """
    Diagnóstico del estado final de una SesionTerapeutica.

    Extrae directamente las probabilidades del historial en lugar de
    hacer inferencia probabilística (útil cuando no hay mediciones raw).
    """
    if not sesion.historial:
        raise ValueError("La sesión no tiene historial.")

    estado_final = sesion.historial[-1]
    p_anima      = estado_final["P_anima"]
    alpha_est    = float(np.sqrt(max(0.0, p_anima)))
    entropia     = estado_final["entropia"]

    return {
        "alpha_estimado": round(alpha_est, 4),
        "p_anima_final":  round(p_anima, 4),
        "entropia_final": round(entropia, 4),
        "pasos":          len(sesion.historial) - 1,
        "estado": (
            "integrado"   if entropia > 0.8 else
            "parcial"     if entropia > 0.4 else
            "polarizado"
        ),
    }


# ─────────────────────────────────────────────
# Visualizaciones
# ─────────────────────────────────────────────

def graficar_posterior(
    observaciones: list[int],
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    titulo: str = "Posterior sobre p = |α|²",
    ax=None,
) -> None:
    """Visualiza la distribución posterior Beta con media, MAP e IC 95%."""
    resultado = inferir_alpha(observaciones, prior_a, prior_b)
    n_anima   = sum(1 for x in observaciones if x == 0)
    n_animus  = len(observaciones) - n_anima
    a_post    = prior_a + n_anima
    b_post    = prior_b + n_animus
    dist      = beta_dist(a_post, b_post)

    ps  = np.linspace(0.001, 0.999, 500)
    pdf = dist.pdf(ps)

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(8, 4))

    ic = resultado["IC_95_p"]
    ax.plot(ps, pdf, color="#6c63ff", lw=2,
            label=f"Posterior Beta({a_post:.0f}, {b_post:.0f})")
    ax.axvline(resultado["p_media"], color="#f38ba8", ls="--", lw=1.5,
               label=f"Media: {resultado['p_media']:.4f}")
    ax.axvline(resultado["p_moda"],  color="#a6e3a1", ls="-.",  lw=1.5,
               label=f"MAP:  {resultado['p_moda']:.4f}")
    ax.axvspan(ic[0], ic[1], alpha=0.12, color="#6c63ff",
               label=f"IC 95%: [{ic[0]}, {ic[1]}]")

    ax.set_xlabel("p = |α|² (probabilidad de Ánima)")
    ax.set_ylabel("Densidad posterior")
    ax.set_title(titulo)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    if standalone:
        plt.tight_layout()
        plt.show()
        plt.close()


def convergencia_bayesiana(
    verdadero_alpha: float,
    n_max: int = 300,
    seed: int | None = None,
    ax=None,
) -> None:
    """
    Muestra cómo la estimación de α converge al valor verdadero a medida
    que se acumulan observaciones. Cada 5 obs se recalcula la posterior.

    Junguianamente: el clínico "colapsa la ambigüedad" del arquetipo del
    paciente a medida que acumula sesiones y observaciones.
    """
    from core.experiments import Arquetipo
    rng    = np.random.default_rng(seed)
    beta_v = float(np.sqrt(max(0.0, 1.0 - verdadero_alpha**2)))
    arq    = Arquetipo(verdadero_alpha, beta_v, seed=int(rng.integers(0, 9999)))

    obs    = [arq.medir() for _ in range(n_max)]
    ns, alphas, ic_lo, ic_hi = [], [], [], []

    for n in range(5, n_max + 1, 5):
        r = inferir_alpha(obs[:n])
        ns.append(n)
        alphas.append(r["alpha_MAP"])
        ic_lo.append(r["IC_95_alpha"][0])
        ic_hi.append(r["IC_95_alpha"][1])

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(9, 4))

    ax.plot(ns, alphas, color="#6c63ff", lw=2, label="α estimado (MAP)")
    ax.fill_between(ns, ic_lo, ic_hi, alpha=0.15, color="#6c63ff", label="IC 95%")
    ax.axhline(verdadero_alpha, color="#f38ba8", ls="--", lw=1.5,
               label=f"α verdadero = {verdadero_alpha:.3f}")
    ax.set_xlabel("Número de observaciones")
    ax.set_ylabel("α estimado")
    ax.set_title("Convergencia bayesiana — inferencia del α arquetípico")
    ax.legend()
    ax.grid(alpha=0.2)

    if standalone:
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == "__main__":
    from core.experiments import Arquetipo
    import numpy as np

    arq = Arquetipo(0.70, np.sqrt(1.0 - 0.49), seed=42)
    obs = [arq.medir() for _ in range(150)]

    resultado = inferir_alpha(obs)
    print("=== DIAGNÓSTICO ARQUETIPAL BAYESIANO ===")
    print(f"  n observaciones:  {resultado['n_obs']}")
    print(f"  α estimado (MAP): {resultado['alpha_MAP']}  (real: 0.7000)")
    print(f"  IC 95% de α:      {resultado['IC_95_alpha']}")
    print(f"  p media:          {resultado['p_media']}")

    graficar_posterior(obs, titulo="Diagnóstico del arquetipo (α_real = 0.70)")
    convergencia_bayesiana(0.70, n_max=300)
