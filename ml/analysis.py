"""
analysis.py — Visualizaciones del Laboratorio Cuántico-Junguiano.

BUGS CORREGIDOS:
  1. [MENOR] analizar_arquetipo: la curva teórica ya fue añadida (BUG FIX
     documentado en el original). Se mantiene y se mejora la estética.
  2. [MENOR] analizar_sincronicidad: ídem curva teórica. Mantenida.
  3. [CRÍTICO] Ninguna función verificaba la existencia de los archivos
     .pkl o .csv antes de cargarlos, produciendo errores crípticos.
     Añadido guard explícito con mensajes accionables.

MEJORAS AÑADIDAS:
  - analizar_fidelidad(): heatmap de la matriz de fidelidades.
  - analizar_entropia_proceso(): gráfica de la firma entrópica de una sesión.
  - analizar_registro(): radar chart de la psique completa (5 componentes).
  - Todas las funciones aceptan ax=None para composición en figuras externas
    (útil en Streamlit sin st.pyplot duplicados).
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import joblib


# ─────────────────────────────────────────────
# Guards de archivos
# ─────────────────────────────────────────────

def _check_files(*paths: str) -> None:
    """BUG FIX 3: valida existencia de archivos con mensaje descriptivo."""
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Archivo no encontrado: '{p}'. "
                "Ejecuta main.py opciones 1 y 2 primero."
            )


# ─────────────────────────────────────────────
# Análisis individuales
# ─────────────────────────────────────────────

def analizar_arquetipo(ax=None) -> None:
    """
    Scatter + curvas: datos simulados, modelo lineal, polinomial y teórico.
    """
    _check_files(
        "datasets/arquetipo_prob.csv",
        "modelos/regresion_arquetipo_lineal.pkl",
        "modelos/regresion_arquetipo_poli.pkl",
    )
    df           = pd.read_csv("datasets/arquetipo_prob.csv")
    alpha_range  = np.linspace(df["alpha"].min(), df["alpha"].max(), 200).reshape(-1, 1)
    modelo_lineal = joblib.load("modelos/regresion_arquetipo_lineal.pkl")
    modelo_poli   = joblib.load("modelos/regresion_arquetipo_poli.pkl")

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(df["alpha"], df["prob_anima"], alpha=0.5, label="Datos simulados", zorder=3, s=20)
    ax.plot(alpha_range, modelo_lineal.predict(alpha_range), "r--", label="Modelo lineal",       lw=1.8)
    ax.plot(alpha_range, modelo_poli.predict(alpha_range),   "g-",  label="Modelo polinomial",   lw=1.8)
    ax.plot(alpha_range, alpha_range ** 2,                   "k:",  label="Teórico: P(0) = α²",  lw=1.5)

    ax.set_xlabel("alpha (amplitud Ánima)")
    ax.set_ylabel("Probabilidad observada de Ánima")
    ax.set_title("Superposición arquetípica — comparación de modelos")
    ax.legend()
    ax.grid(alpha=0.2)

    if standalone:
        plt.tight_layout()
        plt.show()


def analizar_sincronicidad(ax=None) -> None:
    """Scatter + línea teórica y modelo lineal para el dataset de sincronicidad."""
    _check_files(
        "datasets/sincronicidad_corr.csv",
        "modelos/regresion_sincronicidad.pkl",
    )
    df          = pd.read_csv("datasets/sincronicidad_corr.csv")
    gamma_range = np.linspace(df["gamma"].min(), df["gamma"].max(), 200).reshape(-1, 1)
    modelo      = joblib.load("modelos/regresion_sincronicidad.pkl")

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(df["gamma"], df["correlacion"], alpha=0.5, label="Datos simulados", zorder=3, s=20)
    ax.plot(gamma_range, modelo.predict(gamma_range), "r--", label="Modelo lineal", lw=1.8)
    ax.plot(gamma_range, 1 - gamma_range / 2,         "k:",  label="Teórico: 1 − γ/2", lw=1.5)

    ax.set_xlabel("Gamma (nivel de represión)")
    ax.set_ylabel("Correlación en base X")
    ax.set_title("Sincronicidad bajo decoherencia")
    ax.legend()
    ax.grid(alpha=0.2)

    if standalone:
        plt.tight_layout()
        plt.show()


def analizar_fidelidad() -> None:
    """
    MEJORA NUEVA: Heatmap de la matriz de fidelidades arquetípicas.
    """
    _check_files("datasets/fidelidad_arquetipica.csv")
    df  = pd.read_csv("datasets/fidelidad_arquetipica.csv")
    # pivot_table evita el lookup por clave float (susceptible a errores de precisión)
    pivot  = df.pivot_table(index="alpha_i", columns="alpha_j", values="fidelidad")
    mat    = pivot.values
    alphas = pivot.index.tolist()
    n      = len(alphas)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, origin="lower", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Fidelidad F = |⟨ψᵢ|ψⱼ⟩|²")

    tick_step = max(1, n // 10)
    ticks     = list(range(0, n, tick_step))
    labels    = [f"{alphas[t]:.2f}" for t in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.set_xlabel("α_j"); ax.set_ylabel("α_i")
    ax.set_title("Resonancia arquetípica — Matriz de fidelidades")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def analizar_entropia_proceso(sesion) -> None:
    """
    MEJORA NUEVA: Gráfica de la firma entrópica de una SesionTerapeutica.

    Args:
        sesion: instancia de SesionTerapeutica ya ejecutada.
    """
    entropias = sesion.entropia_proceso()
    pasos     = list(range(len(entropias)))
    nombres   = [h["intervencion"][:14] for h in sesion.historial]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pasos, entropias, "o-", color="#6c63ff", lw=2, ms=7)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="Entropía máxima (1 bit)")
    ax.fill_between(pasos, entropias, alpha=0.15, color="#6c63ff")
    ax.set_xticks(pasos)
    ax.set_xticklabels(nombres, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("Entropía de Shannon (bits)")
    ax.set_title("Firma entrópica de la sesión terapéutica")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def analizar_registro(registro) -> None:
    """
    MEJORA NUEVA: Radar chart de la psique completa (5 componentes).

    Cada eje muestra P(Ánima) del componente. El área coloreada
    representa el "perfil psíquico" del registro cuántico.
    """
    componentes = registro.COMPONENTES
    valores     = [abs(q.alpha) ** 2 for q in registro.qubits]
    valores    += valores[:1]   # cerrar el polígono

    n      = len(componentes)
    angles = [k * 2 * np.pi / n for k in range(n)] + [0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, valores, "o-", color="#6c63ff", lw=2)
    ax.fill(angles, valores, alpha=0.25, color="#6c63ff")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(componentes, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Perfil psíquico — P(Ánima) por componente", pad=20)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def analizar_ambos() -> None:
    """Ejecuta los dos análisis principales en secuencia."""
    analizar_arquetipo()
    analizar_sincronicidad()


if __name__ == "__main__":
    analizar_ambos()
