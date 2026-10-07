"""
streamlit_app.py — Dashboard interactivo del Laboratorio Cuántico-Junguiano.

Ejecutar con:
    streamlit run streamlit_app.py

Secciones:
  1.  Superposición del Arquetipo  — slider de alpha, predicción en tiempo real
  2.  Sincronicidad bajo represión — slider de gamma, correlación estimada
  3.  Registro cuántico            — psique completa con 5 componentes
  4.  Sesión terapéutica           — pipeline de puertas cuánticas interactivo
  5.  Comparación de modelos       — tabla y gráfica de todos los modelos ML
  6.  Grafo de Sincronicidad       — heatmap y grafo networkx de resonancias
  7.  Diagnóstico Bayesiano        — inferencia de α con intervalos de credibilidad
  8.  Canal de Lindblad            — comparación de regímenes de represión T1/T2
  9.  Tomografía Cuántica (QST)    — reconstrucción del vector de Bloch
  10. Informe Clínico (IA)         — informe narrativo generado por Claude API
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Asegurar que el directorio raíz del proyecto esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATASETS_DIR, MODELOS_DIR,
    MODELO_LINEAL, MODELO_POLI, MODELO_SINCRONICIDAD,
    DATASET_ARQUETIPO, DATASET_SINCRONICIDAD,
)

from core.experiments   import Arquetipo, ParConDecoherencia
from core.archetypes    import Persona, Sombra, SiMismo, RegistroCuantico
from core.interventions import (
    apertura_consciente, integracion_parcial, amplificacion,
    proyeccion, SesionTerapeutica,
)

# ─────────────────────────────────────────────
# Configuración global
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Laboratorio Cuántico-Junguiano",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-box {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .big-number {
        font-size: 2.2rem;
        font-weight: 700;
        color: #cba6f7;
    }
    .label {
        font-size: 0.85rem;
        color: #a6adc8;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def cargar_modelos():
    try:
        import joblib
        lin  = joblib.load(MODELO_LINEAL)
        poli = joblib.load(MODELO_POLI)
        sinc = joblib.load(MODELO_SINCRONICIDAD)
        return lin, poli, sinc
    except Exception:
        return None, None, None


def simular_arquetipo(alpha: float, n_shots: int = 1000) -> float:
    beta = np.sqrt(max(0.0, 1 - alpha ** 2))
    arq  = Arquetipo(alpha, beta)
    return sum(1 - arq.medir() for _ in range(n_shots)) / n_shots


def simular_sincronicidad(gamma: float, n_trials: int = 500) -> float:
    par = ParConDecoherencia()
    par.aplicar_represion(gamma)
    iguales = 0
    for _ in range(n_trials):
        x1, x2 = par.medir_base_X()
        if x1 == x2:
            iguales += 1
    return iguales / n_trials


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("🧬 Lab Cuántico-Junguiano")
    st.markdown("---")
    seccion = st.radio(
        "Sección",
        [
            "🔮 Superposición del Arquetipo",
            "🌀 Sincronicidad bajo represión",
            "🧠 Registro cuántico",
            "💊 Sesión terapéutica",
            "📊 Comparación de modelos ML",
            "🕸️ Grafo de Sincronicidad",
            "🔬 Diagnóstico Bayesiano",
            "⚗️ Canal de Lindblad",
            "🔭 Tomografía Cuántica (QST)",
            "📋 Informe Clínico (IA)",
        ],
    )
    st.markdown("---")
    st.caption("Laboratorio para explorar física cuántica y psicología analítica.")


# ═══════════════════════════════════════════════════════
# SECCIÓN 1 — Superposición del Arquetipo
# ═══════════════════════════════════════════════════════

if seccion == "🔮 Superposición del Arquetipo":
    st.header("🔮 Superposición del Arquetipo")
    st.markdown(
        "Variar `α` cambia la amplitud del polo **Ánima** (`|0⟩`). "
        "La probabilidad teórica sigue **P(Ánima) = α²**."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        alpha   = st.slider("α (amplitud Ánima)", 0.0, 1.0, 0.7, 0.01)
        n_shots = st.select_slider("Mediciones (n_shots)", [100, 500, 1000, 5000], value=1000)
        simular = st.button("⚡ Simular medición", use_container_width=True)

        beta      = np.sqrt(max(0.0, 1 - alpha ** 2))
        p_teorica = alpha ** 2
        p_animus  = 1 - p_teorica

        st.metric("P(Ánima) teórica",  f"{p_teorica:.4f}")
        st.metric("P(Ánimus) teórica", f"{p_animus:.4f}")
        st.metric("β (amplitud Ánimus)", f"{beta:.4f}")

        if simular:
            with st.spinner("Simulando..."):
                p_obs = simular_arquetipo(alpha, n_shots)
            st.metric("P(Ánima) observada", f"{p_obs:.4f}", delta=f"{p_obs - p_teorica:+.4f}")

    with col2:
        alpha_range = np.linspace(0, 1, 200)
        fig, ax     = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        ax.plot(alpha_range, alpha_range ** 2, color="#cba6f7", lw=2.5, label="Teórico: P(0) = α²")
        ax.axvline(alpha,     color="#f38ba8", ls="--", lw=1.5, label=f"α = {alpha:.2f}")
        ax.axhline(p_teorica, color="#a6e3a1", ls=":",  lw=1.2, label=f"P(Ánima) = {p_teorica:.4f}")
        ax.scatter([alpha], [p_teorica], color="#f38ba8", s=80, zorder=5)

        ax.set_xlabel("α", color="#cdd6f4")
        ax.set_ylabel("P(Ánima = |0⟩)", color="#cdd6f4")
        ax.set_title("Curva de probabilidad del arquetipo", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        ax.spines[:].set_color("#45475a")
        ax.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4", edgecolor="#45475a")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Tabla teórica")
    alphas = np.arange(0, 1.05, 0.1)
    df_tab = pd.DataFrame({
        "α":         alphas.round(2),
        "β":         np.sqrt(np.maximum(0, 1 - alphas**2)).round(4),
        "P(Ánima)":  (alphas**2).round(4),
        "P(Ánimus)": (1 - alphas**2).round(4),
    })
    st.dataframe(df_tab, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# SECCIÓN 2 — Sincronicidad bajo represión
# ═══════════════════════════════════════════════════════

elif seccion == "🌀 Sincronicidad bajo represión":
    st.header("🌀 Sincronicidad bajo represión (decoherencia)")
    st.markdown(
        "El parámetro `γ` controla la intensidad del canal de desfase sobre el primer qubit. "
        "La probabilidad teórica de coincidencia en base X es **P(x₁=x₂) = 1 − γ/2** "
        "(⟨X⊗X⟩ = 1 − γ; ver docs/EXPERIMENTOS/E002_DECOHERENCIA.md)."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        gamma    = st.slider("γ (nivel de represión)", 0.0, 1.0, 0.3, 0.01)
        n_trials = st.select_slider("Trials por estimación", [200, 500, 1000], value=500)
        simular2 = st.button("⚡ Estimar correlación", use_container_width=True)

        c_teorica = 1 - gamma
        st.metric("Correlación teórica", f"{c_teorica:.4f}")

        if simular2:
            with st.spinner("Simulando entrelazamiento..."):
                c_obs = simular_sincronicidad(gamma, n_trials)
            st.metric("Correlación observada", f"{c_obs:.4f}", delta=f"{c_obs - c_teorica:+.4f}")

        if gamma < 0.2:
            st.success("🌿 Entrelazamiento casi intacto (sincronicidad alta)")
        elif gamma < 0.6:
            st.warning("⚠️ Decoherencia parcial (represión moderada)")
        else:
            st.error("🔴 Decoherencia severa (sincronicidad baja)")

    with col2:
        gammas_range = np.linspace(0, 1, 200)
        fig2, ax2    = plt.subplots(figsize=(7, 4))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#0e1117")

        ax2.plot(gammas_range, 1 - gammas_range / 2, color="#89b4fa", lw=2.5, label="Teórico: 1 − γ/2")
        ax2.axvline(gamma,     color="#f38ba8", ls="--", lw=1.5, label=f"γ = {gamma:.2f}")
        ax2.axhline(c_teorica, color="#a6e3a1", ls=":",  lw=1.2, label=f"Corr = {c_teorica:.4f}")
        ax2.scatter([gamma], [c_teorica], color="#f38ba8", s=80, zorder=5)

        ax2.set_xlabel("γ (represión)", color="#cdd6f4")
        ax2.set_ylabel("Correlación en base X", color="#cdd6f4")
        ax2.set_title("Sincronicidad bajo decoherencia", color="#cdd6f4")
        ax2.tick_params(colors="#cdd6f4")
        ax2.spines[:].set_color("#45475a")
        ax2.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4", edgecolor="#45475a")
        st.pyplot(fig2)
        plt.close(fig2)


# ═══════════════════════════════════════════════════════
# SECCIÓN 3 — Registro cuántico
# ═══════════════════════════════════════════════════════

elif seccion == "🧠 Registro cuántico":
    st.header("🧠 Registro cuántico — Psique completa")
    st.markdown(
        "El **Registro Cuántico** modela los 5 componentes junguianos clásicos "
        "como qubits con amplitudes iniciales características."
    )

    seed_reg = st.number_input("Semilla aleatoria", value=42, min_value=0, max_value=9999)

    if "registro" not in st.session_state or st.button("🔄 Reiniciar registro"):
        st.session_state.registro    = RegistroCuantico(seed=int(seed_reg))
        st.session_state.mediciones  = []

    reg = st.session_state.registro

    coherencia        = reg.coherencia_global()
    componentes_data  = []
    for nombre, qubit in zip(reg.COMPONENTES, reg.qubits):
        componentes_data.append({
            "Componente":   nombre,
            "P(Ánima)":     round(abs(qubit.alpha)**2, 4),
            "P(Ánimus)":    round(abs(qubit.beta)**2,  4),
            "Entropía (H)": round(coherencia[nombre],  4),
        })
    st.dataframe(pd.DataFrame(componentes_data), use_container_width=True, hide_index=True)

    tension = reg.tension_yo_sombra()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Tensión Yo↔Sombra",    f"{tension:.4f}")
    col_b.metric("H(SiMismo)",           f"{coherencia['SiMismo']:.4f} bits")
    col_c.metric("Mediciones realizadas", len(st.session_state.mediciones))

    if st.button("🎲 Medir todos los componentes"):
        m = reg.medir_todo()
        st.session_state.mediciones.append(m)
        st.success(f"Resultado: {m}")

    if st.session_state.mediciones:
        st.markdown("### Historial de mediciones")
        st.dataframe(pd.DataFrame(st.session_state.mediciones), use_container_width=True)


# ═══════════════════════════════════════════════════════
# SECCIÓN 4 — Sesión terapéutica
# ═══════════════════════════════════════════════════════

elif seccion == "💊 Sesión terapéutica":
    st.header("💊 Sesión terapéutica — Pipeline de intervenciones")
    st.markdown(
        "Construí una secuencia de **puertas cuánticas** aplicadas a un arquetipo inicial. "
        "Cada intervención transforma el estado y se registra en el historial."
    )

    col_i, col_s = st.columns([1, 2])

    with col_i:
        alpha_ini = st.slider("α inicial", 0.0, 1.0, 0.95, 0.01, key="terapia_alpha")
        beta_ini  = np.sqrt(max(0.0, 1 - alpha_ini**2))
        st.caption(f"β inicial = {beta_ini:.4f}")

        if "sesion" not in st.session_state or st.button("🔄 Reiniciar sesión"):
            arq_ini = Arquetipo(alpha_ini, beta_ini)
            st.session_state.sesion = SesionTerapeutica(arq_ini)

        sesion = st.session_state.sesion

        st.markdown("#### Añadir intervención")
        intervencion = st.selectbox(
            "Tipo de intervención",
            [
                "apertura_consciente",
                "integracion_parcial",
                "amplificacion",
                "proyeccion_anima",
                "proyeccion_animus",
            ]
        )

        theta_val = np.pi / 4
        polo_val  = 0
        if intervencion in ("integracion_parcial", "amplificacion"):
            theta_deg = st.slider("θ (grados)", 0, 360, 45, 5)
            theta_val = np.deg2rad(theta_deg)
        if intervencion == "amplificacion":
            polo_val = st.radio(
                "Polo a amplificar", [0, 1],
                format_func=lambda x: "Ánima (|0>)" if x == 0 else "Ánimus (|1>)"
            )

        if st.button("➕ Aplicar intervención", use_container_width=True):
            kwargs = {}
            if intervencion in ("integracion_parcial", "amplificacion"):
                kwargs["theta"] = theta_val
            if intervencion == "amplificacion":
                kwargs["polo"] = polo_val
            try:
                sesion.aplicar(intervencion, **kwargs)
                st.success(f"Aplicado: {intervencion}")
            except ValueError as e:
                st.error(str(e))

    with col_s:
        if sesion.historial:
            hist_df = pd.DataFrame(sesion.historial)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

            pasos      = list(range(len(sesion.historial)))
            p_animas   = [h["P_anima"]   for h in sesion.historial]
            p_animus   = [h["P_animus"]  for h in sesion.historial]
            entropias  = [h["entropia"]  for h in sesion.historial]
            etiquetas  = [h["intervencion"][:12] for h in sesion.historial]

            fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(7, 6),
                                               gridspec_kw={"hspace": 0.45})
            for ax in (ax3a, ax3b):
                ax.set_facecolor("#0e1117")
            fig3.patch.set_facecolor("#0e1117")

            ax3a.plot(pasos, p_animas, "o-", color="#cba6f7", lw=2, label="P(Ánima)")
            ax3a.plot(pasos, p_animus, "s-", color="#f38ba8", lw=2, label="P(Ánimus)")
            ax3a.axhline(0.5, color="#45475a", ls=":", lw=1)
            ax3a.set_xticks(pasos)
            ax3a.set_xticklabels(etiquetas, rotation=30, ha="right",
                                 color="#cdd6f4", fontsize=8)
            ax3a.set_ylim(-0.05, 1.05)
            ax3a.set_ylabel("Probabilidad", color="#cdd6f4")
            ax3a.set_title("Evolución del arquetipo", color="#cdd6f4")
            ax3a.tick_params(colors="#cdd6f4")
            ax3a.spines[:].set_color("#45475a")
            ax3a.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4", edgecolor="#45475a",
                        fontsize=8)

            ax3b.plot(pasos, entropias, "^-", color="#a6e3a1", lw=2, ms=7,
                      label="Entropía (ambigüedad psíquica)")
            ax3b.axhline(1.0, color="#45475a", ls="--", lw=1, label="Máximo (1 bit)")
            ax3b.fill_between(pasos, entropias, alpha=0.15, color="#a6e3a1")
            ax3b.set_xticks(pasos)
            ax3b.set_xticklabels(etiquetas, rotation=30, ha="right",
                                 color="#cdd6f4", fontsize=8)
            ax3b.set_ylim(-0.05, 1.15)
            ax3b.set_ylabel("H Shannon (bits)", color="#cdd6f4")
            ax3b.set_title("Índice de individuación (entropía)", color="#cdd6f4")
            ax3b.tick_params(colors="#cdd6f4")
            ax3b.spines[:].set_color("#45475a")
            ax3b.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4", edgecolor="#45475a",
                        fontsize=8)

            st.pyplot(fig3)
            plt.close(fig3)


# ═══════════════════════════════════════════════════════
# SECCIÓN 5 — Comparación de modelos ML
# ═══════════════════════════════════════════════════════

elif seccion == "📊 Comparación de modelos ML":
    st.header("📊 Comparación de modelos ML")
    st.markdown(
        "Compara el rendimiento de los modelos Lineal y Polinomial "
        "sobre ambos datasets. Generá los datos desde el menú o `main.py`."
    )

    entrenar = st.button("🚀 Generar datos y entrenar modelos ahora")
    if entrenar:
        with st.spinner("Generando datasets..."):
            from ml.collect_data import generar_dataset_arquetipo, generar_dataset_sincronicidad
            DATASETS_DIR.mkdir(exist_ok=True)
            generar_dataset_arquetipo()
            generar_dataset_sincronicidad()
        with st.spinner("Entrenando modelos..."):
            from ml.train_regression import entrenar_modelos_arquetipo, entrenar_modelo_sincronicidad
            MODELOS_DIR.mkdir(exist_ok=True)
            lin_a, poli_a = entrenar_modelos_arquetipo()
            lin_s         = entrenar_modelo_sincronicidad()
        st.success("✅ Modelos entrenados y guardados.")

    lin, poli, sinc = cargar_modelos()

    if lin is not None:
        tab1, tab2 = st.tabs(["Arquetipo (P = α²)", "Sincronicidad (P = 1−γ/2)"])

        with tab1:
            try:
                df_a       = pd.read_csv(DATASET_ARQUETIPO)
                x_range    = np.linspace(0, 1, 200).reshape(-1, 1)
                fig4, ax4  = plt.subplots(figsize=(8, 4.5))
                fig4.patch.set_facecolor("#0e1117")
                ax4.set_facecolor("#0e1117")

                try:
                    degree     = poli.named_steps["polynomialfeatures"].degree
                    label_poli = f"Polinomial (grado {degree})"
                except Exception:
                    label_poli = "Polinomial"

                ax4.scatter(df_a["alpha"], df_a["prob_anima"],
                            alpha=0.5, color="#89dceb", s=20, label="Datos simulados", zorder=3)
                ax4.plot(x_range, x_range**2,              "w:", lw=1.5, label="Teórico: α²")
                ax4.plot(x_range, lin.predict(x_range),  color="#f38ba8", lw=1.8, label="Lineal")
                ax4.plot(x_range, poli.predict(x_range), color="#a6e3a1", lw=1.8, label=label_poli)

                ax4.set_xlabel("α", color="#cdd6f4")
                ax4.set_ylabel("P(Ánima)", color="#cdd6f4")
                ax4.set_title("Superposición arquetípica", color="#cdd6f4")
                ax4.tick_params(colors="#cdd6f4")
                ax4.spines[:].set_color("#45475a")
                ax4.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4", edgecolor="#45475a")
                st.pyplot(fig4)
                plt.close(fig4)
            except FileNotFoundError:
                st.warning("Dataset no encontrado. Entrenálo primero.")

        with tab2:
            try:
                df_s       = pd.read_csv(DATASET_SINCRONICIDAD)
                g_range    = np.linspace(0, 1, 200).reshape(-1, 1)
                fig5, ax5  = plt.subplots(figsize=(8, 4.5))
                fig5.patch.set_facecolor("#0e1117")
                ax5.set_facecolor("#0e1117")

                ax5.scatter(df_s["gamma"], df_s["correlacion"],
                            alpha=0.5, color="#89dceb", s=20, label="Datos simulados", zorder=3)
                ax5.plot(g_range, 1 - g_range / 2,        "w:", lw=1.5, label="Teórico: 1−γ/2")
                ax5.plot(g_range, sinc.predict(g_range), color="#f38ba8", lw=1.8, label="Lineal")

                ax5.set_xlabel("γ", color="#cdd6f4")
                ax5.set_ylabel("Correlación en base X", color="#cdd6f4")
                ax5.set_title("Sincronicidad bajo decoherencia", color="#cdd6f4")
                ax5.tick_params(colors="#cdd6f4")
                ax5.spines[:].set_color("#45475a")
                ax5.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4", edgecolor="#45475a")
                st.pyplot(fig5)
                plt.close(fig5)
            except FileNotFoundError:
                st.warning("Dataset no encontrado. Entrenálo primero.")
    else:
        st.info("Hacé clic en '🚀 Generar datos y entrenar modelos ahora' para comenzar.")


# ═══════════════════════════════════════════════════════
# SECCIÓN 6 — Grafo de Sincronicidad
# ═══════════════════════════════════════════════════════

elif seccion == "🕸️ Grafo de Sincronicidad":
    st.header("🕸️ Grafo de Sincronicidad — Red arquetípica")
    st.markdown(
        "La **matriz de fidelidades** entre los 5 componentes de la psique visualiza "
        "la resonancia arquetípica: F alto = componentes muy similares (posible proyección); "
        "F bajo = componentes ortogonales (tensión creativa)."
    )

    seed_g = st.number_input("Semilla", value=42, min_value=0, max_value=9999, key="seed_grafo")
    reg_g  = RegistroCuantico(seed=int(seed_g))
    mat    = reg_g.grafo_sincronicidad()
    nombres = reg_g.COMPONENTES

    col_h, col_n = st.columns(2)

    with col_h:
        st.markdown("#### Heatmap de fidelidades")
        fig_h, ax_h = plt.subplots(figsize=(5, 4))
        fig_h.patch.set_facecolor("#0e1117")
        ax_h.set_facecolor("#0e1117")
        im = ax_h.imshow(mat, cmap="viridis", vmin=0, vmax=1)
        fig_h.colorbar(im, ax=ax_h, label="F = |⟨ψᵢ|ψⱼ⟩|²")
        ax_h.set_xticks(range(5)); ax_h.set_xticklabels(nombres, rotation=45,
                                                          ha="right", color="#cdd6f4", fontsize=8)
        ax_h.set_yticks(range(5)); ax_h.set_yticklabels(nombres, color="#cdd6f4", fontsize=8)
        ax_h.set_title("Resonancia inter-arquetípica", color="#cdd6f4")
        ax_h.tick_params(colors="#cdd6f4")
        ax_h.spines[:].set_color("#45475a")
        st.pyplot(fig_h)
        plt.close(fig_h)

    with col_n:
        st.markdown("#### Grafo de resonancia")
        try:
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(nombres)
            umbral = st.slider("Umbral mínimo de fidelidad", 0.0, 1.0, 0.5, 0.05)
            for i in range(5):
                for j in range(i + 1, 5):
                    if mat[i, j] >= umbral:
                        G.add_edge(nombres[i], nombres[j], weight=float(mat[i, j]))

            fig_n, ax_n = plt.subplots(figsize=(5, 4))
            fig_n.patch.set_facecolor("#0e1117")
            ax_n.set_facecolor("#0e1117")
            pos    = nx.circular_layout(G)
            edges  = list(G.edges(data="weight", default=0))
            widths = [w * 6 for _, _, w in edges] if edges else []
            nx.draw_networkx_nodes(G, pos, ax=ax_n, node_color="#6c63ff",
                                   node_size=700, alpha=0.9)
            nx.draw_networkx_labels(G, pos, ax=ax_n, font_color="#cdd6f4",
                                    font_size=8)
            if edges:
                nx.draw_networkx_edges(G, pos, ax=ax_n, width=widths,
                                       edge_color="#a6e3a1", alpha=0.7)
            ax_n.set_title(f"Aristas con F ≥ {umbral:.2f}", color="#cdd6f4")
            ax_n.axis("off")
            st.pyplot(fig_n)
            plt.close(fig_n)
        except ImportError:
            st.warning("networkx no instalado. Ejecutá: `pip install networkx`")

    st.markdown("### Métricas del registro")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Índice de individuación", f"{reg_g.indice_individuacion():.4f}")
    col_m2.metric("Tensión Yo↔Sombra",        f"{reg_g.tension_yo_sombra():.4f}")
    col_m3.metric("Narrativa",                 reg_g.narrativa_estado()[:60] + "…")
    st.caption(reg_g.narrativa_estado())


# ═══════════════════════════════════════════════════════
# SECCIÓN 7 — Diagnóstico Bayesiano
# ═══════════════════════════════════════════════════════

elif seccion == "🔬 Diagnóstico Bayesiano":
    st.header("🔬 Diagnóstico Arquetipal Bayesiano")
    st.markdown(
        "Infiere la distribución posterior sobre **α** de cada componente psíquico "
        "a partir de N mediciones. Modelo: _obs ~ Bernoulli(p)_, prior: _p ~ Beta(1,1)_ "
        "(uniforme). Posterior analítica: _p | datos ~ Beta(1 + n_ánima, 1 + n_ánimus)_."
    )

    try:
        from analytics.diagnostico import inferir_alpha, diagnosticar_registro, graficar_posterior

        seed_d  = st.number_input("Semilla", value=42, min_value=0, max_value=9999, key="seed_diag")
        n_obs_d = st.slider("Observaciones por componente", 20, 500, 200, 10)
        correr  = st.button("🔬 Ejecutar diagnóstico")

        if correr or "diag_resultados" in st.session_state:
            if correr:
                reg_d = RegistroCuantico(seed=int(seed_d))
                st.session_state.diag_resultados  = diagnosticar_registro(reg_d, n_obs=n_obs_d)
                st.session_state.diag_reg         = reg_d

            res_d = st.session_state.diag_resultados
            reg_d = st.session_state.diag_reg

            filas = []
            for nom, r in res_d.items():
                filas.append({
                    "Componente":  nom,
                    "α_MAP":       r["alpha_MAP"],
                    "p_media":     r["p_media"],
                    "IC 95% α":    f"[{r['IC_95_alpha'][0]}, {r['IC_95_alpha'][1]}]",
                    "n_obs":       r["n_obs"],
                })
            st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

            st.markdown("### α real vs α estimado")
            alphas_reales = [float(abs(q.alpha)) for q in reg_d.qubits]
            alphas_mape   = [r["alpha_MAP"] for r in res_d.values()]
            comp_df = pd.DataFrame({
                "Componente": reg_d.COMPONENTES,
                "α real":     [round(a, 4) for a in alphas_reales],
                "α MAP":      alphas_mape,
                "Error":      [round(abs(a - b), 4) for a, b in
                               zip(alphas_reales, alphas_mape)],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            st.markdown("### Distribución posterior detallada")
            comp_sel = st.selectbox("Componente", reg_d.COMPONENTES, key="comp_post")
            idx_sel  = reg_d.COMPONENTES.index(comp_sel)
            qubit_sel = reg_d.qubits[idx_sel]
            obs_sel   = [qubit_sel.medir() for _ in range(n_obs_d)]

            fig_p, ax_p = plt.subplots(figsize=(7, 3.5))
            fig_p.patch.set_facecolor("#0e1117")
            ax_p.set_facecolor("#0e1117")
            ax_p.tick_params(colors="#cdd6f4")
            ax_p.spines[:].set_color("#45475a")
            graficar_posterior(obs_sel, titulo=f"Posterior — {comp_sel}", ax=ax_p)
            ax_p.set_xlabel(ax_p.get_xlabel(), color="#cdd6f4")
            ax_p.set_ylabel(ax_p.get_ylabel(), color="#cdd6f4")
            ax_p.set_title(ax_p.get_title(), color="#cdd6f4")
            st.pyplot(fig_p)
            plt.close(fig_p)

    except ImportError as e:
        st.error(f"No se pudo cargar el módulo de diagnóstico: {e}\n"
                 "Verificá que scipy esté instalado: `pip install scipy`")


# ═══════════════════════════════════════════════════════
# SECCIÓN 8 — Canal de Lindblad
# ═══════════════════════════════════════════════════════

elif seccion == "⚗️ Canal de Lindblad":
    st.header("⚗️ Canal de Lindblad — Represión compleja T1/T2")
    st.markdown(
        "El canal de Lindblad modela dos formas de represión psíquica:\n\n"
        "- **γ₁ (T1 — relajación)**: olvido activo — el contenido se disipa al inconsciente\n"
        "- **γ₂ (T2 — desfase puro)**: interferencia — el contenido existe pero no puede "
        "volverse consciente (bloqueo del insight)\n\n"
        "Compará cómo cada canal y su combinación degradan la sincronicidad arquetípica."
    )

    try:
        from core.lindblad import ParConLindblad, escanear_espacio_lindblad

        tab_comp, tab_mapa = st.tabs(["Comparar canales", "Mapa de represión (γ₁ × γ₂)"])

        with tab_comp:
            col_l1, col_l2 = st.columns([1, 2])
            with col_l1:
                gamma_l = st.slider("γ (intensidad)", 0.0, 1.0, 0.5, 0.05, key="lindblad_g")
                n_rep_l = st.select_slider("Repeticiones", [50, 100, 150, 200], value=150)
                correr_l = st.button("▶ Comparar canales", use_container_width=True)

            with col_l2:
                if correr_l:
                    with st.spinner("Simulando los tres canales..."):
                        gamma_vals = np.linspace(0, 1, 21).tolist()
                        rng_l = np.random.default_rng(42)

                        def _corr(par, n):
                            return sum(
                                1 for _ in range(n)
                                if (r := par.medir_base_X())[0] == r[1]
                            ) / n

                        corr_z, corr_t1, corr_mix = [], [], []
                        for g in gamma_vals:
                            sp = int(rng_l.integers(0, 99999))

                            par = ParConDecoherencia(seed=sp)
                            par.aplicar_represion(float(g))
                            corr_z.append(_corr(par, n_rep_l))

                            par = ParConLindblad(seed=sp)
                            par.aplicar_represion_lindblad(float(g), 0.0)
                            corr_t1.append(_corr(par, n_rep_l))

                            par = ParConLindblad(seed=sp)
                            par.aplicar_represion_lindblad(float(g) / 2, float(g) / 2)
                            corr_mix.append(_corr(par, n_rep_l))

                        fig_l, ax_l = plt.subplots(figsize=(8, 4))
                        fig_l.patch.set_facecolor("#0e1117")
                        ax_l.set_facecolor("#0e1117")
                        ax_l.plot(gamma_vals, corr_z,   "o-", color="#89b4fa", lw=2, ms=5,
                                  label="Desfase puro Z (T2)")
                        ax_l.plot(gamma_vals, corr_t1,  "s-", color="#f38ba8", lw=2, ms=5,
                                  label="Relajación T1")
                        ax_l.plot(gamma_vals, corr_mix, "^-", color="#a6e3a1", lw=2, ms=5,
                                  label="Canal mixto (T1/2 + T2/2)")
                        ax_l.plot(gamma_vals, [1 - g / 2 for g in gamma_vals],
                                  color="white", ls=":", lw=1.5, label="Teórico Z: 1−γ/2")
                        ax_l.set_xlabel("γ", color="#cdd6f4")
                        ax_l.set_ylabel("Correlación en base X", color="#cdd6f4")
                        ax_l.set_title("Comparación de canales de represión", color="#cdd6f4")
                        ax_l.tick_params(colors="#cdd6f4")
                        ax_l.spines[:].set_color("#45475a")
                        ax_l.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4",
                                    edgecolor="#45475a", fontsize=8)
                        ax_l.grid(alpha=0.15)
                        st.pyplot(fig_l)
                        plt.close(fig_l)
                else:
                    st.info("Hacé clic en '▶ Comparar canales' para simular.")

        with tab_mapa:
            n_pts = st.slider("Resolución de la grilla (n × n)", 5, 15, 8)
            n_tri = st.select_slider("Trials por punto", [50, 100, 200], value=100)
            correr_mapa = st.button("🗺️ Generar mapa Lindblad", use_container_width=True)

            if correr_mapa:
                with st.spinner(f"Escaneando grilla {n_pts}×{n_pts} (~{n_pts*n_pts} simulaciones)..."):
                    g1v, g2v, mat_l = escanear_espacio_lindblad(n_puntos=n_pts, n_trials=n_tri)

                fig_m, ax_m = plt.subplots(figsize=(6, 5))
                fig_m.patch.set_facecolor("#0e1117")
                ax_m.set_facecolor("#0e1117")
                im_m = ax_m.imshow(mat_l, origin="lower", cmap="RdYlGn",
                                   extent=[0, 1, 0, 1], vmin=0, vmax=1, aspect="auto")
                plt.colorbar(im_m, ax=ax_m, label="Correlación en base X")
                ax_m.set_xlabel("γ₂ — desfase puro (T2)", color="#cdd6f4")
                ax_m.set_ylabel("γ₁ — relajación (T1)", color="#cdd6f4")
                ax_m.set_title("Mapa de represión Lindblad", color="#cdd6f4")
                ax_m.tick_params(colors="#cdd6f4")
                ax_m.spines[:].set_color("#45475a")
                st.pyplot(fig_m)
                plt.close(fig_m)
            else:
                st.info("Hacé clic en '🗺️ Generar mapa Lindblad' para escanear la grilla.")

    except ImportError as e:
        st.error(f"Error importando el módulo Lindblad: {e}")


# ═══════════════════════════════════════════════════════
# SECCIÓN 9 — Tomografía Cuántica (QST)
# ═══════════════════════════════════════════════════════

elif seccion == "🔭 Tomografía Cuántica (QST)":
    st.header("🔭 Tomografía de Estado Cuántico (QST)")
    st.markdown(
        "La **tomografía cuántica** reconstruye el vector de Bloch de un arquetipo "
        "a partir de mediciones en 3 bases: Z (polarización), X (coherencia), Y (fase).\n\n"
        "Para arquetipos con amplitudes reales, `ry ≈ 0` siempre."
    )

    try:
        from analytics.qst import (
            tomografia_bloch, medir_base_x, medir_base_y,
            reconstruir_arquetipo, graficar_esfera_bloch_2d,
        )

        col_q1, col_q2 = st.columns([1, 2])

        with col_q1:
            alpha_q = st.slider("α del arquetipo a reconstruir", 0.01, 0.99, 0.7, 0.01)
            n_obs_q = st.select_slider("Mediciones por base", [100, 200, 300, 500], value=300)
            correr_q = st.button("🔭 Ejecutar tomografía", use_container_width=True)

        if correr_q:
            beta_q  = float(np.sqrt(1.0 - alpha_q**2))
            arq_q   = Arquetipo(alpha_q, beta_q, seed=42)

            with st.spinner("Simulando mediciones en 3 bases..."):
                obs_z = [arq_q.medir()              for _ in range(n_obs_q)]
                obs_x = medir_base_x(arq_q, n_obs_q, seed=7)
                obs_y = medir_base_y(arq_q, n_obs_q, seed=13)
                res_q = tomografia_bloch(obs_z, obs_x, obs_y)
                arq_r = reconstruir_arquetipo(res_q)

            with col_q2:
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("rx (coherencia)", f"{res_q['rx']:+.4f}")
                col_r2.metric("rz (polarización)", f"{res_q['rz']:+.4f}")
                col_r3.metric("Pureza", f"{res_q['pureza']:.4f}")

                col_a1, col_a2, col_a3 = st.columns(3)
                col_a1.metric("α original", f"{alpha_q:.4f}")
                col_a2.metric("α reconstruido", f"{arq_r.alpha:.4f}")
                col_a3.metric("Error |Δα|", f"{abs(alpha_q - arq_r.alpha):.4f}")

                # Visualización esfera de Bloch
                rz_real = 2.0 * arq_q.prob_anima() - 1.0
                rx_real = 2.0 * arq_q.alpha * arq_q.beta

                fig_q, ax_q = plt.subplots(figsize=(5, 5))
                fig_q.patch.set_facecolor("#0e1117")
                ax_q.set_facecolor("#0e1117")
                ax_q.tick_params(colors="#cdd6f4")
                ax_q.spines[:].set_color("#45475a")

                graficar_esfera_bloch_2d(
                    [{"rx": res_q["rx"], "rz": res_q["rz"]},
                     {"rx": rx_real,     "rz": rz_real}],
                    etiquetas=["QST reconstruido", "Estado real"],
                    ax=ax_q,
                )
                ax_q.set_xlabel(ax_q.get_xlabel(), color="#cdd6f4")
                ax_q.set_ylabel(ax_q.get_ylabel(), color="#cdd6f4")
                ax_q.set_title(ax_q.get_title(), color="#cdd6f4")
                st.pyplot(fig_q)
                plt.close(fig_q)

                st.markdown("### Detalle de mediciones")
                st.dataframe(pd.DataFrame({
                    "Base":       ["Z (Ánima/Ánimus)", "X (coherencia)", "Y (fase)"],
                    "n_meds":     [res_q["n_z"], res_q["n_x"], res_q["n_y"]],
                    "r estimado": [res_q["rz"], res_q["rx"], res_q["ry"]],
                }), use_container_width=True, hide_index=True)
        else:
            with col_q2:
                st.info("Ajustá los parámetros y hacé clic en '🔭 Ejecutar tomografía'.")

    except ImportError as e:
        st.error(f"Error importando el módulo QST: {e}")


# ═══════════════════════════════════════════════════════
# SECCIÓN 10 — Informe Clínico (IA)
# ═══════════════════════════════════════════════════════

elif seccion == "📋 Informe Clínico (IA)":
    st.header("📋 Informe Clínico — Interpretación por Claude")
    st.markdown(
        "Genera un **informe clínico narrativo** en lenguaje natural a partir del estado "
        "cuántico de un RegistroCuantico y una sesión terapéutica opcional. "
        "Requiere `ANTHROPIC_API_KEY` configurada como variable de entorno."
    )

    with st.expander("⚙️ Configuración"):
        seed_i  = st.number_input("Semilla del registro", value=42, min_value=0,
                                   max_value=9999, key="seed_inf")
        alpha_i = st.slider("α inicial de la sesión (opcional)", 0.0, 1.0, 0.90, 0.01,
                             key="alpha_inf")
        usar_sesion = st.checkbox("Incluir sesión terapéutica en el informe", value=True)
        modelo_inf  = st.selectbox("Modelo de Claude",
                                    ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
                                    key="modelo_inf")

    generar = st.button("📋 Generar informe clínico", use_container_width=True)

    if generar:
        try:
            from analytics.informe_analitico import generar_informe_con_cache

            reg_i = RegistroCuantico(seed=int(seed_i))
            sesion_i = None
            if usar_sesion:
                beta_i   = float(np.sqrt(max(0.0, 1.0 - alpha_i**2)))
                arq_i    = Arquetipo(alpha_i, beta_i)
                sesion_i = SesionTerapeutica(arq_i)
                sesion_i.aplicar("apertura_consciente")
                sesion_i.aplicar("integracion_parcial", theta=np.pi / 4)

            with st.spinner("Consultando Claude API..."):
                informe = generar_informe_con_cache(reg_i, sesion_i, modelo=modelo_inf)

            st.success("✅ Informe generado")
            st.markdown("---")
            st.markdown(informe)
            st.markdown("---")
            st.download_button(
                "⬇️ Descargar informe (.txt)",
                data=informe,
                file_name="informe_clinico.txt",
                mime="text/plain",
            )

        except ImportError as e:
            st.error(f"Error de importación: {e}")
        except ValueError as e:
            st.error(str(e))
            st.info(
                "Para configurar la API key en Windows PowerShell:\n\n"
                "```powershell\n$env:ANTHROPIC_API_KEY = 'tu-clave-aqui'\n```\n\n"
                "Luego relanzá Streamlit desde la misma terminal."
            )
        except Exception as e:
            st.error(f"Error al generar el informe: {e}")
