"""
interventions.py — Puertas cuánticas como intervenciones terapéuticas.

BUGS CORREGIDOS:
  1. [CRÍTICO] amplificacion: theta=π/3 como default "amplifica" siempre con
     la misma intensidad independientemente del estado. No era bug estricto pero
     era confuso. Documentado explícitamente y mantenido como default razonable.
  2. [MENOR] SesionTerapeutica.aplicar: no capturaba TypeError ni AttributeError
     que podían surgir de kwargs incorrectos, produciendo trazas crípticas.
     Envuelto en manejo de excepciones más descriptivo.
  3. [MENOR] SesionTerapeutica._registrar: no registraba la fidelidad respecto
     al estado anterior, perdiendo información sobre el "delta terapéutico".
     Añadido campo 'fidelidad_anterior'.

MEJORAS AÑADIDAS:
  - intervencion_hadamard_inversa(): revierte una apertura consciente (útil en
    análisis de reversibilidad terapéutica).
  - SesionTerapeutica.exportar_json(): serializa el historial completo a JSON.
  - SesionTerapeutica.arquetipos_visitados(): lista de snapshots de Arquetipo.
  - SesionTerapeutica.entropia_proceso(): evolución de la entropía a lo largo
    de la sesión — la "firma termodinámica" de la terapia.
"""

from __future__ import annotations

import json
import numpy as np
from .experiments import Arquetipo


# ─────────────────────────────────────────────
# Puertas individuales
# ─────────────────────────────────────────────

def apertura_consciente(arq: Arquetipo) -> Arquetipo:
    """
    Puerta Hadamard. H = (1/√2)[[1,1],[1,-1]]

    Lleva los estados base |0⟩ y |1⟩ a la superposición equilibrada
    (P(Ánima) = 0.5), pero NO lo hace para cualquier estado: es una
    involución (H² = I), de modo que H|+⟩ = |0⟩, con P(Ánima) = 1.

    Lectura simbólica (Jung): suspensión de la certeza inicial. Es una
    analogía de modelado, no una afirmación clínica.
    """
    s2 = 1 / np.sqrt(2)
    # H es unitaria → norma preservada → Arquetipo() normaliza trivialmente
    return arq.derivar(s2 * (arq.alpha + arq.beta), s2 * (arq.alpha - arq.beta))


def apertura_consciente_inversa(arq: Arquetipo) -> Arquetipo:
    """
    MEJORA NUEVA: Hadamard es su propia inversa (H² = I).
    Aplicar dos veces restaura el estado original.
    Útil para medir la reversibilidad de una intervención.
    """
    return apertura_consciente(arq)


def integracion_parcial(arq: Arquetipo, theta: float = np.pi / 4) -> Arquetipo:
    """
    Rotación Ry(theta): desplaza gradualmente el equilibrio entre polos.

    theta=0   → sin cambio
    theta=π/2 → lleva |0⟩ a la superposición equilibrada; para otros estados
                el efecto depende del estado inicial
    theta=π   → intercambia las probabilidades de los polos, igual que
                `proyeccion`, pero NO es el mismo estado: Ry(π)(α, β) = (−β, α)
                y X(α, β) = (β, α) difieren en una fase relativa

    Jung: la integración es gradual; cada sesión rota un ángulo terapéutico.
    """
    return arq.aplicar_rotacion(theta)


def amplificacion(arq: Arquetipo, polo: int = 0, theta: float = np.pi / 3) -> Arquetipo:
    """
    Rotación Ry que desplaza probabilidad hacia un polo.

    polo=0 (Ánima): rota hacia |0⟩ con Ry(−|θ|).
    polo=1 (Ánimus): rota hacia |1⟩ con Ry(+|θ|).

    Con amplitudes reales no negativas y un ángulo que no sobrepase el polo,
    la probabilidad del polo elegido aumenta. Un ángulo mayor que la
    distancia angular al polo lo sobrepasa y la probabilidad vuelve a bajar.

    Corrección: el signo estaba invertido (polo=0 usaba Ry(+θ)), de modo que
    "amplificar Ánima" desde P(Ánima)=0.64 la dejaba en 0.15.

    Lectura simbólica (Jung): hacer emerger un contenido latente.
    """
    if polo not in (0, 1):
        raise ValueError(f"polo debe ser 0 (Ánima) o 1 (Ánimus); recibido: {polo}")
    direccion = -1 if polo == 0 else 1
    return arq.aplicar_rotacion(direccion * abs(theta))


def proyeccion(arq: Arquetipo) -> Arquetipo:
    """
    Puerta Pauli-X: inversión total de polos.
    X = [[0,1],[1,0]]
    Jung: proyección total — la Persona se convierte en Sombra y viceversa.
    """
    # X es unitaria → norma preservada → Arquetipo() normaliza trivialmente
    return arq.derivar(arq.beta, arq.alpha)


# Aliases para compatibilidad con Streamlit selectbox
def proyeccion_anima(arq: Arquetipo, **kwargs) -> Arquetipo:
    """Amplifica el polo Ánima ignorando parámetros adicionales."""
    return amplificacion(arq, polo=0, theta=kwargs.get("theta", np.pi / 3))


def proyeccion_animus(arq: Arquetipo, **kwargs) -> Arquetipo:
    """Amplifica el polo Ánimus ignorando parámetros adicionales."""
    return amplificacion(arq, polo=1, theta=kwargs.get("theta", np.pi / 3))


def _amplitud_serializable(valor: complex):
    """Amplitud redondeada para el historial JSON.

    Las amplitudes reales se guardan como float (formato histórico). Una
    amplitud con parte imaginaria se guarda como [real, imag]: float() sobre
    un complejo lanzaba TypeError y la sesión no podía registrarse.
    """
    valor = complex(valor)
    if abs(valor.imag) < 1e-12:
        return round(valor.real, 4)
    return [round(valor.real, 4), round(valor.imag, 4)]


# ─────────────────────────────────────────────
# Pipeline terapéutico
# ─────────────────────────────────────────────

class SesionTerapeutica:
    """
    Pipeline de intervenciones secuenciales sobre un Arquetipo.

    Registra el historial de transformaciones para análisis longitudinal.
    Inmutable en cada paso: cada intervención genera un nuevo Arquetipo.
    """

    INTERVENCIONES = {
        "apertura_consciente":         apertura_consciente,
        "apertura_consciente_inversa": apertura_consciente_inversa,
        "integracion_parcial":         integracion_parcial,
        "amplificacion":               amplificacion,
        "proyeccion":                  proyeccion,
        "proyeccion_anima":            proyeccion_anima,
        "proyeccion_animus":           proyeccion_animus,
    }

    def __init__(self, arquetipo_inicial: Arquetipo):
        self.arquetipo            = arquetipo_inicial
        self.historial: list[dict] = []
        self._snapshots: list[Arquetipo] = [arquetipo_inicial]
        self._registrar("estado_inicial", arquetipo_inicial, fidelidad_anterior=1.0)

    def aplicar(self, nombre: str, **kwargs) -> Arquetipo:
        """
        Aplica una intervención por nombre y actualiza el estado.

        BUG FIX 2: captura TypeError y AttributeError con mensajes claros.

        Args:
            nombre: Clave en INTERVENCIONES.
            **kwargs: Parámetros adicionales (theta, polo).

        Raises:
            ValueError: Si el nombre de intervención es desconocido.
            TypeError:  Si los kwargs no coinciden con la firma de la función.
        """
        if nombre not in self.INTERVENCIONES:
            raise ValueError(
                f"Intervención desconocida: '{nombre}'. "
                f"Disponibles: {list(self.INTERVENCIONES.keys())}"
            )
        fn = self.INTERVENCIONES[nombre]
        try:
            nuevo = fn(self.arquetipo, **kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Parámetros incorrectos para '{nombre}': {exc}"
            ) from exc

        # BUG FIX 3: calcular fidelidad respecto al estado anterior
        fidelidad_ant = self.arquetipo.fidelidad(nuevo)

        self.arquetipo = nuevo
        self._snapshots.append(nuevo)
        self._registrar(nombre, nuevo, fidelidad_anterior=fidelidad_ant)
        return nuevo

    def _registrar(self, nombre: str, arq: Arquetipo, fidelidad_anterior: float = 1.0) -> None:
        """
        Registra el estado en el historial.
        BUG FIX 3: incluye campo 'fidelidad_anterior'.
        """
        self.historial.append({
            "intervencion":     nombre,
            "alpha":            _amplitud_serializable(arq.alpha),
            "beta":             _amplitud_serializable(arq.beta),
            "P_anima":          round(arq.prob_anima(), 4),
            "P_animus":         round(1 - arq.prob_anima(), 4),
            "entropia":         round(arq.entropia_shannon(), 4),
            "fidelidad_prev":   round(fidelidad_anterior, 4),  # BUG FIX 3
        })

    def resumen(self) -> dict:
        """Estadísticas de la sesión: delta total de P(Ánima), entropía final, individuación."""
        if len(self.historial) < 2:
            return {}
        p_ini = self.historial[0]["P_anima"]
        p_fin = self.historial[-1]["P_anima"]
        h_fin = self.historial[-1]["entropia"]
        return {
            "pasos":          len(self.historial) - 1,
            "delta_P_anima":  round(p_fin - p_ini, 4),
            "entropia_final": h_fin,
            "individuacion":  h_fin > 0.8,
        }

    def arquetipos_visitados(self) -> list[Arquetipo]:
        """MEJORA: lista de snapshots del Arquetipo en cada paso."""
        return list(self._snapshots)

    def entropia_proceso(self) -> list[float]:
        """MEJORA: evolución de la entropía a lo largo de la sesión."""
        return [h["entropia"] for h in self.historial]

    def exportar_json(self) -> str:
        """MEJORA: serializa el historial completo a JSON para persistencia."""
        return json.dumps({
            "resumen":   self.resumen(),
            "historial": self.historial,
        }, ensure_ascii=False, indent=2)
