# Laboratorio Cuántico-Junguiano

**Simulaciones computacionales** que entrelazan física cuántica, psicología analítica de Carl Jung y aprendizaje automático.

El proyecto modela los arquetipos psíquicos de Jung como qubits, las intervenciones terapéuticas como puertas cuánticas y la sincronicidad como entrelazamiento cuántico. Incluye un pipeline de ML para predecir relaciones entre parámetros psíquicos, un dashboard interactivo en Streamlit y módulos avanzados de diagnóstico bayesiano, tomografía de estado cuántico e informes clínicos generados por IA.

---

## Analogía central

| Concepto cuántico | Concepto junguiano | Representación en el código |
|---|---|---|
| Qubit | Par de opuestos arquetípicos (Ánima/Ánimus) | Dos niveles que representan la tensión consciente/inconsciente |
| Superposición | Ambigüedad psíquica | El arquetipo no está definido hasta que se "observa" (colapso) |
| Entrelazamiento | Sincronicidad | Correlación entre un contenido interno y un evento externo |
| Medición | Toma de conciencia (insight) | Colapso de la función de onda hacia un polo |
| Decoherencia | Represión, censura | Pérdida de la sincronicidad por canal de desfase |
| Puerta Hadamard | Apertura consciente | Lleva el estado a superposición máxima |
| Puerta Ry(θ) | Integración terapéutica | Rotación gradual entre polos |
| Pauli-Z | Represión de fase | Degrada el entrelazamiento sin invertir el polo |
| Pauli-X | Proyección psíquica | Inversión total del polo dominante |

---

## Experimentos implementados

**Experimento A — Superposición de un arquetipo**
Un qubit `α|0⟩ + β|1⟩` se mide repetidamente variando `α`. La frecuencia de colapso al polo Ánima sigue `P(0) = |α|²` (relación cuadrática). Se compara un modelo lineal y un polinomial (con GridSearchCV para selección automática de grado).

**Experimento B — Sincronicidad por entrelazamiento**
Dos qubits en estado de Bell `|Φ⁺⟩`. Se introduce una probabilidad `γ` de error de fase (represión) y se mide la correlación en base X. La relación teórica es `correlación = 1 − γ` (perfectamente lineal).

**Experimento C — Dataset de fidelidad arquetípica**
Matriz de fidelidades `F = |⟨ψᵢ|ψⱼ⟩|²` entre todos los pares de arquetipos con distintos `α`. Fundamento para el diagnóstico por proximidad arquetípica. Entrenado con un MLPRegressor.

**Experimento D — Sesión terapéutica (pipeline de puertas)**
Secuencia interactiva de intervenciones cuánticas sobre un arquetipo inicial. Cada puerta transforma el estado; se registra la evolución de P(Ánima), P(Ánimus) y la entropía de Shannon en tiempo real.

---

## Módulos principales

| Paquete/Archivo | Función |
|---|---|
| `core/experiments.py` | `Arquetipo` (qubit), `ParConDecoherencia` (entrelazamiento + canal de desfase) |
| `core/archetypes.py` | 5 arquetipos junguianos + `RegistroCuantico` (psique completa) |
| `core/interventions.py` | Puertas cuánticas + `SesionTerapeutica` (pipeline de intervenciones) |
| `core/lindblad.py` | Canal de Lindblad con relajación T1 (olvido activo) y desfase T2 (interferencia) |
| `ml/collect_data.py` | Generación de datasets A, B y C |
| `ml/train_regression.py` | Entrenamiento de modelos: lineal, polinomial (GridSearchCV), MLP |
| `ml/analysis.py` | Visualizaciones: scatter, heatmap de fidelidad, firma entrópica, radar chart |
| `analytics/diagnostico.py` | Diagnóstico arquetipal bayesiano — infiere `α` desde observaciones conductuales |
| `analytics/events.py` | Event Sourcing — `DiarioIndividuacion` persiste y analiza sesiones en JSONL |
| `analytics/qst.py` | Tomografía de Estado Cuántico — reconstruye el vector de Bloch desde 3 bases |
| `analytics/informe_analitico.py` | Informes clínicos narrativos generados por la API de Claude |
| `config.py` | Rutas centralizadas para datasets y modelos |
| `streamlit_app.py` | Dashboard interactivo (10 secciones) |
| `main.py` | CLI con menú de 11 opciones |

---

## Estructura del proyecto

```
laboratorio-cuantico-junguiano/
├── core/                        ← Núcleo cuántico
│   ├── __init__.py
│   ├── experiments.py           ← Arquetipo, ParConDecoherencia
│   ├── archetypes.py            ← 5 arquetipos + RegistroCuantico
│   ├── interventions.py         ← Puertas + SesionTerapeutica
│   └── lindblad.py              ← Canal de Lindblad T1/T2
├── ml/                          ← Pipeline de Machine Learning
│   ├── __init__.py
│   ├── collect_data.py          ← Generación de datasets A, B, C
│   ├── train_regression.py      ← Entrenamiento lineal, poli, MLP
│   └── analysis.py              ← Visualizaciones y radar chart
├── analytics/                   ← Análisis avanzado
│   ├── __init__.py
│   ├── diagnostico.py           ← Diagnóstico bayesiano
│   ├── events.py                ← Event Sourcing JSONL
│   ├── qst.py                   ← Tomografía de estado cuántico
│   └── informe_analitico.py     ← Informes clínicos con Claude API
├── tests/                       ← Suite de pruebas
│   └── __init__.py
├── datasets/                    ← CSVs generados (A, B, C)
├── docs/
│   ├── guia-de-usuario.md       ← Guía completa
│   ├── INFORME_ANALISIS.md      ← Análisis técnico de bugs y mejoras
│   └── manifiesto_cuantico_junguiano.md
├── exploracion/                 ← Scripts de exploración independientes
│   ├── superposición_arquetípica.py
│   └── sincronicidad_mediante_entrelazamiento.py
├── modelos/                     ← Modelos .pkl entrenados
├── config.py                    ← Rutas centralizadas
├── main.py                      ← CLI con 11 opciones
├── streamlit_app.py             ← Dashboard completo (10 secciones)
├── readme.md
└── requirements.txt
```

---

## Instalación

```bash
git clone https://github.com/martinezmarcos93/laboratorio-cuantico-junguiano.git
cd laboratorio-cuantico-junguiano
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

**Dependencias opcionales:**
- `networkx` — grafo de sincronicidad en Streamlit (incluido en requirements.txt)
- `anthropic` — informes clínicos vía Claude API (incluido en requirements.txt)

Para los informes clínicos configurá tu API key:
```powershell
$env:ANTHROPIC_API_KEY = "tu-clave-aqui"
```

---

## Uso

**Modo interactivo (CLI):**
```bash
python main.py
```

**Modo automático (flujo completo):**
```bash
python main.py --auto
```

**Dashboard Streamlit (8 secciones):**
```bash
streamlit run streamlit_app.py
```

**Módulos directamente desde Python:**
```python
import numpy as np
from experiments import Arquetipo, ParConDecoherencia
from archetypes  import RegistroCuantico
from interventions import SesionTerapeutica

# Arquetipo con dominancia de Ánima
arq = Arquetipo(alpha=0.9, beta=np.sqrt(1 - 0.81))
print(arq.entropia_shannon())   # ambigüedad psíquica en bits

# Pipeline terapéutico
sesion = SesionTerapeutica(arq)
sesion.aplicar("apertura_consciente")
sesion.aplicar("integracion_parcial", theta=np.pi / 4)
print(sesion.resumen())

# Diagnóstico bayesiano
from diagnostico import inferir_alpha
obs = [arq.medir() for _ in range(200)]
print(inferir_alpha(obs))

# Tomografía de estado cuántico
from qst import tomografia_bloch, medir_base_x, medir_base_y
obs_x = medir_base_x(arq, 200)
obs_y = medir_base_y(arq, 200)
print(tomografia_bloch(obs, obs_x, obs_y))
```

---

## Secciones del dashboard Streamlit

| # | Sección | Descripción |
|---|---------|-------------|
| 1 | 🔮 Superposición del Arquetipo | Slider de α, predicción en tiempo real, tabla teórica |
| 2 | 🌀 Sincronicidad bajo represión | Slider de γ, estimación de correlación vs teoría |
| 3 | 🧠 Registro cuántico | Psique completa: tabla de componentes, medición simultánea |
| 4 | 💊 Sesión terapéutica | Pipeline interactivo de puertas + gráfica de evolución + entropía |
| 5 | 📊 Comparación de modelos ML | Entrena y visualiza modelos lineal vs polinomial |
| 6 | 🕸️ Grafo de Sincronicidad | Heatmap 5×5 + grafo networkx de resonancias arquetípicas |
| 7 | 🔬 Diagnóstico Bayesiano | Infiere α de cada componente con intervalos de credibilidad |
| 8 | ⚗️ Canal de Lindblad | Comparación de regímenes T1/T2 + mapa de represión 2D |
| 9 | 🔭 Tomografía Cuántica (QST) | Reconstruye el vector de Bloch, visualiza esfera de Bloch 2D |
| 10 | 📋 Informe Clínico (IA) | Informe narrativo generado por Claude API |

---

## Resultados esperados

**Dataset A:** el modelo polinomial (grado 2) obtiene R² ≈ 1.0. El modelo lineal falla sistemáticamente porque la relación real es cuadrática — ilustra el costo epistémico de asumir el modelo incorrecto.

**Dataset B:** la correlación decrece linealmente con γ. El modelo lineal obtiene R² ≈ 1.0.

**Diagnóstico Bayesiano:** con 200 observaciones el IC 95% de α típicamente tiene ancho < 0.15; con 500 observaciones < 0.08.

**QST:** la pureza del estado reconstruido es ≈ 0.9–1.0 para arquetipos con amplitudes reales (el qubit tiene ry ≈ 0).

---

## Dependencias

```
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
streamlit>=1.32.0
scipy>=1.11.0
networkx>=3.0
anthropic>=0.40.0
```

---

## Referencias

- Jung, C. G. *Sincronicidad: un principio de conexión acausal.*
- Nielsen, M. A. & Chuang, I. L. *Quantum Computation and Quantum Information.*
- Lindblad, G. *On the generators of quantum dynamical semigroups.* (1976)
- Documentación de scikit-learn, NumPy, SciPy y Anthropic SDK.

---

## Licencia

MIT.
