# Guía de Usuario — Laboratorio Cuántico-Junguiano v3.0

> *"En el código, el observador no solo colapsa la función de onda; también interpreta el resultado."*

---

## Tabla de contenidos

1. [¿Qué es este proyecto?](#qué-es)
2. [Instalación](#instalación)
3. [Modos de uso](#modos-de-uso)
4. [Los experimentos centrales](#los-experimentos)
5. [Los arquetipos como qubits](#los-arquetipos)
6. [Las intervenciones terapéuticas](#las-intervenciones)
7. [Dashboard Streamlit — guía por sección](#streamlit)
8. [Módulos avanzados](#módulos-avanzados)
   - [diagnostico.py — Inferencia bayesiana](#diagnostico)
   - [events.py — Diario de individuación](#events)
   - [lindblad.py — Canal T1/T2](#lindblad)
   - [qst.py — Tomografía cuántica](#qst)
   - [informe_analitico.py — Claude API](#informe)
9. [CLI — menú de 11 opciones](#cli)
10. [Glosario rápido](#glosario)

---

## ¿Qué es este proyecto? {#qué-es}

El Laboratorio Cuántico-Junguiano es un entorno de simulación computacional que usa mecánica cuántica simplificada para modelar conceptos de la psicología analítica de Carl Jung. No es un simulador clínico ni un simulador cuántico de hardware real — es una herramienta pedagógica, exploratoria y creativa que hace visible, de forma matemática y computacional, cómo funcionan ciertas ideas psicológicas cuando se traducen al lenguaje de la física cuántica.

**Lo que podés hacer:**
- Simular cómo un arquetipo psíquico "colapsa" hacia un polo cuando se lo observa (insight)
- Medir cómo la represión destruye la sincronicidad entre eventos psíquicos
- Entrenar modelos de ML que predicen esas relaciones y visualizar por qué el modelo incorrecto siempre falla
- Explorar intervenciones terapéuticas como puertas cuánticas y ver cómo transforman el estado
- Analizar la psique completa como un registro de 5 qubits con su grafo de resonancias
- Inferir el estado arquetípico de un sujeto desde observaciones conductuales (diagnóstico bayesiano)
- Reconstruir el estado psíquico desde mediciones en múltiples bases (tomografía cuántica)
- Generar informes clínicos narrativos en lenguaje natural usando la API de Claude

---

## Instalación {#instalación}

```bash
git clone https://github.com/martinezmarcos93/laboratorio-cuantico-junguiano.git
cd laboratorio-cuantico-junguiano

python -m venv venv
venv\Scripts\activate          # Windows PowerShell
# source venv/bin/activate     # Linux / Mac

pip install -r requirements.txt
```

**Para los informes clínicos con IA** (sección 8 del dashboard), configurá tu API key de Anthropic en la misma terminal donde lanzás Streamlit:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
streamlit run streamlit_app.py
```

---

## Modos de uso {#modos-de-uso}

### Modo interactivo — menú en consola

```bash
python main.py
```

Presenta un menú con 11 opciones. Podés correr cada paso por separado: generar datos, entrenar modelos, visualizar, diagnosticar, hacer tomografía o comparar canales de represión.

### Modo automático — flujo completo sin pausas

```bash
python main.py --auto
```

Genera todos los datasets → entrena todos los modelos → muestra todas las visualizaciones. Útil para reproducir resultados rápidamente.

### Dashboard interactivo

```bash
streamlit run streamlit_app.py
```

Abre una aplicación web en el navegador con 8 secciones, sliders en tiempo real e integración con la API de Claude.

### Uso desde Python directamente

```python
import numpy as np
from experiments  import Arquetipo, ParConDecoherencia
from archetypes   import RegistroCuantico
from interventions import SesionTerapeutica

arq    = Arquetipo(alpha=0.85, beta=np.sqrt(1 - 0.7225))
sesion = SesionTerapeutica(arq)
sesion.aplicar("apertura_consciente")
sesion.aplicar("integracion_parcial", theta=np.pi / 6)
print(sesion.resumen())
print(sesion.exportar_json())
```

---

## Los experimentos centrales {#los-experimentos}

### Experimento A — Superposición del Arquetipo

**¿Qué simula?**
Un qubit `α|0⟩ + β|1⟩` que se mide muchas veces variando `α`. Se registra con qué frecuencia colapsa al polo |0⟩ (Ánima).

**Física:** la probabilidad de colapso sigue `P(0) = |α|²` — relación cuadrática, no lineal.

**Jung:** el qubit representa un par de opuestos arquetípicos. La medición es el insight: el arquetipo colapsa hacia un polo y se vuelve consciente.

**¿Qué demuestra la comparación de modelos?**
El modelo lineal se desvía sistemáticamente. El polinomial (grado 2, seleccionado automáticamente por GridSearchCV) coincide casi perfectamente con la curva teórica. Lección central: el error no viene siempre de datos ruidosos sino de suponer la forma equivocada de la relación.

**Dataset:** `datasets/arquetipo_prob.csv` — columnas `alpha`, `prob_anima`.

---

### Experimento B — Sincronicidad por entrelazamiento

**¿Qué simula?**
Dos qubits en estado de Bell `|Φ⁺⟩ = (|00⟩ + |11⟩)/√2`. Se aplica un canal de desfase con intensidad `γ` al primer qubit y se mide la correlación en base X.

**Física:** la correlación teórica en base X es `1 − γ`. Con `γ = 0` la correlación es perfecta; con `γ = 1` se aplica Pauli-Z con certeza y la correlación cae a 0 (anticorrelación perfecta en esta métrica).

**Jung:** el entrelazamiento modela la sincronicidad — correlación acausal entre evento interno y externo. El parámetro `γ` modela la represión: cuanto mayor, más se degrada la conexión entre mundo interno y externo.

**Dataset:** `datasets/sincronicidad_corr.csv` — columnas `gamma`, `correlacion`.

---

### Experimento C — Fidelidad arquetípica

**¿Qué simula?**
Para cada par `(αᵢ, αⱼ)` calcula la fidelidad cuántica `F = |⟨ψᵢ|ψⱼ⟩|²`. Esta es la "resonancia arquetípica" entre dos estados psíquicos.

**Uso:** la fidelidad es la base del diagnóstico por proximidad — cuantifica qué tan "cerca" está el estado de un paciente de un arquetipo de referencia (ej. Si-mismo).

**Dataset:** `datasets/fidelidad_arquetipica.csv` — columnas `alpha_i`, `alpha_j`, `fidelidad`.

---

## Los arquetipos como qubits {#los-arquetipos}

El módulo `archetypes.py` define los 5 componentes clásicos de la psique junguiana como qubits con amplitudes iniciales características:

| Arquetipo | P(Ánima) | Entropía | Significado |
|-----------|----------|----------|-------------|
| **Yo (Ego)** | 0.60 | ~0.97 bits | Núcleo de la consciencia, levemente sesgado pero abierto |
| **Persona** | 0.85 | ~0.61 bits | Máscara social, baja ambigüedad, polo "presentable" |
| **Sombra** | 0.15 | ~0.61 bits | Contenidos reprimidos, polo opuesto al Yo |
| **Ánima/Ánimus** | 0.50 | 1.00 bit | Contrasexual interno, máxima ambigüedad |
| **Sí-mismo** | 0.50 | 1.00 bit | Totalidad psíquica, integración de opuestos |

**La entropía de Shannon** mide la ambigüedad del estado. Entropía 0 = polarización rígida (el arquetipo ya "cayó" a un polo, sin tensión). Entropía 1 bit = máxima tensión creativa, máximo potencial para el proceso de individuación.

**El índice de individuación** combina la tensión Yo↔Sombra y la entropía media: valor cercano a 1 indica integración avanzada.

```python
from archetypes import RegistroCuantico

reg = RegistroCuantico(seed=42)
print(reg.coherencia_global())      # entropía por componente
print(reg.tension_yo_sombra())      # |P(Ánima|Yo) - P(Ánimus|Sombra)|
print(reg.indice_individuacion())   # [0, 1]
print(reg.narrativa_estado())       # texto interpretativo
print(reg.grafo_sincronicidad())    # matriz 5×5 de fidelidades mutuas
```

---

## Las intervenciones terapéuticas {#las-intervenciones}

El módulo `interventions.py` implementa puertas cuánticas como transformaciones sobre el estado del arquetipo. Cada una corresponde a una intervención de la psicología analítica:

| Función | Puerta | Efecto | Jung |
|---------|--------|--------|------|
| `apertura_consciente` | Hadamard | Lleva cualquier estado a P(Ánima) = 0.5 | Apertura total al inconsciente |
| `integracion_parcial(theta)` | Ry(θ) | Desplaza gradualmente el equilibrio | Integración progresiva de la Sombra |
| `amplificacion(polo, theta)` | Ry dirigida | Refuerza un polo específico | Técnica de amplificación junguiana |
| `proyeccion` | Pauli-X | Inversión total de polos | Proyección psíquica: Persona ↔ Sombra |

**Usarlas con `SesionTerapeutica`:**

```python
from experiments   import Arquetipo
from interventions import SesionTerapeutica
import numpy as np

# Paciente con Yo muy dominante (α = 0.95)
arq    = Arquetipo(0.95, np.sqrt(1 - 0.9025))
sesion = SesionTerapeutica(arq)

sesion.aplicar("apertura_consciente")
sesion.aplicar("integracion_parcial", theta=np.pi / 4)
sesion.aplicar("amplificacion", polo=1, theta=np.pi / 6)

print(sesion.resumen())
print(sesion.entropia_proceso())     # evolución de la entropía
print(sesion.exportar_json())        # historial completo en JSON
```

**Ángulos de referencia para `integracion_parcial`:**
- `θ = π/6` (30°) — intervención suave
- `θ = π/4` (45°) — integración moderada
- `θ = π/2` (90°) — rotación fuerte hacia el equilibrio
- `θ = π`   (180°) — inversión completa (equivale a proyección)

---

## Dashboard Streamlit — guía por sección {#streamlit}

Lanzá el dashboard con `streamlit run streamlit_app.py`. El panel de navegación está en la barra lateral izquierda.

### 🔮 Sección 1 — Superposición del Arquetipo
Mové el slider de `α` para ver cómo cambia `P(Ánima)` en tiempo real. Presioná **⚡ Simular medición** para ejecutar `n_shots` mediciones y comparar el resultado observado con el teórico. La tabla inferior muestra los valores exactos para todo el rango de `α`.

### 🌀 Sección 2 — Sincronicidad bajo represión
El slider de `γ` controla la intensidad del canal de desfase. El color del indicador inferior cambia según el nivel de represión (verde = sincronicidad alta, rojo = decoherencia severa). Presioná **⚡ Estimar correlación** para medir empíricamente.

### 🧠 Sección 3 — Registro cuántico
Configurá la semilla y visualizás el estado de los 5 arquetipos simultáneamente: P(Ánima), P(Ánimus) y entropía de Shannon de cada uno. El botón **🎲 Medir todos los componentes** simula un "insight total" y registra los resultados en la tabla inferior.

### 💊 Sección 4 — Sesión terapéutica
Configurá el `α` inicial del arquetipo (ej. 0.95 = Yo muy dominante). Seleccioná una intervención del desplegable, ajustá el ángulo `θ` si corresponde y presioná **➕ Aplicar intervención**. La tabla muestra el estado en cada paso y el gráfico tiene dos subgráficas:
- **Arriba:** evolución de P(Ánima) y P(Ánimus) paso a paso
- **Abajo:** evolución de la entropía de Shannon — el **índice de individuación en tiempo real**

### 📊 Sección 5 — Comparación de modelos ML
Si no hay modelos entrenados, presioná **🚀 Generar datos y entrenar modelos ahora**. Las dos pestañas muestran la comparación visual del modelo lineal vs polinomial sobre cada dataset.

### 🕸️ Sección 6 — Grafo de Sincronicidad
Visualiza la **red arquetípica** de la psique:
- **Heatmap izquierdo:** matriz 5×5 de fidelidades mutuas entre componentes. Colores cálidos (amarillo/verde) = alta resonancia; colores fríos (azul/morado) = alta tensión.
- **Grafo derecho:** red donde el grosor de las aristas representa la fidelidad. El slider de umbral filtra aristas débiles. Requiere `networkx` instalado.

**Interpretación típica:** Persona y Yo suelen tener alta fidelidad (la máscara social refleja el Ego). Yo y Sombra deberían tener fidelidad baja (tensión no resuelta). Si la fidelidad Yo↔Sombra es alta, hay posible identificación con la Sombra.

### 🔬 Sección 7 — Diagnóstico Bayesiano
Configurá la semilla y el número de observaciones. Presioná **🔬 Ejecutar diagnóstico** para:
1. Generar N mediciones de cada qubit del `RegistroCuantico`
2. Calcular la distribución posterior Beta analítica de `p = |α|²`
3. Mostrar la tabla comparativa entre `α real` y `α estimado (MAP)` con errores
4. Seleccioná un componente para ver su distribución posterior completa (media, MAP, IC 95%)

**Cuándo usar esto:** cuando tenés una serie de "observaciones conductuales" de un sujeto (sueños registrados, respuestas a pruebas proyectivas) y querés cuantificar la probabilidad de que pertenezca a cada polo arquetípico.

### 📋 Sección 8 — Informe Clínico (IA)
Requiere `ANTHROPIC_API_KEY` configurada. Configurá la semilla del registro y el `α` inicial de la sesión. Presioná **📋 Generar informe clínico** para obtener un análisis narrativo estructurado en tres secciones:

1. **Estado global** — síntesis del nivel de individuación
2. **Análisis por componente** — interpretación clínica de cada arquetipo
3. **Recomendaciones terapéuticas** — sugerencias concretas con las puertas disponibles

El botón **⬇️ Descargar informe** guarda el texto como `.txt`.

---

## Módulos avanzados {#módulos-avanzados}

### diagnostico.py — Inferencia bayesiana del α arquetípico {#diagnostico}

Dado un conjunto de observaciones binarias (0 = Ánima, 1 = Ánimus), infiere la distribución completa sobre `α`. Usa el conjugado bayesiano exacto: prior Beta → posterior Beta, sin necesidad de MCMC.

```python
from diagnostico import inferir_alpha, diagnosticar_registro, convergencia_bayesiana
from archetypes  import RegistroCuantico

# Inferencia desde observaciones brutas
obs      = [arq.medir() for _ in range(200)]
resultado = inferir_alpha(obs)
# → {'p_media': 0.73, 'alpha_MAP': 0.694, 'IC_95_alpha': (0.59, 0.79), ...}

# Diagnosticar toda la psique
reg = RegistroCuantico(seed=42)
diagnostico_completo = diagnosticar_registro(reg, n_obs=200)
for nombre, r in diagnostico_completo.items():
    print(f"{nombre}: α_MAP = {r['alpha_MAP']}, IC = {r['IC_95_alpha']}")

# Ver la convergencia visual
convergencia_bayesiana(verdadero_alpha=0.70, n_max=300)
```

**Interpretación del IC 95%:** si el intervalo de credibilidad al 95% de α es [0.60, 0.78], hay un 95% de probabilidad posterior de que el verdadero estado del arquetipo tenga `α` en ese rango. Con más observaciones el intervalo se estrecha.

---

### events.py — Diario de Individuación (Event Sourcing) {#events}

Persiste sesiones terapéuticas en un archivo JSONL para análisis longitudinal. Cada línea del archivo es una sesión completa con timestamp, ID de paciente y notas clínicas.

```python
from events        import DiarioIndividuacion
from interventions import SesionTerapeutica
from experiments   import Arquetipo
import numpy as np

diario = DiarioIndividuacion("mi_diario.jsonl")

# Registrar una sesión
arq    = Arquetipo(0.9, np.sqrt(1 - 0.81))
sesion = SesionTerapeutica(arq)
sesion.aplicar("apertura_consciente")
sesion.aplicar("integracion_parcial", theta=np.pi / 4)
diario.registrar(sesion, paciente="Ana", notas="Primera sesión. Alta resistencia inicial.")

# Analizar la trayectoria longitudinal
print(diario.estadisticas_longitudinales("Ana"))
# → {'sesiones': 5, 'entropia_media': 0.72, 'tendencia': 'individuación activa', ...}

# Ver evolución gráfica
diario.graficar_trayectoria("Ana")

# Listar todos los pacientes
print(diario.pacientes())

# Replay completo
for sesion_guardada in diario.trayectoria("Ana"):
    print(sesion_guardada["timestamp"], sesion_guardada["resumen"])
```

**Formato del archivo JSONL:** cada línea es un objeto JSON con las claves `timestamp`, `paciente`, `notas`, `resumen` e `historial` (lista de pasos con alpha, beta, P_anima, entropía y fidelidad respecto al paso anterior).

---

### lindblad.py — Canal de Lindblad generalizado {#lindblad}

Extiende `ParConDecoherencia` con dos mecanismos de represión diferenciados:

- **γ₁ — relajación (T1):** el contenido "cae" al inconsciente sin dejar huella consciente. Operador σ₋: |1⟩ → |0⟩.
- **γ₂ — desfase puro (T2):** el contenido existe en el inconsciente pero la coherencia de fase se pierde, imposibilitando el insight sin destruir el contenido. Operador σz.

```python
from lindblad import ParConLindblad, comparar_canales, escanear_espacio_lindblad

# Par con canal T1 + T2
par = ParConLindblad(seed=42)
par.aplicar_represion_lindblad(gamma1=0.3, gamma2=0.4)
print(par.metricas())
# → {'entropia_entrelazamiento': ..., 'correlacion_teorica': ...}

# Comparar los tres regímenes: desfase Z, relajación T1, canal mixto
comparar_canales()

# Mapa completo del espacio de represión (puede tardar ~1 min)
g1, g2, mat = escanear_espacio_lindblad(n_puntos=11, n_trials=200)
graficar_espacio_lindblad(g1, g2, mat)
```

**Diferencia clave con el canal original:** `ParConDecoherencia.aplicar_represion(gamma)` aplica solo desfase Z (equivale a γ₁=0, γ₂=gamma en el Lindblad). El canal Lindblad completo permite separar los dos mecanismos y explorar su espacio 2D.

---

### qst.py — Tomografía de Estado Cuántico {#qst}

Reconstruye el estado psíquico completo desde mediciones en tres bases ortogonales. Es la contraparte computacional de la observación clínica multidimensional.

**Protocolo de 3 bases:**
- **Base Z** `{|0⟩, |1⟩}` — mide rz = polarización Ánima/Ánimus
- **Base X** `{|+⟩, |−⟩}` — mide rx = coherencia entre polos
- **Base Y** `{|+i⟩, |−i⟩}` — mide ry = fase compleja (siempre ≈ 0 para amplitudes reales)

```python
from qst import (tomografia_z, tomografia_bloch, reconstruir_arquetipo,
                 medir_base_x, medir_base_y, graficar_esfera_bloch_2d,
                 tomografia_registro)
from archetypes import RegistroCuantico
import numpy as np

# Tomografía de un solo arquetipo
arq   = Arquetipo(0.70, np.sqrt(1 - 0.49), seed=42)
obs_z = [arq.medir()          for _ in range(300)]
obs_x = medir_base_x(arq, 300)
obs_y = medir_base_y(arq, 300)

res_bloch = tomografia_bloch(obs_z, obs_x, obs_y)
print(f"rz={res_bloch['rz']:.4f}  rx={res_bloch['rx']:.4f}  pureza={res_bloch['pureza']:.4f}")

arq_rec = reconstruir_arquetipo(res_bloch)
print(f"α reconstruido: {arq_rec.alpha:.4f}  (real: 0.7000)")

# Tomografía completa del RegistroCuantico
reg     = RegistroCuantico(seed=42)
qst_reg = tomografia_registro(reg, n_obs=300)
for nombre, r in qst_reg.items():
    print(f"{nombre}: rz={r['rz']:+.4f}, pureza={r['pureza']:.4f}")

# Visualizar en la esfera de Bloch 2D
rz_real = 2 * arq.prob_anima() - 1
rx_real = 2 * arq.alpha * arq.beta
graficar_esfera_bloch_2d(
    [{"rx": res_bloch["rx"], "rz": res_bloch["rz"]},
     {"rx": rx_real, "rz": rz_real}],
    etiquetas=["QST reconstruido", "Estado real"]
)
```

**Lectura de la esfera de Bloch 2D:**
- Polo superior (rz=+1): arquetipo completamente en Ánima
- Polo inferior (rz=-1): completamente en Ánimus
- Ecuador (rz=0, rx=±1): superposición perfecta
- Centro (r=0): estado completamente mixto (máxima decoherencia)

---

### informe_analitico.py — Informes clínicos con Claude API {#informe}

Genera un informe narrativo estructurado combinando el estado cuantitativo del `RegistroCuantico` con el lenguaje de la psicología analítica. Usa la API de Claude con prompt caching para reducir costo en uso intensivo.

```python
from archetypes        import RegistroCuantico
from interventions     import SesionTerapeutica
from experiments       import Arquetipo
from informe_analitico import generar_informe_con_cache
import numpy as np

reg    = RegistroCuantico(seed=42)
arq    = Arquetipo(0.95, np.sqrt(1 - 0.9025))
sesion = SesionTerapeutica(arq)
sesion.aplicar("apertura_consciente")
sesion.aplicar("integracion_parcial", theta=np.pi / 4)

informe = generar_informe_con_cache(reg, sesion)
print(informe)
```

**Estructura del informe generado:**
1. **Estado global** — síntesis del nivel de individuación con los índices numéricos
2. **Análisis por componente** — interpretación de cada uno de los 5 arquetipos
3. **Recomendaciones terapéuticas** — intervenciones concretas usando las puertas implementadas

**Requisitos:**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python informe_analitico.py
```

---

## CLI — menú de 11 opciones {#cli}

```
python main.py
```

| Opción | Función |
|--------|---------|
| 1 | Generar datasets A y B (arquetipo y sincronicidad) |
| 2 | Entrenar modelos de regresión (lineal, polinomial, GridSearchCV) |
| 3 | Mostrar análisis — gráficas de ambos datasets |
| 4 | Ejecutar todo el flujo (1 → 2 → 3) |
| 5 | Generar dataset C (fidelidad arquetípica) |
| 6 | Comparar todos los modelos entrenados (tabla R² y MSE) |
| 7 | Narrativa psíquica de un RegistroCuantico aleatorio |
| 8 | Diagnóstico Arquetipal Bayesiano de un RegistroCuantico |
| 9 | Tomografía de Estado Cuántico (QST) — ingresás α, reconstruye el estado |
| 10 | Canal de Lindblad — comparación de regímenes T1, T2 y mixto |
| 11 | Salir |

**Flujo completo recomendado la primera vez:**

```bash
python main.py
# → opción 4 (ejecutar todo el flujo)
# → opción 8 (diagnóstico bayesiano)
# → opción 9 (tomografía)
```

O directamente con `--auto` para solo el pipeline de ML:

```bash
python main.py --auto
streamlit run streamlit_app.py
```

---

## Glosario rápido {#glosario}

| Término | Física cuántica | Jung |
|---------|----------------|------|
| **Qubit** | Sistema de 2 niveles | Par de opuestos arquetípicos |
| **Superposición** | Estado no definido antes de medir | Ambigüedad psíquica activa |
| **Colapso** | La medición fuerza un valor definido | Insight: el arquetipo se vuelve consciente |
| **Entrelazamiento** | Correlación no local entre qubits | Sincronicidad: conexión acausal |
| **Decoherencia** | Pérdida de correlaciones cuánticas | Represión, censura psíquica |
| **α (alpha)** | Amplitud del estado `\|0⟩` | Fuerza del polo Ánima (consciente) |
| **β (beta)** | Amplitud del estado `\|1⟩` | Fuerza del polo Ánimus (inconsciente) |
| **γ (gamma)** | Probabilidad de error de fase | Intensidad de la represión |
| **Hadamard** | Crea superposición perfecta | Apertura total al inconsciente |
| **Ry(θ)** | Rotación en el plano de Bloch | Intervención terapéutica gradual |
| **Pauli-Z** | Inversión de fase | Represión de contenido psíquico |
| **Pauli-X** | Inversión de polo | Proyección psíquica total |
| **Entropía Shannon** | Incertidumbre de la medición en la base Z | Grado de ambigüedad psíquica activa |
| **Fidelidad** | `F = \|⟨ψ\|φ⟩\|²` | Resonancia arquetípica entre dos estados |
| **Esfera de Bloch** | Representación geométrica del qubit | Mapa del espacio de estados psíquicos |
| **T1 (relajación)** | Transición `\|1⟩ → \|0⟩` por disipación | Olvido activo — el contenido se hunde en el inconsciente |
| **T2 (desfase puro)** | Pérdida de coherencia de fase sin cambio de polo | Represión que bloquea el insight sin borrar el contenido |
| **R²** | Coeficiente de determinación | Qué tan bien captura el modelo la relación real |
| **QST** | Tomografía de Estado Cuántico | Reconstrucción del estado desde observaciones múltiples |
| **Prior Beta(1,1)** | Distribución uniforme sobre [0,1] | Ausencia de supuestos iniciales sobre el arquetipo |
