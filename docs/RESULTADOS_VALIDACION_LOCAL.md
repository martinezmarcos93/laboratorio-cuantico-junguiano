# Resultados de la validación local

Este documento registra una auditoría local completa del repositorio. Distingue en todo momento dos conclusiones diferentes:

- **el software funciona**: el código calcula lo que dice calcular;
- **la hipótesis científica recibe soporte**: los datos favorecen una afirmación sobre el mundo.

Aquí sólo se establece la primera. Ningún experimento de este repositorio usa datos empíricos, de modo que **ninguna hipótesis psicológica o junguiana recibe soporte** de estos resultados.

## Contexto de la ejecución

| | |
|---|---|
| Fecha | 2026-10-06 |
| Rama | `research/programa-cuantico-2026` |
| Commit auditado | `b176d2f` (fix: use exact Lindblad propagation and validate parameters) |
| Sistema | Linux 7.0 (Ubuntu), 4 núcleos |
| Python | 3.12.3, entorno virtual dedicado fuera del repositorio |
| Dependencias | numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, pandas 3.0.6, scikit-learn 1.9.1, joblib 1.6.0, streamlit 1.65.0, networkx 3.7, anthropic 1.11.0, pytest 9.1.1 |

Se instalaron únicamente los paquetes de `requirements.txt` más `pytest`. No se actualizó ningún paquete preexistente.

### Comandos

```bash
python -m pytest -q                      # suite completa
python -m py_compile <todos los .py>     # 23 archivos
python main.py                           # pasos no interactivos del menú, uno por uno
python -m analytics.qst | analytics.diagnostico | analytics.events | core.lindblad
streamlit run streamlit_app.py --server.headless true
```

Los pasos que escriben datasets y modelos se ejecutaron sobre una copia del repositorio, para no modificar los `.pkl` versionados.

## Tests

| Momento | Resultado |
|---|---|
| Suite original, sin tocar código | 11 de 11 |
| Suite final, tras las correcciones | **174 de 174** (11 originales + 163 nuevos en `tests/test_validacion_local.py`) |

Ningún test se desactivó ni se marcó `skip`/`xfail`. Los 11 tests originales pasan sin cambios.

La suite original pasaba completa y aun así el código tenía los errores de la sección "Problemas encontrados": ninguno estaba cubierto.

### Warnings

- `InconsistentVersionWarning` de scikit-learn al cargar los tres `.pkl` versionados: se serializaron con 1.8.0 y se cargan con 1.9.1. Cargan y predicen; conviene regenerarlos con `main.py` en el entorno de trabajo.
- Streamlit: `use_container_width` está obsoleto (reemplazo: `width`). No afecta al funcionamiento con 1.65.0.
- Ningún warning numérico (overflow, división por cero, NaN) en el recorrido de los módulos.

## Estado por componente

### VALIDADO

| Componente | Qué se comprobó |
|---|---|
| **E001 — fundamentos** | Normalización con amplitudes reales, complejas y extremas (1e-200 a 1e200). Frecuencia de medición con 2·10⁵ muestras: 0.64096 frente a 0.64 teórico (error 0.001, 3σ = 0.003). Ry conserva la norma y Ry(θ)Ry(−θ) = I. Bell: negatividad 0.5, concurrencia 1, entropía reducida 1 bit. |
| **E002 — desfase** | Para γ ∈ {0, 0.1, 0.25, 0.5, 0.75, 0.9, 1} el código coincide con la solución analítica con error < 1e-10: coherencia (1−γ)/2, ⟨X⊗X⟩ = 1−γ, P(x₁=x₂) = 1−γ/2, negatividad (1−γ)/2, concurrencia 1−γ, pureza 1−γ+γ²/2. Traza 1, hermiticidad y autovalores ≥ 0 en todos los puntos. Monotonía estricta en 41 puntos. Composición de canales: (1−γ₁)(1−γ₂). |
| **E002 — Monte Carlo** | γ=0.5: 0.7420 (n=10³), 0.7502 (10⁴), 0.7498 (10⁵) frente a 0.75. γ=1: 0.478, 0.500, 0.501 frente a 0.5. Siempre dentro de 3σ. |
| **E003 — correlación y entrelazamiento** | La entropía reducida vale 1 bit para todo γ, mientras negatividad y concurrencia caen a 0 en γ=1: confirma que la entropía reducida no mide entrelazamiento de estados mixtos. En γ=1 el estado es diag(½,0,0,½): correlación clásica perfecta en Z, sin entrelazamiento, coincidencia en X igual al azar. |
| **Lindblad** | 270 combinaciones de (γ₁, γ₂, dt, `pasos`), con dt de 0 a 1000: diferencia máxima con la solución analítica 1.1e-16; traza, hermiticidad y positividad dentro de 1e-15. Resultado idéntico bit a bit con `pasos` = 1 y 5000. Propiedad de semigrupo (0.3 + 0.7 = 1.0) con error 5.6e-17. Límite t→∞ con γ₁: \|0⟩⟨0\|⊗I/2. Parámetros inválidos rechazados. |
| **E004 — tomografía (tras la corrección)** | Ver tabla de convergencia. La matriz reconstruida es hermítica, de traza 1 y con autovalores ≥ −3e-16 en todas las semillas. |
| **Inferencia bayesiana** | Posterior Beta correcta (64 de 100 → media 65/102). Cobertura empírica del IC 95 % en 2000 repeticiones: 0.965 (n=20), 0.947 (n=200), 0.965 (p=0.05, n=50), 0.952 (n=1000). |
| **Event sourcing** | Escritura sólo por agregado, relectura desde disco, trayectorias por paciente, estadísticas longitudinales. |
| **Pipeline ML** | Generación de datasets, entrenamiento, serialización, carga e inferencia. R² de test: lineal 0.785, polinomial grado 2 0.993, sincronicidad 0.9998. |
| **Streamlit** | Las 10 secciones cargan sin excepción y los 10 botones se ejecutan (AppTest). El servidor headless arranca y responde 200 en `/_stcore/health` y en `/`. |
| **Reproducibilidad** | Misma semilla → misma secuencia en `Arquetipo`, `ParConDecoherencia`, `RegistroCuantico`, QST y, tras la corrección, en estados transformados. |

### Convergencia de la tomografía

40 semillas independientes por punto, semillas distintas para Z, X e Y.

| Estado | Shots | Fidelidad media | Error de Frobenius | Error·√shots |
|---|---:|---:|---:|---:|
| real (0.8, 0.6) | 100 | 0.98810 | 0.0909 | 0.91 |
| | 1 000 | 0.99760 | 0.0277 | 0.88 |
| | 10 000 | 0.99891 | 0.0089 | 0.89 |
| \|0⟩ | 100 | 0.99460 | 0.0954 | 0.95 |
| | 10 000 | 0.99996 | 0.0079 | 0.79 |
| \|+⟩ | 100 | 0.99553 | 0.0851 | 0.85 |
| | 10 000 | 0.99996 | 0.0077 | 0.77 |
| complejo (0.6, 0.8i) | 100 | 0.98916 | 0.0870 | 0.87 |
| | 10 000 | 0.99953 | 0.0074 | 0.74 |
| \|+i⟩ | 100 | 0.99471 | 0.0947 | 0.95 |
| | 10 000 | 0.99996 | 0.0076 | 0.76 |

El error cae como 1/√shots (el producto error·√shots se mantiene entre 0.74 y 0.95) y la fidelidad crece con los shots en todos los estados. La hipótesis de E004 se cumple.

Antes de la corrección, los dos estados complejos **no convergían**: fidelidad 0.546 → 0.538 para (0.6, 0.8i) y 0.508 → 0.500 para \|+i⟩ al pasar de 100 a 10 000 shots.

### VALIDADO PARCIALMENTE

| Componente | Alcance y límite |
|---|---|
| **Pipeline ML** | Funciona de extremo a extremo, pero los modelos recuperan la fórmula con la que se simularon los datos (P = α², P = 1 − γ/2). Es una comprobación del pipeline, no un resultado. |
| **Índices junguianos** (`indice_individuacion`, `tension_yo_sombra`, `narrativa_estado`, etiquetas "integrado/parcial/polarizado") | Están acotados, son reproducibles y se calculan como dice el código. Los pesos (0.6/0.4) y los umbrales (0.8, 0.75, 0.45) son elecciones de diseño sin calibración contra ningún dato. |
| **`reconstruir_arquetipo`** | Recupera el estado con fidelidad > 0.99 cuando la matriz reconstruida es casi pura. Para matrices mixtas es una proyección deliberada, ya documentada. |

### NO VALIDADO

- Toda interpretación junguiana: represión como decoherencia, sincronicidad como entrelazamiento, individuación como entropía. Son analogías de modelado; nada en el repositorio las contrasta con datos.
- Las hipótesis H1–H8 de `docs/HIPOTESIS.md`. E001–E004 validan infraestructura matemática y no ponen a prueba ninguna de ellas.

### BLOQUEADO POR DEPENDENCIA EXTERNA

- `analytics/informe_analitico.py` (`generar_informe`, `generar_informe_con_cache`) y la sección "Informe Clínico (IA)" de Streamlit requieren `ANTHROPIC_API_KEY`. Sin la clave se comprobó sólo lo local: el módulo importa, el prompt se construye y la falta de clave produce un `ValueError` claro (y un `st.error` en la app). La llamada real a la API no se ejecutó.

## Problemas encontrados y correcciones

| # | Problema | Tipo | Corrección |
|---|---|---|---|
| 1 | La tomografía era incorrecta para amplitudes complejas: `medir_base_y` devolvía P = 0.5 para cualquier estado y `medir_base_x` usaba (α+β)² sin módulo. | Matemático | P(+) = \|α+β\|²/2 y P(+i) = \|α−iβ\|²/2. |
| 2 | `amplificacion` tenía el signo de la rotación invertido: amplificar Ánima desde P = 0.64 la dejaba en 0.15. | Implementación | polo 0 usa Ry(−θ), polo 1 usa Ry(+θ). |
| 3 | Las gráficas de Streamlit, del menú y del análisis ML dibujaban `1 − γ` como curva teórica de la coincidencia en X; el canal implementado da `1 − γ/2` (como deriva E002). La guía de usuario afirmaba además "anticorrelación perfecta" en γ = 1. | Documentación / código inconsistentes | Curvas y textos corregidos a `1 − γ/2`. |
| 4 | `lindblad.py` afirmaba que el canal de Kraus equivale al de Lindblad con γ₂ = γ. Es falso: la coherencia escala como (1 − γ) frente a e^{−γ₂t} (0 frente a 0.184 en γ = 1). | Conceptual | Documentada la relación correcta, γ₂·t = −ln(1 − γ), con test. |
| 5 | El docstring de `lindblad.py` seguía describiendo integración de Euler y `pasos` como parámetro de precisión. | Documentación | `pasos` documentado como parámetro sin efecto físico. |
| 6 | Un `Arquetipo` transformado (rotación, Hadamard, X) recibía un generador sin semilla: la secuencia semilla → puerta → medición no era reproducible. | Reproducibilidad | El hijo deriva una `SeedSequence` del padre sin consumir su flujo. |
| 7 | `Arquetipo` aceptaba NaN e infinito y devolvía un estado con amplitudes NaN; tomaba 1e-200 por cero y desbordaba con 1e200. | Caso límite | Validación de finitud y norma con `np.hypot`. |
| 8 | `tomografia_bloch` con una base vacía devolvía 0.5 en silencio, y cualquier valor distinto de 0 se contaba como 1. Igual en `inferir_alpha`. | Resultado silenciosamente incorrecto | `ValueError`. |
| 9 | `aplicar_represion` rechazaba `np.float32` y `np.int64`; con `float32` la traza se desviaba 1e-8. | Tipos | Acepta reales de NumPy y calcula en doble precisión. |
| 10 | `SesionTerapeutica` fallaba con `TypeError` al registrar amplitudes complejas. | Implementación | Amplitudes complejas serializadas como `[real, imag]`. |
| 11 | Docstrings: Hadamard "lleva cualquier estado a la superposición máxima" (H\|+⟩ = \|0⟩) y Ry(π) "equivale a la proyección" (fidelidad entre ambos 0.078 para (0.8, 0.6)). | Conceptual | Corregidos. |

Ninguna corrección modificó una hipótesis, una métrica o un valor esperado para hacer pasar un test. Los valores esperados de los tests nuevos se derivan de las ecuaciones del canal y de los estados.

## Limitaciones

- No hay datos empíricos. Todo lo validado es consistencia interna entre código y matemática.
- El canal de desfase actúa sobre un solo qubit de un estado de Bell fijo; no se exploraron otros estados iniciales.
- La tomografía se validó para estados puros de un qubit. El protocolo no incluye estimación por máxima verosimilitud ni intervalos de confianza de la fidelidad.
- La interfaz se probó con AppTest y con un arranque headless; no hubo inspección visual de las gráficas.
- `main.py` es un menú interactivo; se ejecutaron sus funciones una a una, no la navegación por teclado.

## Próximos experimentos

1. **Tomografía de estados mixtos**: aplicar el canal de desfase a un qubit, reconstruir ρ y comparar la pureza estimada con la analítica. Es el paso que E004 declara como extensión.
2. **Sesgo de la fidelidad**: la proyección al interior de la esfera de Bloch sesga la fidelidad hacia abajo para estados puros (1−F cae como 1/√shots, no como 1/shots). Cuantificarlo y comparar con un estimador de máxima verosimilitud.
3. **Contraste con datos**: ninguna de H1–H8 puede evaluarse sin un dataset observable. El primer experimento con contenido empírico corresponde al programa semiótico (ver `saussure-quantum`, S002) y exige un diseño que distinga los modelos antes de recolectar datos.
