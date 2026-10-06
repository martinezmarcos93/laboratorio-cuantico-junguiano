# E002 — Decoherencia y pérdida de coherencia

## Pregunta

¿Cómo evoluciona un estado correlacionado cuando introducimos ruido de fase?

## Convención

Se utiliza un canal de desfase parametrizado por γ ∈ [0,1]:

- γ=0: estado intacto.
- γ=1: pérdida completa de coherencia de fase.

La parametrización se implementa mediante:

K₀ = √(1−γ/2) I
K₁ = √(γ/2) Z

Para el estado Bell |Φ+⟩ medido en base X, esta convención produce:

P(x₁=x₂) = 1−γ/2? 

NOTA: esta expresión debe derivarse y validarse analíticamente en el test; el experimento no debe confiar en una fórmula escrita de antemano.

## Hipótesis

H1. La coherencia disminuye monótonamente con γ.
H2. La negatividad y la concurrencia disminuyen con γ.
H3. Las correlaciones observadas convergen a la predicción analítica con suficiente muestreo.
H4. El estado permanece físico: hermiticidad, traza 1 y autovalores no negativos dentro de tolerancia numérica.

## Métricas

- traza;
- hermiticidad;
- autovalores mínimos;
- pureza;
- negatividad;
- concurrencia;
- correlación X.

## Control

Comparar siempre el resultado del canal con:
1. estado sin ruido;
2. predicción analítica;
3. simulación Monte Carlo.

## Nota

El significado junguiano de "represión" queda fuera de E002. Aquí sólo validamos el canal matemático.
