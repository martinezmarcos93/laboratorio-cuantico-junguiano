# E002 — Decoherencia y pérdida de coherencia

## Pregunta

¿Cómo evoluciona un estado correlacionado cuando introducimos ruido de fase?

## Convención

Se utiliza un canal de desfase parametrizado por γ ∈ [0,1]:

- γ=0: estado intacto.
- γ=1: pérdida completa de coherencia de fase en la base Z.

La parametrización se implementa mediante:

K₀ = √(1−γ/2) I  
K₁ = √(γ/2) Z

Para el estado Bell |Φ⁺⟩ = (|00⟩+|11⟩)/√2, el canal multiplica los términos fuera de la diagonal asociados a |00⟩⟨11| y |11⟩⟨00| por (1−γ). Por tanto:

⟨X⊗X⟩ = 1−γ

y, al medir ambos qubits en la base X,

P(x₁=x₂) = (1 + ⟨X⊗X⟩)/2 = 1−γ/2.

La predicción de control es entonces:

- γ=0 → P(igual)=1.
- γ=0.5 → P(igual)=0.75.
- γ=1 → P(igual)=0.5.

El límite γ=1 no implica ausencia de toda correlación física: implica que la correlación X deja de ser superior al azar en esta medición concreta.

## Hipótesis

H1. La coherencia X disminuye monótonamente con γ.
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
- correlación X;
- error absoluto frente a la predicción analítica P_observada − (1−γ/2).

## Control

Comparar siempre el resultado del canal con:
1. estado sin ruido;
2. predicción analítica 1−γ/2;
3. simulación Monte Carlo.

La predicción analítica debe derivarse del canal implementado, no sustituirse por una curva ajustada a los datos.

## Nota

El significado junguiano de "represión" queda fuera de E002. Aquí sólo validamos el canal matemático.
