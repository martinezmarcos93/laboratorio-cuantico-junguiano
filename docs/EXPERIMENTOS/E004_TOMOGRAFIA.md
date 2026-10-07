# E004 — Tomografía de estado

## Objetivo

Reconstruir un estado de un qubit a partir de mediciones en bases Z, X e Y.

## Procedimiento

1. definir un estado conocido;
2. generar muestras Z/X/Y;
3. estimar el vector de Bloch;
4. reconstruir ρ;
5. comparar ρ reconstruida con ρ verdadera;
6. repetir con diferentes tamaños muestrales.

## Métricas

- fidelidad de Uhlmann;
- error de cada componente del vector de Bloch;
- pureza;
- positividad;
- error de reconstrucción.

## Hipótesis

La fidelidad aumenta con el tamaño muestral y el error estadístico disminuye aproximadamente con la escala esperada de una estimación Bernoulli.

## Probabilidades de medición

Para |ψ⟩ = α|0⟩ + β|1⟩:

- base Z: P(0) = |α|²;
- base X: P(+) = |α + β|²/2;
- base Y: P(+i) = |α − iβ|²/2.

Las tres bases son necesarias cuando las amplitudes tienen fase relativa compleja (r_y = 2·Im(α*β) ≠ 0). La simulación de la base Y devolvía 0.5 para cualquier estado y la de la base X omitía el módulo; con eso la tomografía de |+i⟩ convergía a un estado con fidelidad ≈ 0.5. Corregido y cubierto por `tests/test_validacion_local.py`.

## Punto crítico

Una matriz de densidad mixta no debe convertirse automáticamente en un vector de amplitudes puro. La matriz reconstruida es el objeto científico primario.

## Extensión

Posteriormente se utilizará QST para seguir trayectorias de estados durante intervenciones y procesos de decoherencia.
