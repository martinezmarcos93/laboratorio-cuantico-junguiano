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

## Punto crítico

Una matriz de densidad mixta no debe convertirse automáticamente en un vector de amplitudes puro. La matriz reconstruida es el objeto científico primario.

## Extensión

Posteriormente se utilizará QST para seguir trayectorias de estados durante intervenciones y procesos de decoherencia.
