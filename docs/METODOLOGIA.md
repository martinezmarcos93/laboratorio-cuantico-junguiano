# Metodología

## Principio

El laboratorio debe evolucionar de demostración conceptual a plataforma experimental reproducible.

## Capas

1. Matemática.
2. Simulación.
3. Inferencia.
4. Datos.
5. Comparación de modelos.
6. Interpretación simbólica.

Las capas superiores no deben utilizarse para justificar afirmaciones que las capas inferiores no sostienen.

## Modelo clásico de control

Para cada experimento se debe implementar una alternativa clásica razonable.

Ejemplos:
- mezcla de probabilidades;
- cadena de Markov;
- actualización bayesiana;
- regresión;
- modelos de estado clásicos.

El modelo de control no debe ser artificialmente débil. Antes de comparar hay que comprobar su dimensión observable: si es saturado para el diseño, reproduce cualquier dato y la comparación de ajuste no puede favorecer a ningún otro modelo (ver S002 y S003 en `saussure-quantum`).

## Identificabilidad

Ningún experimento pasa a la fase de datos sin:

1. conteo de observables libres del diseño;
2. dimensión de la familia observable de cada modelo;
3. búsqueda de modelos distintos con observables idénticos;
4. recuperación del modelo generador sobre datos sintéticos;
5. análisis de potencia.

## Modelo quantum-like

Puede utilizar:
- vectores de Hilbert;
- matrices de densidad;
- operadores de medición;
- unitarias;
- canales de ruido;
- dinámica Lindblad cuando sea apropiada.

## Jung como capa interpretativa

Las categorías arquetípicas deben mantenerse separadas de la infraestructura matemática.

Esto permite responder experimentalmente:

    ¿el modelo funciona?
    ¿el modelo junguiano mejora algo?
    ¿o sólo cambia la interpretación?

## Prohibiciones metodológicas

No presentar:
- correlación simulada como entrelazamiento físico;
- salida de clasificación como diagnóstico clínico;
- ajuste computacional como demostración de una teoría metafísica;
- coincidencia estadística como prueba de sincronicidad objetiva.

## Resultado esperado

Una plataforma en la que una hipótesis pueda pasar de texto a:
hipótesis → dataset → simulación → ajuste → comparación → informe reproducible.
