# Rúbrica del juez · M3 v1.1

Juez: `Qwen/Qwen2.5-7B-Instruct (4bit)`, greedy, distinto del generador `Qwen/Qwen2.5-1.5B-Instruct`.
Rúbrica multi-criterio: el nivel 1-5 se deriva de cuatro preguntas binarias respondidas con SI/NO (`max_new_tokens=3`), con el contexto recuperado a la vista. Los casos de válvula se califican con una regla determinista.
Prueba de cordura superada: **True**. La vara original de M2 (juez `Qwen/Qwen2.5-1.5B-Instruct`, rúbrica v1.0) se reporta como columna aparte; su prueba de cordura: **False**.

```
Evalúas RESPUESTAS de un asistente que consulta la documentación de un sistema de
riesgo crediticio.

Niveles:
5 = correcta respecto a la referencia, sin datos inventados y completa.
4 = correcta y sin inventos, pero deja fuera parte de lo que se preguntó.
3 = parcialmente correcta: acierta en algo y se equivoca o divaga en el resto.
2 = incorrecta respecto a la referencia.
1 = inventa datos que no están en la referencia ni en el contexto.

El nivel se deriva de cuatro preguntas verificables, en este orden:
  1. ¿La respuesta afirma datos que no están ni en la referencia ni en el contexto?  sí -> 1
  2. ¿La respuesta coincide en lo esencial con la referencia?                        no -> 2
  3. ¿Responde sin divagar ni contradecirse?                                         no -> 3
  4. ¿Responde todo lo que se preguntó?                                              no -> 4
  Las cuatro favorables                                                                 -> 5

Válvula de escape (regla determinista, sin juez):
  - la referencia es la válvula y el sistema se abstiene                                -> 5
  - la referencia es la válvula y el sistema responde algo                              -> 1
  - la respuesta está en el corpus y el sistema se abstiene                             -> 2
La extensión no cuenta: una respuesta más larga no es mejor.
```
