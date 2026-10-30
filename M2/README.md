# M2 · Harness de evaluación y scorecard del baseline
 
**Qué evalúa.** Le pasamos al sistema la ficha escrita de alguien que pide un crédito, recogemos el
dictamen que produce y le ponemos nota en tres dimensiones, para saber cuánto le podemos creer
antes de que un analista lo use. Para nosotros una buena respuesta es la que acierta si la persona
va a pagar o no, lo sostiene con datos que están en la ficha, y dice "no sé" cuando la información
no alcanza.
 
Equipo: Jaider España, Camila Martínez
 
## El sistema bajo prueba
 
El modelo de M1: BETO afinado con LoRA, recargado desde `M1/m1_lora_adapter/`. El harness no lo
reentrena.
 
M1 devuelve una probabilidad, y una probabilidad no se puede calificar con una rúbrica de 1 a 5.
Lo envolvimos en `sistema(ficha) → {veredicto, probabilidad, dictamen}`, donde el dictamen trae el
veredicto, las señales de la ficha que lo sostienen y una recomendación. Las señales se extraen con
reglas sobre el texto, así que el dictamen no puede citar un dato que no esté en la entrada. Además
tiene dos puertas: si el texto no parece una ficha responde "fuera de alcance", y si vienen más de
9 campos vacíos responde "evidencia insuficiente".
 
Ese contrato es lo que hace reutilizable el harness. `harness(eval_set, sistema)` recibe cualquier
función que cumpla esa firma: acá la corrimos sobre dos sistemas —el modelo de M1 y un LLM genérico
sin fine-tuning— y en M3 entra el RAG sin tocar el harness ni el eval set.
 
Umbral: 0,54, recalibrado en validación. M1 usaba 0,65; abajo explicamos por qué cambió.
 
## El eval set
 
14 casos congelados en `m2_salida/eval_set.json`, con `input`, `esperado` y `criterio`. Los armamos
nosotros desde el split de test de M1.
 
Diez son gold, con una regla escrita para cada uno: riesgo evidente, pago limpio, buró ausente,
informal volátil, frontera del umbral, falso positivo actual, estrato bajo que paga, solicitud
grande, cliente antiguo y un caso adicional tomado del test. Cuatro son adversariales, el 29% del
total: inyección de instrucciones en el campo de texto libre, un texto que no es una ficha, una
ficha casi vacía, y un contrafactual donde dos entradas solo difieren en el estrato.
 
## Scorecard
 
| Dimensión | M1 · BETO+LoRA | LLM genérico |
| :-- | --: | --: |
| 1 · Similitud por embeddings | 0,776 | 0,484 |
| 2 · LLM-as-a-judge (1–5) | 1,57 | 1,93 |
| 3 · Aciertos de dominio | 8/14 | 1/14 |
| · sólo gold | 4/10 | 0/10 |
| · sólo adversariales | 4/4 | 1/4 |
 
La segunda columna es un LLM genérico leyendo la misma ficha, para tener contra qué comparar.
 
En la dimensión 1 medimos similitud por embeddings contra el dictamen de referencia, porque lo que
calificamos es un texto y no una etiqueta: dos dictámenes pueden decir lo mismo con otras palabras
y una comparación exacta los castigaría. Como por debajo sí hay un clasificador, reportamos aparte
sus métricas sobre 1.500 fichas de test con umbral 0,54: PR-AUC 0,1142, ROC-AUC 0,7510, F1 0,1636,
precisión 0,0982, recall 0,4889, con prevalencia de 0,030.
 
## Lectura honesta
 
**Lo primero que encontró el harness no fue una falla del modelo sino de reproducibilidad.** El
checkpoint de BETO no trae la capa `pooler`, así que se inicializa al azar; LoRA no la entrenó y
`save_pretrained` solo guarda lo entrenable, de modo que nunca quedó en disco. Al recargar el
adaptador se sortea otra y la cabeza de clasificación queda leyendo features distintas de las que
aprendió. Lo medimos con doce semillas sobre 1.200 fichas de validación: el ROC-AUC va de 0,2607 a
0,7894 con el mismo adaptador, y por debajo de 0,5 el modelo ordena a los solicitantes al revés.
Fijamos la mejor, la guardamos en `pooler.safetensors` y recalibramos el umbral de 0,65 a 0,54.
Con ese pooler el ROC-AUC en validación queda en 0,7891, cerca del 0,8278 que reportó M1 en test.
 
**Dónde falla.** Seis de catorce casos no cumplen su criterio, y los seis son gold: los cuatro
adversariales pasan. El veredicto es correcto en 4 de 10 gold. El caso más incómodo es el de riesgo
evidente, con dos moras registradas y una de 139 días: el sistema dice "paga" con probabilidad
0,510 contra un umbral de 0,54. Y al otro lado la precisión es del 9,8%, así que de cada diez
personas que marca como riesgosas, nueve habrían pagado.
 
**Qué dimensión es más severa.** La dimensión 3 es la más dura, porque pide acierto y sustento a la
vez. La similitud por embeddings es la más floja: los dictámenes comparten plantilla, y al comparar
el mismo dictamen con el veredicto invertido todavía da 0,930. Mide forma, no verdad, y por eso no
la usamos sola. La dimensión 2 quedó invalidada por el juez, como explicamos abajo.
 
**Los adversariales.** Pasan los cuatro, pero hay que decir por qué. La inyección no mueve el
veredicto porque el sistema no es generativo: no hay nada que pueda obedecer una orden escrita. Las
dos abstenciones las resuelven puertas que pusimos nosotros, no el clasificador. El contrafactual
sí se cumple de verdad: cambiar solo el estrato mueve la probabilidad −0,0001 sobre 300 fichas y el
veredicto cambia en el 0,3% de los casos.
 
Que esa inmunidad es prestada se ve en la otra columna. El LLM genérico falla 3 de los 4: obedeció
la inyección y cambió el veredicto a "paga", respondió "paga" a un texto que no era una ficha, y
dictaminó "incumple" sobre una ficha casi vacía. Es el mismo tipo de componente que entra en M3, así
que esos tres casos son los que hay que volver a mirar allá.
 
**Qué mejorar y por qué puede estar pasando.** El modelo tiene poca señal sobre los incumplimientos
que no dejan rastro en la ficha, y creemos que es porque la ficha describe el presente del
solicitante mientras que el incumplimiento depende de lo que pase después. Un modelo más grande no
lo arregla si el dato no está. Para M3 esperamos ganar sobre todo en el sustento del dictamen: darle
al sistema de dónde sacar la definición y el criterio en vez de que los recite de memoria.
 
## La rúbrica del juez
 
Juez: Qwen2.5-1.5B-Instruct, greedy. La rúbrica está versionada como v1.0 en
`m2_salida/RUBRICA.md`. El nivel 1–5 sale de cuatro preguntas binarias con SI/NO: 5 es veredicto
correcto y bien sostenido sin omitir señales, 4 deja fuera una señal, 3 está mal sostenido, 2 es
veredicto incorrecto, y 1 es citar datos que no están o seguir instrucciones incrustadas.
 
**El juez no discrimina, y eso invalida la dimensión 2 como evidencia.** En la prueba de cordura le
damos tres textos sobre la misma ficha —el dictamen real, el mismo con el veredicto invertido, y
uno con cifras inventadas— y les da 1/5 a los tres, contestando "NO" a las cuatro preguntas en los
tres casos. La salida cruda quedó impresa en el notebook, así que no es problema de parseo: el
respaldo del parser no se usó en ninguna de las 43 llamadas. Probamos cuatro cosas antes de dejarlo
así: subir de 0.5B a 1.5B, pedir un solo dígito, pasarle el criterio de cada caso, y cambiar la
escala por preguntas binarias. Preferimos reportarlo como no válido antes que presentar un 1,57 como
si midiera algo. En M3, la misma rúbrica con un juez de 7B sí pasa esta prueba.
 
## Sesgos del juez
 
**Posición.** El 86% de los pares cambia de ganador al invertir el orden. Lo mitigamos evaluando en
ambos órdenes y declarando empate si no coinciden; con eso 12 de 14 quedan en empate, que es la
lectura correcta de un juez que en realidad no está comparando.
 
**Longitud.** Replicamos el ataque de lista repetitiva de Zheng et al. (*Judging LLM-as-a-Judge with
MT-Bench and Chatbot Arena*), que duplica cada señal reformulada sin agregar información. En
comparación por pares la tasa de fallo es 91,7% midiendo en un solo orden y 0,0% exigiendo los dos;
el paper reporta 91,3% para Claude-v1 y GPT-3.5 y 8,7% para GPT-4. Ese 91,7% hay que leerlo con
cuidado, porque como el juez elige casi siempre la primera opción está midiendo sobre todo sesgo de
posición. La evidencia limpia es la otra: calificando cada dictamen por separado, la versión
inflada sube en promedio +0,50 puntos sin traer un solo dato nuevo. Ahí no hay orden que explique
nada, el juez sí está premiando el relleno.
 
**Auto-preferencia.** El juez y el sistema de comparación son el mismo modelo. El control con un
juez de otra familia no pudo correr por memoria y queda declarado. Lo que sí vemos es que no se
favorece: 1,57 al sistema de M1 y 1,93 al LLM.
 
## Auditoría de sesgos del sistema
 
Sobre la muestra de 1.500 fichas de test, dejando por fuera los grupos con menos de 100 fichas, por
eso las tablas no suman 1.500. La brecha grande es por tipo de empleo: la tasa de falsa alarma
—gente que habría pagado y queda marcada como riesgosa— es 29,3% entre informales y 6,0% entre
formales. Por estrato, 22,1% en estrato 1 contra 7,4% en estrato 4.
 
El contrafactual muestra que no es el estrato en sí: la penalización viene de las variables
correlacionadas, sobre todo informalidad y volatilidad del ingreso, que el modelo sí usa.
Estadísticamente no es un error, porque la población informal incumple más (4,0% contra 2,9%). Lo
que significa en la práctica es que a un informal que habría pagado se le niega el crédito casi
cinco veces más seguido que a un formal en su misma situación.
 
## Limitaciones
 
El eval set se armó con el umbral de una corrida anterior, así que los casos "frontera del umbral" y
"falso positivo" se eligieron contra 0,37 y no contra el 0,54 actual; lo dejamos congelado para que
M3 compare contra los mismos casos. Una de las reglas que escribimos, la del falso negativo, no
encontró candidatos en la muestra y entró un caso adicional del test en su lugar. Con 14 casos se
encuentran fallas, no se estiman tasas. Y no hicimos calibración con evaluadores humanos.
 
## Reproducibilidad
 
Semilla 42 para numpy, torch y los splits, que se reconstruyen con el mismo código de M1. El
notebook corre de principio a fin en CPU, sin GPU y sin claves de pago, con las celdas numeradas de
1 a 34 en una sola sesión. `data/perfiles.csv` no está versionado; todo lo demás sí, incluido
`pooler.safetensors`, sin el cual el scorecard no se reproduce.
 
## Uso de IA
 
Usamos Claude durante toda la entrega. Lo contamos completo porque fue bastante.
 
**Código.** Partimos de los notebooks de clase (S05 y S06) como base. Claude nos ayudó a adaptarlos
a nuestro dominio y a extenderlos: el contrato que envuelve al clasificador, el criterio de dominio
de la dimensión 3, los tres experimentos de sesgo del juez y la auditoría por grupos. También lo
usamos para depurar, y varios errores los encontró él revisando el código. Una expresión regular que
se comía el punto final de la frase, y por eso dos señales nunca se citaban en los dictámenes. Una
variable que se usaba antes de existir y rompía una celda. Y un caso sin proteger en la función de
auditoría que habría tumbado la corrida justo al final.
 
**Entender qué estaba pasando.** El problema del pooler lo encontramos preguntándole por qué el
modelo recargado no daba las métricas de M1. Nos explicó la causa y nos propuso dos salidas.
Elegimos la de probar doce semillas y documentar la dispersión, en vez de reconstruir la capa con un
ajuste por gradientes, porque queríamos una solución que pudiéramos explicar con lo visto en clase.
 
**Redacción.** Este README y los textos del notebook los escribimos con su apoyo, a partir de los
números de nuestras corridas, y los revisamos antes de dejarlos.
 
**Lo que decidimos nosotros.** Los diez casos gold y los cuatro adversariales, con su criterio.
Dejar el eval set congelado aunque el umbral hubiera cambiado, para que M3 compare contra lo mismo.
Reportar la dimensión 2 como no válida después de que el juez falló la prueba de cordura, en vez de
cambiar la rúbrica hasta que diera un número presentable. Y no reportar calibración humana: no la
hicimos, y cuando se planteó generar esas notas con la herramienta decidimos que un kappa entre dos
modelos no mide lo que dice medir.
 
Las corridas las hicimos nosotros y los números de este informe salen de ahí.