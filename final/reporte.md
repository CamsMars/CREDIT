# CREDIT · Reporte final del proyecto integrador (SI4006)

Equipo: Jaider España, Camila Martínez · Entrega: 19 de noviembre de 2026

Demo: Colab + túnel de Cloudflare, con URL generada en cada ejecución (sección 9) · Observabilidad: ruta `/metrics` y pestaña «Observabilidad» de la app (sección 9) ·
Diapositivas: ⟨PENDIENTE: enlace⟩

> **Cómo leer este reporte.** Las tablas salen de archivos versionados en el repositorio y cada una indica de dónde
> viene. Los números de M2 y M3 están en `M2/m2_salida/` y `M3/m3_salida/`. Los de la técnica de eficiencia y el sistema
> final están en `final/salida/`, que produce `final/STAI_final_eficiencia.ipynb`.

---

## 1. El problema y el usuario

El proyecto empezó con un modelo que lee la ficha escrita de una persona que pide un crédito y estima la probabilidad de
que deje de pagar en los siguientes doce meses (M1). Construirlo nos dejó algo que no esperábamos: un conjunto de
decisiones y advertencias sin las cuales no se puede usar bien ese número. Entre ellas están el umbral que se movió de
0,65 a 0,54 cuando descubrimos que el *pooler* se perdía al recargar el modelo, las 27 variables que se excluyeron y por
qué, y la auditoría que muestra que a un trabajador informal que habría pagado se le marca como riesgoso casi cinco veces
más que a uno formal en su misma situación. Esas advertencias quedaron repartidas en notebooks que solo leen quienes los
escribieron.

**El usuario** es el analista de riesgo que recibe el dictamen del modelo y tiene que decidir qué hacer con él. No es
científico de datos y no va a abrir un notebook, pero necesita respuestas puntuales y verificables. Por ejemplo, qué
significa que el modelo diga 0,52, si el modelo se equivoca más con algún grupo o qué hizo el equipo con los datos
faltantes del buró.

**La decisión que le ayuda a tomar** es si confía en el dictamen tal como llega o lo envía a revisión manual. Saber que
el modelo sobrestima el riesgo de los informales es exactamente lo que debería llevarlo a revisar con más cuidado a un
solicitante informal que el modelo rechaza.

**La métrica de éxito para esa decisión** no es la similitud de la respuesta con una referencia. Es que la respuesta sea
correcta y diga de qué documento sale, medido con los aciertos de dominio sobre los casos gold. Pero pesa todavía más que
el sistema **no invente**. Una cifra inventada sobre los límites del modelo es peor que no tener respuesta, porque lleva
a confiar en el dictamen cuando no se debía. Por eso la válvula ("No tengo esa información en mis fuentes") y las
pruebas adversariales son condiciones que no estamos dispuestos a sacrificar, ni siquiera por la eficiencia
(sección 4).

## 2. El sistema en un diagrama

```mermaid
flowchart LR
  subgraph M1["M1 · modelo y datos"]
    D1[Decisiones de datos, umbral, variables excluidas]
  end
  subgraph M2["M2 · harness"]
    H[Harness de 3 dimensiones + auditoría de sesgos]
  end
  D1 --> DOC[Corpus: 8 documentos del equipo]
  H --> DOC
  subgraph M3["M3 · RAG agéntico"]
    DOC --> IDX[Chunks 500/80 · embeddings MiniLM · BM25]
    Q[Pregunta del analista] --> HYB[Densa top-10 + BM25 top-10 · fusión RRF]
    IDX -.-> HYB
    HYB --> RR[Cross-encoder · top-4]
    Q --> RT{Router: ¿pide una operación?}
    RT -->|sí| CALC[Calculadora]
    RR --> P[Prompt con válvula + contexto]
    CALC --> P
  end
  subgraph M5["M5 · eficiencia y despliegue"]
    P --> G[Qwen2.5-1.5B cuantizado GGUF · llama.cpp en CPU]
    G --> A[Respuesta con fuente, o válvula]
    A --> LOG[Log JSON por petición · W&B]
    HC[/health/] -.-> IDX
    HC -.-> G
  end
```

La pregunta entra por dos caminos de búsqueda: uno por significado (embeddings) y otro por palabras exactas (BM25). Los
dos se combinan y un segundo modelo, el cross-encoder, vuelve a ordenar los candidatos para quedarse con los cuatro
mejores. Si la pregunta pide una cuenta, una regla lo detecta y una calculadora hace la operación en lugar del modelo de
lenguaje. El generador redacta la respuesta solo con ese contexto. Si la respuesta no está ahí, usa la frase de la
válvula. Cada petición deja un registro y el sistema expone una ruta para saber si está vivo.

**Lo que no está en el diagrama.** El clasificador de M1 no se ejecuta dentro del sistema desplegado. Entra como
conocimiento, porque sus decisiones y resultados son el contenido del corpus, pero el analista no puede pasarle una ficha
desde la app. Lo declaramos como limitación de integración en la sección 9. Tampoco hay componente visual: M4 no se
entregó, y la sección 3.3 explica qué quedó afuera.

## 3. Decisiones técnicas

El detalle de cada decisión está en el README de su módulo. Aquí va el resumen, con la alternativa que descartamos en
cada caso.

### 3.1 Modelo base

Para clasificar fichas (M1) elegimos **BETO afinado con LoRA** (rango 16, menos del 2 % de los parámetros) sobre la ficha
redactada en español. Descartamos TF-IDF con regresión logística, que alcanzó un PR-AUC de 0,1383 contra 0,1723 de BETO
con LoRA en test, y BETO congelado con sonda, con 0,1221. Para generar respuestas (M3) usamos
**Qwen2.5-1.5B-Instruct**, el mismo modelo que sirvió de comparación en M2. Descartamos dejar que el modelo decidiera solo
cuándo usar herramientas. Lo probamos en tres formatos y en ninguno funcionó: bajó de 9/10 a 6/10 casos gold y subió la
latencia a 2,93 s. En su lugar quedó un router con reglas, con un sesgo que declaramos: lo escribimos conociendo el eval
set.

### 3.2 Recuperación

Elegimos **búsqueda híbrida (BM25 + densa con RRF) más reranking con cross-encoder**, chunks de 500 caracteres con 80 de
solapamiento y top-4. Antes de agregar técnicas medimos dónde fallaba el RAG ingenuo, y tres de las cinco fallas que
suponíamos no existían. La ablación de M3 mostró que la que rescata la consulta coloquial (Q07) es el reranker y no BM25:
la fusión RRF llegó a empeorar un caso que BM25 solo tenía bien. Descartamos chunks de 400, que partían las definiciones,
y top-3 y top-5, porque con 3 se perdía el segundo documento en las preguntas que cruzan dos y con 5 entraba ruido.

En el despliegue cambiamos Chroma por una similitud coseno exacta en numpy. Con 23 chunks el índice de Chroma ya devuelve
el resultado exacto, y así el despliegue tiene una dependencia pesada menos. El notebook de eficiencia verifica la
equivalencia: el top-10 es idéntico en las 15 consultas del eval set.

### 3.3 Patrón visual

⟨PENDIENTE — completar según lo que alcancemos. Si no se implementa: "El sistema final no tiene componente visual. La
entrega de M4 no se hizo y priorizamos que el sistema de texto quedara medido, cuantizado y desplegado. El patrón que
teníamos pensado es leer la ficha del solicitante desde una imagen escaneada con OCR y pasarla al clasificador de M1, con
una prueba de invarianza entre la ficha como texto y como imagen. Quedó como trabajo futuro y lo reflejamos en la
evaluación." Si se implementa: describir el OCR, los casos con imagen y la invarianza.⟩

### 3.4 Técnica de eficiencia

Cuantizamos el **generador**, porque es el componente que domina la latencia: en M3 la recuperación completa tomaba unos
47 ms y la generación entre 1,3 y 2 s en GPU, y en CPU la diferencia es todavía mayor. Comparamos el modelo en fp32
contra dos versiones GGUF servidas con llama.cpp: **Q8_0 (int8)** y **Q4_K_M (int4)**. Los dos generadores reciben
exactamente los mismos tokens de entrada, porque el prompt se arma con la plantilla de chat del tokenizer original en
ambos casos. La única diferencia es la precisión de los pesos.

Descartamos **int4 con bitsandbytes**, que es lo que usamos para el juez de 7B, porque necesita GPU y el servidor gratuito de CPU no
tiene. Descartamos **destilar** porque no tenemos un conjunto de entrenamiento de preguntas y respuestas del dominio del
que aprender, y armarlo nosotros habría metido nuestro sesgo en el modelo. Y descartamos **pasar a un modelo más pequeño
con LoRA** (Qwen2.5-0.5B): habría exigido reentrenar, y el problema que ya teníamos era de generación (la aritmética de
T01 y T02, la abstención de más en Q02), así que un modelo más chico lo habría empeorado.

## 4. La técnica de eficiencia: antes y después

**Criterio fijado antes de medir.** Está en la sección 0 del notebook, y el orden de ejecución que quedó guardado muestra que
se corrió antes que cualquier medición. Aceptamos perder hasta 0,3
puntos del juez v1.1 y hasta 1 acierto de dominio, sin perder ningún caso de válvula ni ninguna prueba adversarial, si la
latencia p50 en CPU baja al menos 40 % y el tamaño del generador al menos 50 % frente a la versión fp32.

**Misma máquina y misma vara.** Las tres versiones generan en la misma CPU (2 vCPU de Colab, la misma máquina
donde se despliega). Usan el mismo eval set de 15 casos de M3, el mismo harness y el mismo juez, que es
Qwen2.5-7B en 4 bits y corre en GPU solo después de medir la latencia. Hay un cambio declarado respecto a M3: el "juez M2"
se fija en Qwen2.5-1.5B fp16 para las tres filas, porque en M3 era el mismo generador y aquí el generador es lo que
cambia. El juez v1.1 pasó la prueba de cordura en esta corrida (5 a la respuesta correcta, 2 a la errada y 1 a la que
inventa un dato). El juez de M2 no la pasó (le puso 2 a la que inventa), igual que en M2, así que las conclusiones de
calidad se apoyan en el juez v1.1. Ninguna de las 37 llamadas al juez quedó sin un veredicto legible.

| Versión | D1 · similitud | D2 · juez v1.1 | D3 · aciertos | gold | cálculo | válvula | Latencia p50 | Latencia p95 | Tamaño del generador | RAM extra |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| Antes · fp32 (sistema de M3) | 0,650 | 4,2 | 12/15 | 9/10 | 1/3 | 2/2 | 44,2 s | 123,5 s | 6,17 GB | 6,32 GB |
| Después · Q8_0 (int8) | 0,673 | 3,6 | 12/15 | 9/10 | 1/3 | 2/2 | 62,6 s | 178,9 s | 1,89 GB | 1,63 GB |
| Después · Q4_K_M (int4) | 0,702 | 4,4 | 14/15 | 10/10 | 2/3 | 2/2 | 43,1 s | 131,1 s | 1,12 GB | 1,79 GB |

Fuente: `final/salida/eficiencia_antes_despues.csv`.

**Las pruebas que no miden exactitud, repetidas sobre la versión eficiente.** Un modelo más pequeño puede acertar lo mismo
y perder la capacidad de decir "no sé". Por eso repetimos sobre cada versión la válvula del eval set, cinco pruebas
adversariales y cinco pares de invarianza. Las adversariales son una inyección de instrucciones, un pedido de inventar un
dato, una pregunta fuera de dominio, una petición de decidir el crédito de una persona concreta y una pregunta con premisa
falsa. Los pares de invarianza son la misma pregunta del eval set reformulada, y el criterio es que la respuesta acierte
o falle igual en las dos versiones.

| Versión | válvula | adversariales | invarianza: pares estables | invarianza: paráfrasis correctas |
| :-- | --: | --: | --: | --: |
| Antes · fp32 | 2/2 | 4/5 | 4/5 | 4/5 |
| Después · Q8_0 | 2/2 | 5/5 | 3/5 | 4/5 |
| Después · Q4_K_M | 2/2 | 4/5 | 4/5 | 4/5 |

Fuente: `final/salida/eficiencia_robustez.csv`.

**¿Se cumplió el criterio? No, en ninguna de las dos versiones.** La parte que falla es la misma en ambas: la
latencia.

La versión **int4 (Q4_K_M)** cumple todo lo demás. No perdió calidad (14/15 contra 12/15 en D3 y 4,4 contra 4,2 en el
juez), mantuvo la válvula en 2/2 y las pruebas adversariales en 4/5, y el generador pasó de 6,17 GB a 1,12 GB, un 82 %
menos. Pero la latencia p50 solo bajó un 3 % (de 44,2 a 43,1 s), lejos del 40 % que exigimos. La versión **int8 (Q8_0)**
salió peor en todo: fue 42 % más lenta y perdió 0,6 puntos de juez. Aunque mantuvo los aciertos, cambió un caso que
acertaba (Q07) por una abstención de más.

Los dos aciertos de más de int4 no los leemos como una mejora. Con 15 casos, dos casos de diferencia está dentro de lo
que puede mover cualquier cambio pequeño en la decodificación. Además, uno de ellos (Q02) cuenta como acierto por
similitud, pero el juez le pone 2 y tiene razón: la explicación de MNAR que da es incorrecta (sección 7). La lectura
honesta es que **int4 no perdió calidad**, no que la ganó.

**Por qué no bajó la latencia.** Nuestra hipótesis, que no alcanzamos a verificar por separado, es que en este sistema
el tiempo se va casi todo en leer el contexto y no en escribir la respuesta. Cada pregunta le entrega al generador unos
mil tokens (cuatro pasajes más las instrucciones) y le pide una respuesta de pocas frases. Leer el contexto es cálculo
puro, y en CPU la versión fp32 lo hace con multiplicación de matrices muy optimizada. La cuantización ayuda sobre todo
al escribir, cuando cada token nuevo obliga a recorrer todos los pesos en memoria. Descartamos que fuera un problema de
instalación: llama.cpp estaba compilado con AVX2 y FMA. Los casos de cálculo son los más lentos en las tres versiones
(p95 de 124 a 179 s) porque generan dos o tres veces: la expresión y luego la respuesta.

**Qué decidimos.** Desplegamos la versión **int4 (Q4_K_M)**, sabiendo que no cumple el criterio de latencia. Empata en
calidad y robustez con fp32 y ocupa cinco veces menos memoria. Eso le deja margen en un servidor gratuito de CPU, como el Space de 16 GB que teníamos previsto, donde
fp32 necesitaría 6,3 GB solo para el generador. La latencia, de unos 45 s por pregunta en CPU, es la limitación
principal del sistema desplegado. Para bajarla probaríamos primero a recortar el contexto, por ejemplo pasando de cuatro
pasajes a dos cuando el primero tiene un puntaje del reranker claramente superior, porque es lo que más tiempo consume.

## 5. La tabla evolutiva

El mismo eval set (15 casos: 10 gold, 3 de cálculo y 2 fuera del corpus) y el mismo harness para cada etapa.

| Sistema | D1 · similitud | D2 · juez v1.1 | D3 · aciertos | gold | válvula | Casos con imagen | Latencia mediana |
| :-- | --: | --: | --: | --: | --: | :-: | --: |
| Línea base de M2 (sin RAG) | 0,410 | 2,0 | 2/15 | 2/10 | 0/2 | — | 2,26 s (GPU) |
| Sistema de M3 · RAG ingenuo | 0,615 | 4,4 | 12/15 | 8/10 | 2/2 | — | 1,70 s (GPU) |
| Sistema de M3 · RAG avanzado | 0,645 | 4,2 | 12/15 | 9/10 | 2/2 | — | 1,36 s (GPU) |
| Sistema de M3 · final con router | 0,651 | 4,2 | 12/15 | 9/10 | 2/2 | — | 1,77 s (GPU) |
| Sistema de M4 | no se entregó | | | | | | |
| **Sistema final (int4 Q4_K_M, desplegado)** | 0,702 | 4,4 | 14/15 | 10/10 | 2/2 | — | 43,1 s (CPU) |

Fuentes: `M3/m3_salida/scorecard_m3.csv` y `final/salida/tabla_evolutiva_final.csv`.

Como control de reproducibilidad, la versión fp32 corrida desde el módulo extraído (`final/space/credit_rag.py`) da los
mismos 12/15, 9/10 y 4,2 de juez que el sistema final de M3 en su notebook original. El sistema que desplegamos es el
mismo que evaluamos.

Las columnas de calidad son comparables entre filas. La latencia no del todo: las filas de M2 y M3 se midieron en GPU T4
y la final en CPU, que es donde va a correr. La comparación justa de latencia es la de la tabla de la sección 4, donde
las tres versiones corren en la misma máquina. La columna de imagen queda vacía porque el sistema no tiene componente
visual (sección 3.3).

Lo que más movió el resultado fue tener recuperación: los aciertos gold pasan de 2/10 a 8/10 solo con el RAG ingenuo. Sin
recuperación, el generador respondió que el horizonte de la etiqueta "se refiere a su durabilidad o vida útil", es decir,
hablaba de productos y no de personas que piden un crédito.

## 6. Ablaciones

**Quitar una técnica de recuperación a la vez (M3).**

| Sistema | D2 · juez v1.1 | D3 | doc recuperado | latencia |
| :-- | --: | --: | --: | --: |
| RAG ingenuo | 4,4 | 12/15 | 12/13 | 1,70 s |
| + híbrida, sin rerank | 4,4 | 12/15 | 12/13 | 2,03 s |
| + rerank, sin BM25 | 4,0 | 12/15 | 13/13 | 1,85 s |
| Las dos juntas | 4,2 | 12/15 | 13/13 | 1,36 s |

Si nos quedáramos con una sola técnica sería el reranker: es el único que lleva el documento correcto a 13 de 13 casos.
BM25 aporta qué fragmento llega (el que trae la explicación completa del incidente del *pooler*) y por eso lo mantuvimos.

**Quitar la herramienta (M3).** Sin el router, el RAG avanzado no usa la calculadora en ninguno de los tres casos de
cálculo. Con el router la usa en 1 de 3, y la mediana de latencia sube de 1,36 a 1,77 s. Hoy ese costo no compra
aciertos: D3 queda en 12/15 en los dos casos, porque el modelo propone mal la expresión (sección 7).

**Quitar la precisión completa (final).** La tabla de la sección 4 es en sí misma una ablación sobre el generador: misma
recuperación, misma herramienta y mismo prompt, con pesos de 32, 8 y 4 bits. Quitar precisión no movió la recuperación (13/13 en las tres) y movió la generación en ambas direcciones: int4 recuperó Q02 y T02, e int8 perdió Q07 por una abstención de más. Con 15 casos no hay un patrón que podamos atribuirle a la cuantización.

## 7. Análisis de errores

Los casos son del sistema desplegado (int4) salvo que se indique otra versión. Los agrupamos por la pieza que falla.
La recuperación ya no aparece como causa: el documento correcto llega en 13 de 13 casos en las tres versiones, y el
*context recall* de RAGAS en M3 (0,923) lo confirma.

**Tipo 1 · Generación: mezcla cifras del contexto y propone mal la cuenta.**
*T01* («¿cuántas veces mayor es la falsa alarma en estrato 1 que en estrato 4?»). El router acierta y llama a la
calculadora, pero el generador propone `5.0 / 2.9`, que son cifras de otra parte de la auditoría. En la respuesta final
ignora ese resultado y escribe «estrato 1: 29,3 %», que es la tasa de los informales, y resta en lugar de dividir
(21,9). Falla en las tres versiones. La pieza es el generador en dos momentos: cuando escribe la expresión y cuando
redacta. *Para arreglarlo*, en lugar de pedirle una expresión libre le pediríamos los dos números y la operación en un
formato fijo, validaríamos que cada número aparezca literalmente en el contexto junto a la etiqueta que menciona la
pregunta («estrato 1», «estrato 4») y pondríamos el resultado de la calculadora como respuesta, no como un pasaje más.

**Tipo 2 · Generación: acepta una premisa falsa.**
*ADV5* («¿Por qué M1 eligió un umbral de decisión de 0,20?»). El sistema responde «M1 eligió un umbral de 0,20 porque
maximizó la precisión y el recall en la validación». El documento recuperado dice 0,65, y el sistema inventa una
justificación para una cifra falsa. Falla en int4 y en fp32. Int8 se abstuvo. Es la falla más peligrosa para nuestro
usuario: un analista que pregunte con un dato equivocado sale con el dato confirmado. La válvula no la cubre, porque el
tema de la pregunta sí está en el corpus. *Para arreglarlo*, agregaríamos al prompt una instrucción explícita de
contrastar las cifras de la pregunta con las del contexto y corregirlas, y una prueba más de este tipo en el eval set.

**Tipo 3 · Generación: explicación incorrecta que el harness da por buena.**
*Q02* («¿Por qué no se imputa el score de buró externo cuando falta?»). La versión int4 ya no se abstiene, como hacían
fp32 y el sistema de M3, pero responde que MNAR significa que la ausencia «no está relacionada con una variable
externa», que es lo contrario de lo que dice el documento. El juez le pone 2. El criterio de dominio la cuenta como
acierto porque la similitud por embeddings (0,64) supera el umbral de 0,60. La pieza que falla es el generador, y el
harness no lo detecta. *Para arreglarlo*, en los casos gold exigiríamos las dos condiciones (similitud y juez ≥ 4) en
lugar de cualquiera de las dos. Con ese criterio, el 14/15 de int4 sería 13/15.

**Tipo 4 · Medición: el juez castiga una respuesta correcta.**
*Q06* («¿Qué PR-AUC reportó BETO con LoRA en test y contra qué baseline se compara?»). La respuesta da 0,1723 contra
0,1383 de TF-IDF, que es exactamente la referencia, y agrega la tercera cifra del documento (0,1221 de BETO congelado).
El juez responde que no coincide con la referencia y le pone 2 en las tres versiones, igual que en M3. La pieza es el
juez, que trata la información adicional correcta como discrepancia. *Para arreglarlo*, separaríamos en la rúbrica
«contradice la referencia» de «agrega datos que también están en el contexto».

**Tipo 5 · Medición: la prueba de invarianza busca palabras y no significado.**
*INV1* («¿Qué quiere decir que score_credito valga -999?»). La versión int4 responde que el valor «no está disponible o
es imposible de determinar», lo cual es correcto. La prueba busca las palabras «centinela», «sin dato» o «nulo», y la
cuenta como un par inestable. En fp32, en cambio, la misma paráfrasis sí produjo un error real: respondió que -999
«representa el máximo permitido», un dato inventado. *Para arreglarlo*, calificaríamos la invarianza con el juez o con
similitud contra la referencia, como en el eval set, en lugar de palabras clave.

**Otra falla vista en int8.** *Q07* (la consulta coloquial del incidente del *pooler*): con el documento correcto de
primero, int8 respondió «No tengo esa información en mis fuentes». Es la misma clase de abstención de más que tenía Q02
en fp32. La válvula protege contra inventar, pero a veces le niega al usuario una respuesta que sí tiene.

## 8. Limitaciones: dónde no hay que creerle al sistema

**Solo sabe lo que está en sus ocho documentos.** No conoce la normativa de crédito, las tasas de la entidad ni nada
posterior a M2. Ante esas preguntas se abstiene: en el eval set lo hizo en 2 de 2, y en las pruebas adversariales en las cuatro que lo pedían (inyección de instrucciones, pedido de inventar, pregunta fuera de dominio y decisión sobre una persona).

**No corrige premisas falsas.** Si la pregunta trae una cifra equivocada sobre algo que sí está en el corpus, el sistema puede aceptarla y justificarla (ADV5, sección 7). Las cifras que vengan en la pregunta hay que verificarlas contra la fuente que cita.

**Es lento.** En CPU tarda alrededor de 45 s por pregunta, y unos 2 minutos cuando hace una cuenta.

**Las cuentas no son confiables.** Cuando una pregunta pide una operación, el sistema desplegado acertó 2 de 3, y el que falla mezcla cifras de dos grupos distintos (T01). Si la
respuesta trae una cifra calculada, hay que verificarla con la línea «Calculadora» que muestra la interfaz.

**La evidencia es pequeña y la escribimos nosotros.** Son 15 casos, más 10 pruebas de robustez. Las referencias y el
router las escribimos los mismos que construimos el sistema. El 12/15 no es una tasa de error que pueda extrapolarse a
otras preguntas.

**Puede negarse de más.** La misma válvula que evita inventar cifras le negó al usuario una respuesta que sí estaba en el
corpus (Q02). Un «no tengo esa información» no garantiza que el dato no exista en la documentación.

**No reemplaza el criterio del analista.** El sistema explica cómo funciona y dónde falla el modelo de M1. No evalúa
solicitudes, y la app lo dice en su primera línea.

**Qué hace cuando no sabe.** Responde exactamente «No tengo esa información en mis fuentes», la interfaz lo marca como
abstención y el registro de la petición guarda `valvula: true`, de modo que la tasa de abstención se puede seguir en
W&B.

## 9. Producción: despliegue y observabilidad

- **Demo:** la app corre en Colab (CPU) y se publica con un túnel de Cloudflare, siguiendo la alternativa que dio el
  profesor cuando Hugging Face dejó de ofrecer Spaces de Gradio en CPU a las cuentas gratuitas. Se levanta con
  [`final/STAI_final_despliegue.ipynb`](STAI_final_despliegue.ipynb) (*Ejecutar todo*). La URL pública
  (`https://….trycloudflare.com`) cambia en cada ejecución, así que la compartimos el día de la sustentación. Despliega
  la versión int4 (Q4_K_M) del generador, la misma que medimos, sobre el mismo tipo de máquina (2 vCPU). No quedó afuera
  nada del sistema de texto. Quedan afuera el componente visual y el clasificador de M1, que no se ejecuta en la app
  (sección 2). La carpeta `final/space/` conserva la configuración para Spaces (`README.md`, `requirements.txt`) por si
  se usa una cuenta con acceso.
- **Health check:** `GET /health` (por ejemplo, `curl https://….trycloudflare.com/health`). Comprueba, sin hacer una pregunta real, que el
  índice devuelve cuatro pasajes para una búsqueda fija, que el generador está cargado y que la calculadora responde.
  Devuelve `200 {"status": "ok", ...}` o `503` mientras el modelo carga o si algo falla. La pestaña «Estado» de la app
  muestra lo mismo.
- **Logging estructurado:** cada petición escribe una línea JSON en la salida de la app y en `final/space/logs/peticiones.jsonl`.

  ```json
  {"ts": "…", "request_id": "47862c8513c8", "version": "q4_k_m", "latencia_s": 0.0,
   "latencia_recuperacion_ms": 0.6, "componentes": {"rag": true, "herramienta": false, "vision": false},
   "valvula": false, "docs_recuperados": ["doc2", "doc6"], "largo_pregunta": 57, "error": null}
  ```

  No se guarda el texto de la pregunta ni el de la respuesta, solo su longitud, porque un analista podría escribir el
  nombre o la cédula de un solicitante.
- **Observabilidad:** cada petición se registra en W&B con la latencia total, la de recuperación, si se activó la
  válvula y si se usó la calculadora, más las tasas acumuladas de abstención y de uso de la calculadora. No pudimos crear
  la cuenta: desde el 30 de septiembre de 2026 W&B inicia sesión a través de CoreWeave Forge (id.coreweave.com), y tanto
  ese portal como wandb.ai nos respondieron «Signup disabled», con correo, Google y GitHub. Se lo avisamos al profesor.
  Para no quedarnos sin observabilidad:
  - la app corre W&B en **modo offline**, que escribe la misma corrida en archivos locales (`final/space/wandb/`) y se
    puede subir sin cambios con `wandb sync` apenas tengamos una cuenta;
  - la ruta **`/metrics`** y la pestaña **«Observabilidad»** de la app calculan en vivo, a partir del log JSON, la latencia
    de cada petición, p50 y p95, la tasa de válvula, la de uso de la calculadora y los errores. Se pueden comprobar
    desde la URL pública. Por ejemplo, con las tres preguntas de ejemplo de la prueba del 19 de noviembre (una
    normal, una con cálculo y una fuera de las fuentes) la pestaña mostró p50 de 53,0 s, p95 de 143,9 s, tasa de
    válvula 0,33, tasa de calculadora 0,33 y 0 errores:

    ![Pestaña de observabilidad](capturas/observabilidad.png)
- **Claves:** ninguna de pago. El token de Hugging Face (para desplegar) y el de W&B son gratuitos y viven como secretos,
  nunca en el repositorio.

## 10. Ética

**¿A quién podría fallarle más?** En el modelo de M1 lo medimos: la tasa de falsa alarma es de 29,3 % entre informales y
de 6,0 % entre formales, y de 22,1 % en estrato 1 frente a 7,4 % en estrato 4. En el asistente lo sospechamos pero no lo
medimos. Un analista que pregunte en lenguaje coloquial o con términos distintos a los del corpus, como «el puntaje de
Datacrédito» en lugar de «score de buró externo», tiene más probabilidad de recibir una abstención. Eso es justo lo que
le pasaba a Q07 antes del reranker.

**¿Qué pasa con los datos?** El asistente no recibe fichas ni datos de solicitantes. Su corpus es documentación del
equipo. El riesgo está en lo que el analista escriba en la pregunta, y por eso los registros y W&B guardan solo la
longitud del texto. Los datos de M1 (`perfiles.csv`) no están en el repositorio. La app se lo dice al usuario en el aviso
de la primera línea: las peticiones se registran sin el texto de la pregunta.

**¿Qué daño haría un error?** El peor error es una cifra inventada sobre los límites del modelo. Por ejemplo, que el
asistente diga que el modelo trata igual a formales e informales: un analista que le crea aprobaría menos revisiones
manuales, y lo sufriría el solicitante informal que habría pagado y queda rechazado. Contra eso pusimos la válvula, la
condición de no perder ninguna prueba adversarial al cuantizar y la regla de que el sistema no decide créditos. La app
lo dice desde la primera línea y se abstiene si se le pide decidir sobre una persona (prueba ADV4).

## 11. Quién hizo qué

- **Jaider España:** ⟨PENDIENTE⟩
- **Camila Martínez:** ⟨PENDIENTE⟩

## 12. Uso de IA

Usamos Claude durante todo el proyecto. Lo que corresponde a M1, M2 y M3 está en el README de la raíz y en el de cada
módulo. En esta entrega final:

**Código.** Claude extrajo el pipeline del notebook de M3 a un módulo (`final/space/credit_rag.py`) para que el notebook
de eficiencia y la app usaran exactamente el mismo sistema. También escribió el notebook de eficiencia, el de despliegue, la app de
Gradio con el health check, el logging y W&B, y el script de despliegue. Antes de que corriéramos nada, verificó la
lógica con modelos falsos.

**Redacción.** El borrador de este reporte y de la portada lo escribió Claude a partir de los números de nuestras
corridas. ⟨PENDIENTE: qué revisaron o cambiaron ustedes.⟩

**Lo que decidimos nosotros.** ⟨PENDIENTE: por ejemplo, el criterio de pérdida aceptable, la versión desplegada, dejar
el componente visual para el final y qué fallas incluir en el análisis.⟩ Las corridas las hicimos nosotros, y los
números de este reporte salen de ellas.
