# CREDIT · SI4006

Proyecto de curso sobre riesgo de crédito. Tres entregas encadenadas sobre el mismo sistema:
primero se entrena un modelo que lee fichas de solicitantes, después se construye la vara con la
que se mide, y por último se le da una capa de conocimiento consultable.

Equipo: Jaider España, Camila Martínez

| | Qué contiene |
| :-- | :-- |
| [`M1/`](M1/) | Fine-tuning de BETO con LoRA sobre 205k perfiles, y el adaptador resultante |
| [`M2/`](M2/) | Harness de evaluación de tres dimensiones y scorecard del baseline |
| [`M3/`](M3/) | RAG agéntico sobre la documentación del proyecto, medido con el harness de M2 y RAGAS |

Cada carpeta tiene su propio README con los resultados de esa entrega.

## Datos

`perfiles.csv` pesa 135 MB y no está versionado. Hay que colocarlo en `data/` antes de correr los
notebooks de M1 o M2. El de M3 no lo necesita: trabaja sobre un corpus de documentos propio.

## Estructura

    M1/   notebook de fine-tuning y m1_lora_adapter/ (adaptador, pooler, umbral)
    M2/   notebook del harness y m2_salida/ con el scorecard y las auditorías
    M3/   notebook del RAG y m3_salida/ con las tablas, RAGAS y el corpus
    data/ perfiles.csv (no versionado)

## Uso de IA

Usamos Claude durante todo el proyecto. Lo contamos completo porque fue bastante.

**Código.** Partimos de los notebooks de clase (S05, S06, S07 y S10) como base. Claude nos ayudó a
adaptarlos a nuestro dominio y a extenderlos: el criterio de dominio del harness, los experimentos
de sesgo del juez, la auditoría por grupos y el pipeline de indexación de M3. También lo usamos
para depurar, y varios errores los encontró él revisando el código. Una expresión regular que se
comía el punto final de la frase, y por eso dos señales nunca se citaban en los dictámenes. Una
variable que se usaba antes de existir y rompía una celda. Y un caso sin proteger en la función de
auditoría que habría tumbado la corrida justo al final.

**Entender qué estaba pasando.** El problema del pooler lo encontramos preguntándole por qué el
modelo recargado no daba las métricas de M1. Nos explicó la causa y nos propuso dos salidas.
Elegimos la de probar doce semillas y documentar la dispersión, en vez de reconstruir la capa con
un ajuste por gradientes, porque queríamos una solución que pudiéramos explicar con lo visto en
clase.

**Redacción.** Los README y los textos de los notebooks los escribimos con su apoyo, a partir de
los números de nuestras corridas, y los revisamos antes de dejarlos.

**Lo que decidimos nosotros.** El corpus y las preguntas del eval set. Subir el juez de M3 a
Qwen2.5-7B y las tres correcciones de la rúbrica v1.1, después de ver que el de 1.5B contestaba lo
mismo a las cuatro preguntas binarias. Botar el agente libre tras tres intentos con formatos
distintos y poner el router en su lugar, dejando dicho que lo escribimos conociendo el eval set.
Qué medir en cada ablación. Y no reportar la calibración humana de M2: no la hicimos, y cuando se
planteó generar esas notas con la herramienta decidimos que un kappa entre dos modelos no mide lo
que dice medir.

Las corridas las hicimos nosotros y los números de los informes salen de ahí.
