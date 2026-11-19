# CREDIT · SI4006

**Un asistente que le responde al analista de riesgo, con la fuente de cada respuesta, cómo funciona y dónde se equivoca
el modelo de crédito del proyecto, para que decida si confía en su dictamen o lo envía a revisión manual.** Si la
respuesta no está en la documentación, el asistente lo dice en lugar de inventarla.

| | |
| :-- | :-- |
| **Demo** | Colab + túnel de Cloudflare: [`final/STAI_final_despliegue.ipynb`](final/STAI_final_despliegue.ipynb) imprime la URL pública y el health check (`<URL>/health`) |
| **Observabilidad** | `<URL>/metrics` y pestaña «Observabilidad» de la app; W&B en modo offline (ver la sección 9 del reporte) |
| **Reporte final** | [`final/reporte.md`](final/reporte.md) |
| **Diapositivas** | ⟨PENDIENTE⟩ |

Equipo: Jaider España, Camila Martínez

## Cómo correrlo

**La app, localmente (CPU, sin GPU ni claves):**

```bash
git clone https://github.com/CamsMars/CREDIT && cd CREDIT/final/space
pip install -r requirements.txt
python app.py            # http://localhost:7860  ·  health check: http://localhost:7860/health
```

La primera vez descarga el generador cuantizado (~1 GB) y los modelos de embeddings y reranking. `CREDIT_VERSION=q8_0`
o `fp32` cambia la versión del generador, y `WANDB_API_KEY` (opcional) activa el registro en W&B.

**Las tablas del reporte, en Colab:** abrir [`final/STAI_final_eficiencia.ipynb`](final/STAI_final_eficiencia.ipynb)
con GPU T4 y ejecutar todo (alrededor de una hora). El notebook clona el repositorio, corre las tres versiones del
generador en CPU, las califica con el harness y escribe todo en `final/salida/`. Semilla 42, decodificación greedy y
versiones de modelos y librerías fijadas en `final/salida/versiones_final.json`.

**Desplegar con enlace público:** abrir [`final/STAI_final_despliegue.ipynb`](final/STAI_final_despliegue.ipynb) en Colab (CPU) y
ejecutar todo. Levanta la app y un túnel de Cloudflare, y muestra la URL. Hugging Face ya no ofrece Spaces de Gradio
en CPU a cuentas gratuitas; `final/desplegar_space.py` queda para una cuenta con acceso.

## Índice

| | Qué contiene |
| :-- | :-- |
| [`M1/`](M1/) | Fine-tuning de BETO con LoRA sobre 205k perfiles, y el adaptador resultante |
| [`M2/`](M2/) | Harness de evaluación de tres dimensiones y scorecard del baseline |
| [`M3/`](M3/) | RAG agéntico sobre la documentación del proyecto, medido con el harness de M2 y RAGAS |
| M4 | No se entregó (era opcional este semestre); ver la sección 3.3 del reporte final |
| [`final/`](final/) | Técnica de eficiencia (cuantización GGUF del generador), despliegue y [reporte final](final/reporte.md) |

Cada carpeta de módulo tiene su propio README con los resultados de esa entrega.

## Datos

`perfiles.csv` pesa 135 MB y no está versionado. Hay que colocarlo en `data/` antes de correr los notebooks de M1 o M2.
M3, la carpeta `final/` y la app no lo necesitan: trabajan sobre el corpus de documentos propio.

## Estructura

    M1/      notebook de fine-tuning y m1_lora_adapter/ (adaptador, pooler, umbral)
    M2/      notebook del harness y m2_salida/ con el scorecard y las auditorías
    M3/      notebook del RAG y m3_salida/ con las tablas, RAGAS y el corpus
    final/   notebook de eficiencia, harness_final.py, reporte.md, salida/ (tablas) y space/ (la app desplegada)
    data/    perfiles.csv (no versionado)

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
