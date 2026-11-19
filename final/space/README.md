---
title: CREDIT SI4006
emoji: 📄
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
python_version: "3.13"
app_file: app.py
pinned: false
---

# CREDIT · Space del proyecto final de SI4006

Asistente que responde preguntas sobre la documentación del modelo de riesgo crediticio del proyecto
(RAG híbrido + reranking + calculadora). Generador: Qwen2.5-1.5B-Instruct cuantizado en GGUF y servido con
llama.cpp en CPU.

- **Health check:** `GET /health` → `200` con `{"status": "ok", ...}` cuando el índice, el generador y la
  calculadora responden; `503` mientras el modelo carga o si algo falla.
  `curl https://<usuario>-<space>.hf.space/health`
- **Logs:** una línea JSON por petición en la pestaña *Logs* del Space (hora, latencia, componentes usados,
  válvula). El texto de la pregunta no se guarda.
- **W&B:** opcional. Agrega `WANDB_API_KEY` como *secret* del Space (Settings → Variables and secrets).
  Sin la clave la app funciona igual.
- **Versión del generador:** variable `CREDIT_VERSION` (`q4_k_m` por defecto, `q8_0` o `fp32`).

Código fuente y evaluación: https://github.com/CamsMars/CREDIT (carpeta `final/`).
