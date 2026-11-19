"""CREDIT · app del sistema final (Colab + túnel de Cloudflare, o Hugging Face Spaces).

Rutas:
    /          interfaz de Gradio
    /metrics   latencia p50/p95 y métricas de uso calculadas del log JSON
    /health    health check: revisa que el índice responde, que el generador cargó y que la
               calculadora funciona, sin hacer una pregunta real. 200 si todo está bien, 503 si no
               (o si el modelo todavía está cargando).

Cada petición deja un registro JSON en stdout (la salida de la app) y en logs/peticiones.jsonl.
Las peticiones también se registran en Weights & Biases: en línea si existe WANDB_API_KEY, y en modo
offline (archivos locales en ./wandb, sin cuenta) si no existe. /metrics y la pestaña "Observabilidad"
resumen el log JSON sin depender de W&B.

Variables de entorno (opcionales):
    CREDIT_VERSION   q4_k_m (por defecto) | q8_0 | fp32
    CREDIT_THREADS   hilos de CPU para el modelo (por defecto 2, los del Space gratuito)
    WANDB_API_KEY    clave de W&B (como secreto del Space, nunca en el repositorio)
    WANDB_PROJECT    nombre del proyecto en W&B (por defecto credit-si4006)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import credit_rag as cr

# El Space gratuito tiene 2 vCPU, pero os.cpu_count() puede reportar los núcleos de la máquina anfitriona.
# Más hilos que núcleos reales hace mucho más lento a llama.cpp y a torch.
N_HILOS = int(os.getenv('CREDIT_THREADS', '2'))
try:
    import torch
    torch.set_num_threads(N_HILOS)
except Exception:
    pass

VERSION = os.getenv('CREDIT_VERSION', 'q4_k_m')
INICIO = time.time()

# --------------------------------------------------------------------------------------------
# Logging estructurado: una línea JSON por petición, sin datos personales
# --------------------------------------------------------------------------------------------
RUTA_LOGS = Path('logs'); RUTA_LOGS.mkdir(exist_ok=True)
log = logging.getLogger('credit')
log.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout); _h.setFormatter(logging.Formatter('%(message)s'))
log.addHandler(_h)
_lock_archivo = threading.Lock()


def registrar(evento: dict):
    linea = json.dumps(evento, ensure_ascii=False)
    log.info(linea)
    with _lock_archivo, open(RUTA_LOGS / 'peticiones.jsonl', 'a', encoding='utf-8') as f:
        f.write(linea + '\n')


# --------------------------------------------------------------------------------------------
# Weights & Biases (opcional)
# --------------------------------------------------------------------------------------------
WB = None
MODO_WB = {'modo': None}
CONTADORES = {'peticiones': 0, 'valvula': 0, 'calculadora': 0, 'errores': 0}


def iniciar_wandb():
    global WB
    # Sin clave, W&B corre en modo offline: registra todo en ./wandb sin cuenta, y se puede subir
    # después con `wandb sync wandb/offline-run-*`.
    modo = 'online' if os.getenv('WANDB_API_KEY') else 'offline'
    try:
        import wandb
        WB = wandb.init(project=os.getenv('WANDB_PROJECT', 'credit-si4006'), job_type='produccion', mode=modo,
                        name=f'space-{VERSION}-{datetime.now(timezone.utc):%Y%m%d-%H%M}',
                        config={'version_generador': VERSION, 'top_k': cr.TOP_K, 'embeddings': cr.EMB_MODEL,
                                'reranker': cr.RERANK_MODEL, 'generador': cr.GEN_MODEL})
        WB.define_metric('peticion')
        WB.define_metric('*', step_metric='peticion')
        print(f'W&B activo en modo {modo}:', WB.url if modo == 'online' else WB.dir)
        MODO_WB['modo'] = modo
    except Exception as e:
        WB = None
        print('W&B no se pudo iniciar:', type(e).__name__, str(e)[:200])


def a_wandb(evento: dict):
    if WB is None:
        return
    try:
        n = CONTADORES['peticiones']
        WB.log({'peticion': n,
                'latencia_s': evento['latencia_s'],
                'latencia_recuperacion_ms': evento.get('latencia_recuperacion_ms'),
                'valvula': int(evento['valvula']),
                'uso_calculadora': int(evento['componentes']['herramienta']),
                'error': int(evento['error'] is not None),
                'tasa_valvula': CONTADORES['valvula'] / n,
                'tasa_calculadora': CONTADORES['calculadora'] / n,
                'peticiones_total': n})
    except Exception as e:  # W&B nunca debe tumbar una respuesta
        print('W&B log falló:', type(e).__name__)


# --------------------------------------------------------------------------------------------
# Carga del sistema en segundo plano (el health check responde 503 mientras tanto)
# --------------------------------------------------------------------------------------------
ESTADO = {'sistema': None, 'error': None, 'cargando': True}


def cargar():
    try:
        t0 = time.time()
        indice = cr.Indice(device='cpu')
        gen = (cr.crear_generador(VERSION, n_threads=N_HILOS) if not VERSION.startswith('fp')
               else cr.crear_generador(VERSION))
        ESTADO['sistema'] = cr.SistemaCredit(indice, gen)
        print(f'Sistema listo en {time.time() - t0:.0f} s con {gen.nombre}')
    except Exception as e:
        ESTADO['error'] = f'{type(e).__name__}: {e}'
        print('Error cargando el sistema:', ESTADO['error'])
    finally:
        ESTADO['cargando'] = False


def health_payload():
    base = {'version': VERSION, 'uptime_s': round(time.time() - INICIO),
            'peticiones': CONTADORES['peticiones'], 'wandb': MODO_WB['modo']}
    if ESTADO['cargando']:
        return 503, {'status': 'cargando', **base}
    if ESTADO['sistema'] is None:
        return 503, {'status': 'error', 'detalle': ESTADO['error'], **base}
    h = ESTADO['sistema'].health()
    return (200 if h['ok'] else 503), {'status': 'ok' if h['ok'] else 'degradado', **h, **base}


# --------------------------------------------------------------------------------------------
# Observabilidad propia: métricas calculadas del log JSON (no depende de W&B)
# --------------------------------------------------------------------------------------------
def leer_log():
    ruta = RUTA_LOGS / 'peticiones.jsonl'
    if not ruta.exists():
        return []
    with _lock_archivo, open(ruta, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def metricas():
    ev = leer_log()
    if not ev:
        return {'peticiones': 0}
    lat = sorted(e['latencia_s'] for e in ev)
    pct = lambda q: round(lat[min(len(lat) - 1, int(round(q * (len(lat) - 1))))], 2)
    n = len(ev)
    return {
        'peticiones': n,
        'desde': ev[0]['ts'], 'hasta': ev[-1]['ts'],
        'latencia_p50_s': pct(0.50), 'latencia_p95_s': pct(0.95), 'latencia_max_s': round(lat[-1], 2),
        'tasa_valvula': round(sum(e['valvula'] for e in ev) / n, 3),
        'tasa_calculadora': round(sum(e['componentes']['herramienta'] for e in ev) / n, 3),
        'tasa_rag': round(sum(e['componentes']['rag'] for e in ev) / n, 3),
        'errores': sum(e['error'] is not None for e in ev),
        'version_generador': VERSION,
    }


def tabla_latencias():
    import pandas as pd
    ev = leer_log()
    return pd.DataFrame([{'petición': f'#{i + 1}', 'latencia (s)': e['latencia_s'],
                          'tipo': ('válvula' if e['valvula'] else
                                   'calculadora' if e['componentes']['herramienta'] else 'respuesta')}
                         for i, e in enumerate(ev)]) if ev else pd.DataFrame(
        {'petición': [], 'latencia (s)': [], 'tipo': []})


# --------------------------------------------------------------------------------------------
# Lógica de una petición
# --------------------------------------------------------------------------------------------
AVISO = ('Este asistente consulta la documentación del proyecto de riesgo crediticio. No evalúa '
         'solicitudes reales ni toma decisiones de crédito: esas las toma una persona. Cada consulta se '
         'registra (hora, latencia, componentes usados) sin guardar el texto de la pregunta.')


def _formato(out: cr.Respuesta) -> tuple[str, str]:
    fuentes = []
    for p in out.pasajes:
        if p['fuente'] not in fuentes:
            fuentes.append(p['fuente'])
    partes = []
    if out.valvula:
        partes.append('**El sistema se abstuvo:** la respuesta no está en sus fuentes.')
    if out.llamadas:
        l = out.llamadas[-1]
        partes.append(f"**Calculadora:** `{l['args']['expresion']} = {l['obs']}`")
    partes.append('**Pasajes recuperados de:**\n' + '\n'.join(f'- {f}' for f in fuentes))
    comp = ', '.join(k for k, v in out.componentes.items() if v) or 'ninguno'
    partes.append(f'*Componentes usados: {comp} · {out.segundos:.1f} s*')
    return out.respuesta, '\n\n'.join(partes)


def responder(pregunta: str):
    pregunta = (pregunta or '').strip()
    if not pregunta:
        return '', 'Escribe una pregunta.'
    if ESTADO['sistema'] is None:
        return '', ('El modelo todavía está cargando, intenta en un minuto.' if ESTADO['cargando']
                    else f"El sistema no cargó: {ESTADO['error']}")
    rid, t0, error, out = uuid.uuid4().hex[:12], time.perf_counter(), None, None
    try:
        out = ESTADO['sistema'].responder(pregunta[:1000])
    except Exception as e:
        error = type(e).__name__
    lat = time.perf_counter() - t0

    CONTADORES['peticiones'] += 1
    evento = {
        'ts': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
        'request_id': rid,
        'version': VERSION,
        'latencia_s': round(lat, 3),
        'latencia_recuperacion_ms': round(out.segundos_recuperacion * 1000, 1) if out else None,
        'componentes': out.componentes if out else {'rag': False, 'herramienta': False, 'vision': False},
        'valvula': bool(out and out.valvula),
        'docs_recuperados': sorted({p['id'].split('_')[0] for p in out.pasajes}) if out else [],
        'largo_pregunta': len(pregunta),   # solo la longitud: el texto no se guarda
        'error': error,
    }
    CONTADORES['valvula'] += int(evento['valvula'])
    CONTADORES['calculadora'] += int(evento['componentes']['herramienta'])
    CONTADORES['errores'] += int(error is not None)
    registrar(evento)
    a_wandb(evento)

    if error:
        return '', f'Ocurrió un error ({error}). Quedó registrado con el id `{rid}`.'
    return _formato(out)


# --------------------------------------------------------------------------------------------
# Interfaz
# --------------------------------------------------------------------------------------------
EJEMPLOS = [
    ['¿Qué significa el valor -999 en la columna score_credito?'],
    ['Según la auditoría, ¿cuántas veces mayor es la tasa de falsa alarma en estrato 1 que en estrato 4?'],
    ['¿Cuál es la tasa de interés que cobra la entidad por estos créditos?'],
]

with gr.Blocks(title='CREDIT · SI4006') as demo:
    gr.Markdown('# CREDIT · consulta la documentación del modelo de riesgo crediticio\n' + AVISO)
    with gr.Tab('Consultar la documentación'):
        pregunta = gr.Textbox(label='Pregunta', lines=2,
                              placeholder='¿Por qué no se usa el sexo del solicitante?')
        boton = gr.Button('Preguntar', variant='primary')
        respuesta = gr.Textbox(label='Respuesta', lines=5)
        detalle = gr.Markdown()
        gr.Examples(EJEMPLOS, inputs=pregunta, label='Ejemplos: fácil · con cálculo · fuera de las fuentes')
        boton.click(responder, pregunta, [respuesta, detalle], api_name='responder')
        pregunta.submit(responder, pregunta, [respuesta, detalle])
    with gr.Tab('Observabilidad'):
        gr.Markdown('Métricas calculadas del log JSON de cada petición (lo mismo expone la ruta `/metrics`). '
                    'El log no guarda el texto de las preguntas.')
        resumen = gr.JSON()
        grafica = gr.BarPlot(x='petición', y='latencia (s)', color='tipo', title='Latencia por petición',
                             sort=None)
        actualizar = gr.Button('Actualizar')
        actualizar.click(lambda: (metricas(), tabla_latencias()), None, [resumen, grafica])
        demo.load(lambda: (metricas(), tabla_latencias()), None, [resumen, grafica])
    with gr.Tab('Estado'):
        gr.Markdown('El mismo health check que expone la ruta `/health`.')
        estado = gr.JSON()
        gr.Button('Revisar').click(lambda: health_payload()[1], None, estado)

app = FastAPI()


@app.get('/health')
def health():
    codigo, cuerpo = health_payload()
    return JSONResponse(cuerpo, status_code=codigo)


@app.get('/metrics')
def metrics():
    return JSONResponse(metricas())


app = gr.mount_gradio_app(app, demo, path='/')

if __name__ == '__main__':
    import uvicorn
    iniciar_wandb()
    threading.Thread(target=cargar, daemon=True).start()
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 7860)))
