"""Clasificador de M1 empaquetado para el sistema final: ficha (texto) -> dictamen.

Es el `sistema_m1` del notebook de M2 sin cambios de lógica: BETO afinado con LoRA (adaptador de
M1), el pooler fijado en M2 (pooler.safetensors) y el umbral recalibrado en validación en M2
(0,54). Antes de clasificar aplica las dos abstenciones de M2: "fuera de alcance" si el texto no
tiene la forma de una ficha, y "evidencia insuficiente" si llega con demasiados campos vacíos.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

MODELO_BASE = 'dccuchile/bert-base-spanish-wwm-cased'
LONGITUD_MAX = 320
UMBRAL = 0.54          # recalibrado en validación en M2 (M1 usaba 0,65); ver M2/README.md
MAX_SIN_DATO = 9       # M2: máximo de campos "sin dato" en test (7) + 2

BLOQUES_FICHA = ['Perfil.', 'Ocupación.', 'Situación crediticia.',
                 'Comportamiento de pago.', 'Relación con la entidad.', 'Solicitud.']


def ruta_adaptador_defecto() -> Path:
    aqui = Path(__file__).resolve().parent
    for c in (aqui / 'm1_lora_adapter', aqui.parents[1] / 'M1' / 'm1_lora_adapter'):
        if (c / 'adapter_config.json').exists():
            return c
    raise FileNotFoundError('No encuentro m1_lora_adapter (ni junto a la app ni en M1/).')


def es_ficha(texto):
    """Puerta de dominio de M2: ¿esto tiene forma de ficha de crédito?"""
    return sum(b in texto for b in BLOQUES_FICHA) >= 4


# ---- Señales de riesgo y redacción: copiadas del notebook de M2 (celda 20) -----------------
def _buscar(patron, texto, grupo=1):
    m = re.search(patron, texto)
    return m.group(grupo) if m else None

def _float(valor):
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def senales(ficha, maximo=3):
    """Señales de riesgo presentes en la ficha, en orden de peso. Sólo lee; no infiere."""
    encontradas = []

    moras = _float(_buscar(r'(\d+) moras? en los últimos 24 meses', ficha))
    if moras and moras > 0:
        encontradas.append(f'{int(moras)} mora(s) registrada(s) en los últimos 24 meses')

    peor = _float(_buscar(r'mora máxima histórica de (\d+) días', ficha))
    if peor and peor >= 30:
        encontradas.append(f'mora máxima histórica de {int(peor)} días')

    if 'tipo de empleo informal' in ficha:
        encontradas.append('empleo informal')

    ratio = _float(_buscar(r'relación deuda/ingreso (\d+\.\d+|\d+)', ficha))
    if ratio and ratio >= 0.40:
        encontradas.append(f'relación deuda/ingreso de {ratio:.2f}')

    util = _buscar(r'Utilización de tarjeta (sin dato|\d+)%', ficha)
    if util and util != 'sin dato' and float(util) >= 70:
        encontradas.append(f'utilización de tarjeta en {util}%')

    if 'score de buró externo sin dato' in ficha:
        encontradas.append('score de buró externo ausente')

    vol = _float(_buscar(r'volatilidad del ingreso (\d+\.\d+|\d+)', ficha))
    if vol and vol >= 0.25:
        encontradas.append(f'volatilidad del ingreso de {vol:.2f}')

    monto = _float(_buscar(r'Pide (\d+\.\d+|\d+) millones de pesos a (\d+) meses', ficha))
    plazo = _float(_buscar(r'Pide (?:\d+\.\d+|\d+) millones de pesos a (\d+) meses', ficha))
    if monto and plazo and monto >= 15 and plazo >= 36:
        encontradas.append(f'solicitud de {monto:.1f} millones a {int(plazo)} meses')

    if 'saldos de los últimos seis meses creciente' in ficha:
        encontradas.append('saldos crecientes en el último semestre')

    consultas = _float(_buscar(r'(\d+) consultas? a centrales', ficha))
    if consultas and consultas >= 3:
        encontradas.append(f'{int(consultas)} consultas a centrales en seis meses')

    return encontradas[:maximo]




def nivel_riesgo(p):
    if p >= UMBRAL + (1 - UMBRAL) / 2:  return 'alto'
    if p >= UMBRAL:                     return 'moderado-alto'
    if p >= UMBRAL / 2:                 return 'moderado-bajo'
    return 'bajo'


def redactar_dictamen(ficha, p):
    veredicto = 'incumple' if p >= UMBRAL else 'paga'
    vistas = senales(ficha)
    if vistas:
        sustento = 'Señales en la ficha: ' + '; '.join(vistas) + '.'
    elif veredicto == 'incumple':
        sustento = ('Señales en la ficha: ninguna de las alertas revisadas aparece; el modelo '
                    'marca riesgo por la combinación global del perfil, no por una alerta '
                    'puntual, así que la confianza en este dictamen es menor.')
    else:
        sustento = 'Señales en la ficha: ninguna de las alertas revisadas aparece.'
    accion = ('Recomendación: escalar a revisión manual antes de aprobar.' if veredicto == 'incumple'
              else 'Recomendación: puede seguir el flujo estándar de aprobación.')
    return (f'Veredicto: {veredicto} (probabilidad de incumplimiento a 12 meses {p:.2f}, '
            f'umbral {UMBRAL:.2f}; riesgo {nivel_riesgo(p)}). {sustento} {accion}')


class ClasificadorM1:
    def __init__(self, ruta_adaptador=None, device='cpu'):
        import torch
        from peft import PeftConfig, PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        t0 = time.perf_counter()
        ruta = Path(ruta_adaptador) if ruta_adaptador else ruta_adaptador_defecto()
        cfg = PeftConfig.from_pretrained(str(ruta))
        torch.manual_seed(42)
        base = AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model_name_or_path or MODELO_BASE, num_labels=2,
            id2label={0: 'paga', 1: 'incumple'}, label2id={'paga': 0, 'incumple': 1})
        pooler = ruta / 'pooler.safetensors'
        if not pooler.exists():
            raise FileNotFoundError('Falta pooler.safetensors: sin él las predicciones no se reproducen (M2).')
        base.load_state_dict(load_file(str(pooler)), strict=False)
        self.model = PeftModel.from_pretrained(base, str(ruta)).to(device).eval()
        try:
            self.tok = AutoTokenizer.from_pretrained(str(ruta))
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path or MODELO_BASE)
        self.device = device
        self.segundos_carga = time.perf_counter() - t0

    def probabilidad(self, texto: str) -> float:
        torch = self.torch
        with torch.no_grad():
            enc = self.tok([texto], truncation=True, max_length=LONGITUD_MAX, padding=True,
                           return_tensors='pt').to(self.device)
            logits = self.model(**enc).logits.float()
            return float(torch.softmax(logits, dim=-1)[0, 1])

    def dictamen(self, ficha: str) -> dict:
        """Contrato de M2: ficha -> veredicto, probabilidad, dictamen."""
        if not es_ficha(ficha):
            return {'veredicto': 'fuera_de_alcance', 'p': None,
                    'dictamen': ('Fuera de alcance: el texto recibido no tiene la estructura de una '
                                 'ficha de crédito, así que no emito un dictamen de riesgo.')}
        if ficha.count('sin dato') > MAX_SIN_DATO:
            p = self.probabilidad(ficha)
            return {'veredicto': 'evidencia_insuficiente', 'p': p,
                    'dictamen': ('Evidencia insuficiente: la ficha llega con demasiados campos vacíos '
                                 f'para sostener un dictamen (la probabilidad cruda sería {p:.2f}, '
                                 'pero descansa sobre ausencias). Recomendación: completar la '
                                 'información antes de decidir.')}
        p = self.probabilidad(ficha)
        return {'veredicto': 'incumple' if p >= UMBRAL else 'paga', 'p': p,
                'dictamen': redactar_dictamen(ficha, p)}

    def vivo(self):
        return self.model is not None
