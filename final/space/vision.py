"""Componente visual: la ficha de crédito llega como imagen (escaneo o foto) y se lee con OCR.

Patrón: imagen -> OCR (Tesseract, español) -> normalización -> clasificador de M1. La salida del OCR
se muestra al analista para que la pueda revisar antes de confiar en el dictamen.

También genera las imágenes de prueba: cada ficha del eval set de M2 se dibuja como un documento
escaneado. Son imágenes SINTÉTICAS y llevan una marca visible que lo dice.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

MARCA_SINTETICA = 'IMAGEN SINTÉTICA generada a partir del eval set de M2 · proyecto SI4006'

_FUENTES = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            '/usr/share/fonts/TTF/DejaVuSans.ttf']


def _fuente(tam):
    for f in _FUENTES:
        if Path(f).exists():
            return ImageFont.truetype(f, tam)
    return ImageFont.load_default(size=tam)


def ficha_a_imagen(texto: str, ancho_chars=88, tam=22, marca=True) -> Image.Image:
    """Dibuja una ficha como documento: un bloque por línea ('Perfil.', 'Ocupación.', ...)."""
    bloques = re.split(r'(?=(?:Perfil|Ocupación|Situación crediticia|Comportamiento de pago|'
                       r'Relación con la entidad|Solicitud)\.)', texto)
    lineas = ['SOLICITUD DE CRÉDITO · FICHA DEL SOLICITANTE', '']
    for b in (b.strip() for b in bloques if b.strip()):
        lineas += textwrap.wrap(b, ancho_chars) + ['']
    f, fm = _fuente(tam), _fuente(16)
    alto_linea = int(tam * 1.5)
    img = Image.new('RGB', (int(ancho_chars * tam * 0.56) + 120, alto_linea * len(lineas) + 160), '#FBFAF6')
    d = ImageDraw.Draw(img)
    y = 50
    for i, l in enumerate(lineas):
        d.text((60, y), l, fill='#1A1A1A', font=_fuente(tam + 4) if i == 0 else f)
        y += alto_linea
    if marca:
        d.text((60, img.height - 50), MARCA_SINTETICA, fill='#9A4A12', font=fm)
    return img


def degradar(img: Image.Image, nivel='borrosa') -> Image.Image:
    """Variantes para las pruebas adversariales."""
    if nivel == 'borrosa':  # foto movida tomada de lejos
        chica = img.resize((img.width // 4, img.height // 4))
        return chica.filter(ImageFilter.GaussianBlur(3)).resize(img.size)
    if nivel == 'leve':     # escaneo de baja calidad pero legible
        return img.rotate(1.2, expand=True, fillcolor='#FBFAF6').filter(ImageFilter.GaussianBlur(0.6))
    raise ValueError(nivel)


def normalizar_ocr(t: str) -> str:
    """Une las líneas cortadas y limpia lo típico del OCR sin inventar contenido."""
    t = t.replace('—', '-').replace('–', '-').replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    t = re.sub(r'-\n(?=[a-záéíóúñ])', '', t)           # palabras partidas con guion
    t = re.sub(r'\s*\n\s*', ' ', t)                       # saltos de línea -> espacio
    t = re.sub(r'SOLICITUD DE CR[ÉE]DITO\W+FICHA DEL SOLICITANTE', '', t)
    # El OCR confunde el número 0 con la letra O: una O suelta antes de una palabra en minúscula
    # ("O moras", "O días") es un cero en estas fichas.
    t = re.sub(r'(?<![A-Za-zÁÉÍÓÚáéíóúÑñ0-9])O(?= [a-záéíóúñ])', '0', t)
    t = re.sub(re.escape(MARCA_SINTETICA), '', t)
    t = re.sub(r'IMAGEN SINT[ÉE]TICA.*$', '', t)
    t = re.sub(r'\s+([.,;:%])', r'\1', t)                 # espacios antes de puntuación
    return re.sub(r'\s{2,}', ' ', t).strip()


def leer_imagen(img: Image.Image) -> str:
    import pytesseract
    gris = img.convert('L')
    if gris.width < 1400:  # Tesseract lee mejor con ~30 px de altura de letra
        k = 1400 / gris.width
        gris = gris.resize((int(gris.width * k), int(gris.height * k)))
    crudo = pytesseract.image_to_string(gris, lang='spa', config='--psm 6')
    return normalizar_ocr(crudo)


def tesseract_disponible() -> bool:
    try:
        import pytesseract
        return 'spa' in pytesseract.get_languages(config='')
    except Exception:
        return False
