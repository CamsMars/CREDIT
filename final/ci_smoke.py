"""Prueba de humo del sistema final (corre en GitHub Actions en cada push).

Carga el sistema desplegado (índice + generador int4 Q4_K_M con llama.cpp en CPU), revisa el
health check y responde 4 casos del eval set de M3, uno de cada tipo. No usa el juez de 7B (no
cabe en un runner de GitHub): cada caso tiene una comprobación determinista, tomada del criterio
de dominio del harness.

    Q03  gold          la respuesta menciona que la variable es "protegida"
    Q06  gold (cifra)  la respuesta da el PR-AUC 0,1723
    T02  herramienta   usa la calculadora y la respuesta da 45
    A01  adversarial   activa la válvula ("No tengo esa información en mis fuentes")

Falla (exit 1) si el health check no da ok, si A01 no se abstiene, o si acierta menos de 3 de 4.
Escribe el resultado en el resumen del job (GITHUB_STEP_SUMMARY) y en final/salida/ci_smoke.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'final' / 'space'))
import credit_rag as cr  # noqa: E402

CASOS = [
    {'id': 'Q03', 'tipo': 'gold', 'claves': ['protegida']},
    {'id': 'Q06', 'tipo': 'gold (cifra)', 'claves': ['0,1723', '0.1723']},
    {'id': 'T02', 'tipo': 'herramienta', 'numero': 45, 'usa_calculadora': True},
    {'id': 'A01', 'tipo': 'adversarial', 'valvula': True},
]
MIN_ACIERTOS = 3


def numeros(t):
    return [float(x.replace(',', '.')) for x in re.findall(r'\d+(?:[.,]\d+)?', str(t))]


def revisar(caso, out):
    r = out.respuesta
    if caso.get('valvula'):
        return out.valvula, 'se abstuvo' if out.valvula else 'NO se abstuvo'
    if out.valvula:
        return False, 'se abstuvo teniendo la respuesta'
    if 'numero' in caso:
        ok = any(abs(x - caso['numero']) < 1e-9 for x in numeros(r))
        if caso.get('usa_calculadora') and not out.llamadas:
            return False, 'no usó la calculadora'
        return ok, f"{'da' if ok else 'NO da'} {caso['numero']}"
    ok = any(k.lower() in r.lower() for k in caso['claves'])
    return ok, f"{'menciona' if ok else 'NO menciona'} {caso['claves'][0]}"


def main():
    eval_set = {c['id']: c for c in json.load(open(RAIZ / 'M3' / 'm3_salida' / 'eval_set_m3_v2.json', encoding='utf-8'))}
    hilos = int(os.getenv('CREDIT_THREADS', os.cpu_count() or 2))
    t0 = time.perf_counter()
    sistema = cr.SistemaCredit(cr.Indice(device='cpu'), cr.crear_generador('q4_k_m', n_threads=hilos))
    carga = time.perf_counter() - t0
    health = sistema.health()
    print('health:', json.dumps(health, ensure_ascii=False))

    filas = []
    for caso in CASOS:
        out = sistema.responder(eval_set[caso['id']]['input'])
        ok, nota = revisar(caso, out)
        filas.append({'id': caso['id'], 'tipo': caso['tipo'], 'ok': ok, 'nota': nota,
                      'segundos': round(out.segundos, 1), 'calculadora': bool(out.llamadas),
                      'respuesta': out.respuesta[:200]})
        print(f"{'✓' if ok else '✗'} {caso['id']} ({caso['tipo']}) · {nota} · {out.segundos:.1f} s")

    aciertos = sum(f['ok'] for f in filas)
    valvula_ok = next(f['ok'] for f in filas if f['id'] == 'A01')
    exito = health['ok'] and valvula_ok and aciertos >= MIN_ACIERTOS
    resultado = {'exito': exito, 'aciertos': f'{aciertos}/{len(filas)}', 'health_ok': health['ok'],
                 'valvula_ok': valvula_ok, 'carga_s': round(carga, 1), 'hilos': hilos,
                 'generador': health['generador'], 'casos': filas}
    salida = RAIZ / 'final' / 'salida'
    salida.mkdir(parents=True, exist_ok=True)
    (salida / 'ci_smoke.json').write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding='utf-8')

    resumen = [f"## Prueba de humo · {'✅ pasa' if exito else '❌ falla'}",
               f"Health check: {'ok' if health['ok'] else 'FALLA'} · aciertos {aciertos}/{len(filas)} "
               f"(mínimo {MIN_ACIERTOS}, válvula obligatoria) · generador {health['generador']} · carga {carga:.0f} s",
               '', '| Caso | Tipo | Resultado | Detalle | Latencia |', '| :-- | :-- | :-: | :-- | --: |']
    resumen += [f"| {f['id']} | {f['tipo']} | {'✓' if f['ok'] else '✗'} | {f['nota']} | {f['segundos']} s |" for f in filas]
    texto = '\n'.join(resumen)
    print('\n' + texto)
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
            f.write(texto + '\n')
    sys.exit(0 if exito else 1)


if __name__ == '__main__':
    main()
