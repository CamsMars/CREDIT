"""CREDIT · sistema final (SI4006).

El RAG agéntico de M3 (búsqueda híbrida BM25 + densa con RRF, reranking con cross-encoder y
calculadora decidida por un router) empaquetado como módulo, con el generador intercambiable:

    GeneradorHF    Qwen2.5-1.5B-Instruct con transformers (fp32 en CPU, fp16 en GPU) -> "antes"
    GeneradorGGUF  el mismo modelo cuantizado en GGUF (Q8_0 = int8, Q4_K_M = int4) con
                   llama.cpp -> "después"

Los dos generadores reciben exactamente el mismo texto de entrada: el prompt se arma con la
plantilla de chat del tokenizer de Qwen y llama.cpp recibe esos mismos tokens. Así la única
diferencia entre "antes" y "después" es la precisión de los pesos.

El mismo archivo lo usan el notebook de eficiencia (final/STAI_final_eficiencia.ipynb) y la app
desplegada en Hugging Face Spaces (final/space/app.py): lo que se mide es lo que se despliega.

Cambio respecto a M3: la búsqueda densa se hace con similitud coseno exacta en numpy en lugar de
Chroma. Con 23 chunks el índice HNSW de Chroma ya devuelve el resultado exacto, así que el orden
es el mismo, y el despliegue se ahorra una dependencia pesada. El notebook lo verifica.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------------
# Configuración (idéntica a M3)
# --------------------------------------------------------------------------------------------
SEMILLA = 42
CHUNK_SIZE, OVERLAP, TOP_K, N_CANDIDATOS = 500, 80, 4, 10
MAX_PASOS = 3

EMB_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
RERANK_MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
GEN_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
GGUF_REPO = 'Qwen/Qwen2.5-1.5B-Instruct-GGUF'
GGUF_ARCHIVOS = {
    'q8_0': 'qwen2.5-1.5b-instruct-q8_0.gguf',      # int8
    'q4_k_m': 'qwen2.5-1.5b-instruct-q4_k_m.gguf',  # int4 (k-quants, mixto 4/6 bits)
}

# Qwen2.5-Instruct trae repetition_penalty=1.1 en su generation_config, y transformers lo aplica
# aunque do_sample=False. Para que llama.cpp decodifique igual se le pasa el mismo valor.
REPETITION_PENALTY = 1.1

VALVULA = 'No tengo esa información en mis fuentes.'

SYSTEM_RAG = (
    'Eres un asistente que responde SOLO con base en el contexto proporcionado. '
    f'Si la respuesta no está en el contexto, responde exactamente: "{VALVULA}" '
    'No inventes datos. Menciona la fuente que usaste. Sé breve.')

SYSTEM_EXPRESION = ('Traduces preguntas a operaciones aritméticas. Responde SOLO con la operación, '
                    'sin texto, sin resultado y sin unidades.')

RUTA_CORPUS_DEFECTO = Path(__file__).with_name('corpus_m3.json')


def es_valvula(respuesta: str) -> bool:
    t = str(respuesta).lower()
    return 'no tengo esa información' in t or 'no tengo esa informacion' in t


# --------------------------------------------------------------------------------------------
# Generadores
# --------------------------------------------------------------------------------------------
class Generador:
    """Interfaz común. `generar_chat(msgs, tools, max_new_tokens)` -> texto."""
    nombre = 'generador'
    tamano_bytes: int | None = None
    segundos_carga: float | None = None

    def generar_chat(self, msgs, tools=None, max_new_tokens=180) -> str:  # pragma: no cover
        raise NotImplementedError

    def generar(self, system, user, max_new_tokens=180) -> str:
        msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        return self.generar_chat(msgs, None, max_new_tokens)

    def vivo(self) -> bool:
        return True


class GeneradorHF(Generador):
    """Qwen2.5-1.5B con transformers. Es el generador de M3 ("antes")."""

    def __init__(self, modelo=GEN_MODEL, device=None, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if dtype is None:
            dtype = torch.float16 if self.device == 'cuda' else torch.float32
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(modelo)
        try:
            m = AutoModelForCausalLM.from_pretrained(modelo, dtype=dtype)
        except TypeError:  # transformers < 4.56 usa torch_dtype
            m = AutoModelForCausalLM.from_pretrained(modelo, torch_dtype=dtype)
        self.model = m.to(self.device).eval()
        self.segundos_carga = time.perf_counter() - t0
        self.nombre = f'{modelo} ({str(dtype).replace("torch.", "")}, {self.device})'
        # Tamaño de los pesos tal como quedan cargados (fp32 = 4 bytes por parámetro).
        self.tamano_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

    def generar_chat(self, msgs, tools=None, max_new_tokens=180):
        torch = self.torch
        p = self.tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
        ent = self.tok(p, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**ent, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0][ent['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    def vivo(self):
        return self.model is not None


class GeneradorGGUF(Generador):
    """El mismo Qwen2.5-1.5B cuantizado en GGUF, servido con llama.cpp en CPU ("después").

    El prompt se arma con la plantilla de chat del tokenizer original de Qwen (la misma de
    GeneradorHF) y se le pasa a llama.cpp como lista de tokens. Decodificación greedy."""

    def __init__(self, cuantizacion='q4_k_m', ruta_gguf=None, n_threads=None, n_ctx=4096):
        from llama_cpp import Llama
        from transformers import AutoTokenizer
        t0 = time.perf_counter()
        if ruta_gguf is None:
            from huggingface_hub import hf_hub_download
            ruta_gguf = hf_hub_download(GGUF_REPO, GGUF_ARCHIVOS[cuantizacion])
        self.tok = AutoTokenizer.from_pretrained(GEN_MODEL)
        self.llm = Llama(model_path=str(ruta_gguf), n_ctx=n_ctx, n_gpu_layers=0,
                         n_threads=n_threads or os.cpu_count(), seed=SEMILLA, verbose=False)
        self.segundos_carga = time.perf_counter() - t0
        self.tamano_bytes = Path(ruta_gguf).stat().st_size
        self.nombre = f'{GGUF_REPO}/{Path(ruta_gguf).name} (llama.cpp, cpu)'
        self._eos = [self.tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|endoftext|>')]

    def generar_chat(self, msgs, tools=None, max_new_tokens=180):
        # Igual que GeneradorHF: plantilla a texto y luego el mismo tokenizer -> mismos tokens.
        p = self.tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
        ids = [int(i) for i in self.tok(p)['input_ids']]
        self.llm.reset()
        salida = []
        for tok in self.llm.generate(ids, top_k=1, top_p=1.0, min_p=0.0, temp=0.0,
                                     repeat_penalty=REPETITION_PENALTY, reset=True):
            if tok in self._eos or len(salida) >= max_new_tokens:
                break
            salida.append(tok)
        return self.tok.decode(salida, skip_special_tokens=True).strip()

    def vivo(self):
        return self.llm is not None


def crear_generador(version: str, **kw) -> Generador:
    """version: 'fp32' | 'fp16' | 'q8_0' | 'q4_k_m'."""
    if version in ('fp32', 'fp16'):
        import torch
        dt = torch.float32 if version == 'fp32' else torch.float16
        return GeneradorHF(dtype=dt, device=kw.get('device', 'cpu' if version == 'fp32' else None))
    return GeneradorGGUF(cuantizacion=version, **kw)


# --------------------------------------------------------------------------------------------
# Índice: chunks, búsqueda densa, BM25, RRF y reranking (idéntico a M3)
# --------------------------------------------------------------------------------------------
def partir(texto, size=CHUNK_SIZE, overlap=OVERLAP):
    out, i = [], 0
    while i < len(texto):
        out.append(texto[i:i + size]); i += size - overlap
    return out


def tokenizar(t):
    return re.findall(r'[\w\-]+', t.lower())


class Indice:
    def __init__(self, ruta_corpus=RUTA_CORPUS_DEFECTO, device='cpu', usar_reranker=True):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer
        with open(ruta_corpus, encoding='utf-8') as f:
            self.corpus = json.load(f)
        self.chunks, self.metadatos, self.ids = [], [], []
        for d in self.corpus:
            for j, ch in enumerate(partir(d['texto'])):
                self.chunks.append(ch)
                self.metadatos.append({'fuente': d['fuente'], 'fecha': d['fecha'], 'doc_id': d['id']})
                self.ids.append(f"{d['id']}_c{j}")
        self.st = SentenceTransformer(EMB_MODEL, device=device)
        E = np.asarray(self.st.encode(self.chunks, show_progress_bar=False), dtype=np.float32)
        self.E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        self.bm25 = BM25Okapi([tokenizar(c) for c in self.chunks])
        self.reranker = CrossEncoder(RERANK_MODEL, device=device) if usar_reranker else None

    def _pasaje(self, i, score):
        return {'id': self.ids[i], 'texto': self.chunks[i], 'fuente': self.metadatos[i]['fuente'],
                'score': round(float(score), 4)}

    def buscar_densa(self, consulta, k=TOP_K):
        q = np.asarray(self.st.encode([consulta]), dtype=np.float32)[0]
        s = self.E @ (q / (np.linalg.norm(q) + 1e-12))
        return [self._pasaje(i, s[i]) for i in np.argsort(-s)[:k]]

    def buscar_bm25(self, consulta, k=TOP_K):
        s = self.bm25.get_scores(tokenizar(consulta))
        return [self._pasaje(i, s[i]) for i in np.argsort(-s)[:k]]

    @staticmethod
    def rrf(listas, k_rrf=60, k=TOP_K):
        puntos, ref = {}, {}
        for lista in listas:
            for rango, p in enumerate(lista):
                puntos[p['id']] = puntos.get(p['id'], 0) + 1 / (k_rrf + rango + 1)
                ref[p['id']] = p
        orden = sorted(puntos, key=puntos.get, reverse=True)[:k]
        return [{**ref[i], 'score': round(puntos[i], 4)} for i in orden]

    def buscar_hibrida(self, consulta, k=TOP_K, n=N_CANDIDATOS):
        return self.rrf([self.buscar_densa(consulta, n), self.buscar_bm25(consulta, n)], k=k)

    def rerankear(self, consulta, pasajes, k=TOP_K):
        if self.reranker is None or not pasajes:
            return pasajes[:k]
        s = self.reranker.predict([(consulta, p['texto']) for p in pasajes])
        orden = np.argsort(-np.asarray(s))[:k]
        return [{**pasajes[i], 'score': round(float(s[i]), 3)} for i in orden]

    def recuperar_avanzado(self, consulta, k=TOP_K, n=N_CANDIDATOS):
        return self.rerankear(consulta, self.buscar_hibrida(consulta, k=n, n=n), k=k)

    def vivo(self):
        """El índice responde: una búsqueda fija devuelve TOP_K pasajes."""
        return len(self.recuperar_avanzado('umbral de decisión', TOP_K)) == TOP_K


# --------------------------------------------------------------------------------------------
# Herramienta y router (idéntico a M3)
# --------------------------------------------------------------------------------------------
def calculadora(expresion):
    if not re.fullmatch(r'[\d\s\+\-\*/\(\)\.,%]+', str(expresion) or ''):
        return 'error: solo se admiten expresiones aritméticas'
    try:
        return str(round(eval(str(expresion).replace(',', '.').replace('%', '/100'),
                              {'__builtins__': {}}, {}), 4))
    except Exception as e:
        return f'error: {e}'


ESQUEMA_CALC = [{'type': 'function', 'function': {
    'name': 'calculadora',
    'description': ('Evalúa una expresión aritmética y devuelve el resultado. Úsala SIEMPRE que la '
                    'pregunta pida una operación (cuántas veces mayor, diferencia, resta, suma, cuántos '
                    'faltan) cuyo resultado no esté escrito en el contexto. No la uses si el número ya '
                    'aparece en el contexto.'),
    'parameters': {'type': 'object',
                   'properties': {'expresion': {'type': 'string',
                                                'description': 'Expresión con punto decimal, p. ej. "18.0 / 4.5"'}},
                   'required': ['expresion']}}}]

PATRONES_OPERACION = [
    r'\bcu[aá]nt[oa]s\s+veces\b',
    r'\bdiferencia\b',
    r'\bpuntos\s+porcentuales\b',
    r'\bcu[aá]nt[oa]s\b[^?]*\b(faltan|quedan|sobran|entran|restan|suman|más|menos)\b',
    r'\b(suma|resta|divide|multiplica|promedio)\b',
]


def necesita_calculo(pregunta):
    q = pregunta.lower()
    return any(re.search(p, q) for p in PATRONES_OPERACION)


def _extraer_json(t):
    i, j = t.find('{'), t.rfind('}')
    if i == -1 or j == -1 or j < i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        return None


def extraer_llamada(t):
    obj = None
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)', t, re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
        except Exception:
            obj = None
    if obj is None:
        obj = _extraer_json(t)
    if not isinstance(obj, dict):
        return None
    nombre = obj.get('name') or obj.get('tool')
    args = obj.get('arguments', obj.get('args', {})) or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if nombre != 'calculadora':
        return None
    return nombre, (args if isinstance(args, dict) else {})


def _expresion_valida(t):
    t = str(t).replace(',', '.')
    m = re.search(r'[\d\(][\d\.\s\+\-\*/\(\)]*[\d\)]', t)
    if not m:
        return None
    e = m.group(0).strip()
    return e if re.search(r'[\d\)]\s*[\+\-\*/]\s*[\d\(]', e) else None


def _contexto(pasajes):
    return '\n\n'.join(f"[Fuente: {p['fuente']}]\n{p['texto']}" for p in pasajes)


# --------------------------------------------------------------------------------------------
# Sistema final
# --------------------------------------------------------------------------------------------
@dataclass
class Respuesta:
    respuesta: str
    pasajes: list = field(default_factory=list)
    llamadas: list = field(default_factory=list)
    segundos: float = 0.0
    segundos_recuperacion: float = 0.0

    @property
    def valvula(self):
        return es_valvula(self.respuesta)

    @property
    def componentes(self):
        return {'rag': bool(self.pasajes), 'herramienta': bool(self.llamadas), 'vision': False}


class SistemaCredit:
    """RAG avanzado + calculadora decidida por un router (sistema final de M3)."""

    def __init__(self, indice: Indice, generador: Generador):
        self.indice, self.gen = indice, generador

    def _proponer_calculo(self, pasajes, pregunta, nota=''):
        ctx = f'Contexto:\n{_contexto(pasajes)}\n\nPregunta: {pregunta}' + (f'\n\n{nota}' if nota else '')
        salida = self.gen.generar_chat([{'role': 'system', 'content': SYSTEM_RAG},
                                        {'role': 'user', 'content': ctx}], tools=ESQUEMA_CALC,
                                       max_new_tokens=60)
        pedido = extraer_llamada(salida)
        if pedido:
            e = _expresion_valida(pedido[1].get('expresion', ''))
            if e:
                return e, 'nativa'
        salida = self.gen.generar(SYSTEM_EXPRESION,
                                  ctx + '\n\nEscribe SOLO la operación con los números del contexto y punto '
                                        'decimal. Ejemplos: 18.0 / 4.5 · 72 - 27', max_new_tokens=30)
        e = _expresion_valida(salida)
        return (e, 'texto') if e else (None, None)

    def responder(self, pregunta: str) -> Respuesta:
        t0 = time.perf_counter()
        pasajes = self.indice.recuperar_avanzado(pregunta, TOP_K)
        t_rec = time.perf_counter() - t0
        llamadas, extra, nota = [], '', ''
        if necesita_calculo(pregunta):
            for _ in range(MAX_PASOS - 1):
                expr, via = self._proponer_calculo(pasajes, pregunta, nota)
                if expr is None:
                    break
                obs = calculadora(expr)
                llamadas.append({'tool': 'calculadora', 'args': {'expresion': expr}, 'obs': obs, 'via': via})
                if not obs.startswith('error'):
                    extra = f'\n\n[Fuente: calculadora]\n{expr} = {obs}'
                    break
                nota = f'La operación "{expr}" dio {obs}. Corrígela.'
        r = self.gen.generar(SYSTEM_RAG, f'Contexto:\n{_contexto(pasajes)}{extra}\n\nPregunta: {pregunta}')
        return Respuesta(r, pasajes, llamadas, time.perf_counter() - t0, t_rec)

    # Firma compatible con el harness de M2/M3: sistema(pregunta, devolver_contexto=...)
    def __call__(self, pregunta, devolver_contexto=False):
        out = self.responder(pregunta)
        return (out.respuesta, out.pasajes, out.llamadas) if devolver_contexto else out.respuesta

    def health(self) -> dict:
        """Revisa que todo cargó sin hacer una pregunta real."""
        checks = {}
        try:
            checks['indice'] = self.indice.vivo()
        except Exception as e:  # pragma: no cover
            checks['indice'] = f'error: {type(e).__name__}'
        try:
            checks['generador'] = self.gen.vivo()
        except Exception as e:  # pragma: no cover
            checks['generador'] = f'error: {type(e).__name__}'
        checks['calculadora'] = calculadora('2 + 2') == '4'
        return {'ok': all(v is True for v in checks.values()), 'checks': checks,
                'generador': self.gen.nombre, 'chunks': len(self.indice.chunks)}
