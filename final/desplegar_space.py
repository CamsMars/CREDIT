"""
Sube final/space/ a un Space de Hugging Face (lo crea si no existe).

Uso, desde la raíz del repo (en Colab o local):
    pip install -q huggingface_hub
    HF_TOKEN=hf_xxx python final/desplegar_space.py <usuario-o-org>/credit-si4006

El token necesita permiso de escritura (huggingface.co/settings/tokens). Nunca va en el repo.
Si además se pasa WANDB_API_KEY en el entorno, se guarda como secreto del Space.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

if len(sys.argv) != 2:
    sys.exit('Uso: python final/desplegar_space.py <usuario>/<nombre-del-space>')

repo_id = sys.argv[1]
token = os.environ.get('HF_TOKEN')
if not token:
    sys.exit('Falta HF_TOKEN en el entorno.')

api = HfApi(token=token)
api.create_repo(repo_id, repo_type='space', space_sdk='gradio', exist_ok=True)
if os.environ.get('WANDB_API_KEY'):
    api.add_space_secret(repo_id, 'WANDB_API_KEY', os.environ['WANDB_API_KEY'])
    print('Secreto WANDB_API_KEY guardado en el Space.')

carpeta = Path(__file__).resolve().parent / 'space'
api.upload_folder(repo_id=repo_id, repo_type='space', folder_path=str(carpeta),
                  ignore_patterns=['logs/*', '__pycache__/*', '*.pyc'],
                  commit_message='Despliegue del sistema final')
usuario, nombre = repo_id.split('/')
url = f'https://{usuario}-{nombre}.hf.space'.lower().replace('_', '-')
print(f'Space: https://huggingface.co/spaces/{repo_id}')
print(f'App:   {url}')
print(f'Health check: curl {url}/health')
