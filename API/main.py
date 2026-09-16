"""Application web FastAPI pour utiliser le checkpoint de traduction.

Depuis la racine du projet : ``uvicorn API.main:app --reload``
"""

from contextlib import asynccontextmanager
from pathlib import Path
import sys
import threading

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translate import load_model, load_tokenizer, translate  # noqa: E402

CHECKPOINT = ROOT / "checkpoint.pt"
TOKENIZER = ROOT / "tokenizer.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le checkpoint une seule fois au démarrage."""
    if not CHECKPOINT.is_file() or not TOKENIZER.is_file():
        raise RuntimeError("checkpoint.pt ou tokenizer.json est introuvable à la racine du projet.")
    app.state.model = load_model(CHECKPOINT, DEVICE)
    app.state.tokenizer = load_tokenizer(TOKENIZER)
    yield


app = FastAPI(title="Traducteur Français-Lingala", version="1.0.0", lifespan=lifespan)


class TranslationRequest(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=1000)
    method: str = Field("beam", pattern="^(greedy|beam)$")
    max_len: int = Field(64, ge=2, le=256)
    beam_size: int = Field(4, ge=1, le=10)


class TranslationResponse(BaseModel):
    translation: str
    method: str


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/translate", response_model=TranslationResponse)
def translate_api(request: TranslationRequest):
    sentence = request.sentence.strip()
    if not sentence:
        raise HTTPException(status_code=422, detail="La phrase ne peut pas être vide.")
    with LOCK:
        output = translate(
            sentence=sentence,
            model=app.state.model,
            tokenizer=app.state.tokenizer,
            device=DEVICE,
            max_len=request.max_len,
            method=request.method,
            beam_size=request.beam_size,
        )
    return TranslationResponse(translation=output, method=request.method)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return """<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Traducteur</title><style>body{max-width:700px;margin:4rem auto;padding:0 1rem;font:16px system-ui;background:#f5f7fb}main{padding:2rem;background:white;border-radius:14px;box-shadow:0 5px 24px #0002}textarea,select,button{box-sizing:border-box;width:100%;padding:.75rem;font:inherit;border-radius:8px;border:1px solid #cbd5e1}textarea{height:110px}label{display:block;margin:1rem 0 .4rem}button{margin-top:1rem;background:#2563eb;color:white;border:0;font-weight:bold;cursor:pointer}#result{white-space:pre-wrap;padding:1rem;background:#eff6ff;border-radius:8px}.error{color:#b91c1c}</style></head><body><main><h1>Traducteur Français → Lingala</h1><form id="form"><label>Phrase en français</label><textarea id="sentence" required></textarea><label>Méthode</label><select id="method"><option value="beam">Beam search</option><option value="greedy">Greedy</option></select><button>Traduire</button></form><h2>Traduction</h2><div id="result">—</div><script>const f=document.querySelector('#form'),r=document.querySelector('#result'),s=document.querySelector('#sentence'),m=document.querySelector('#method');f.onsubmit=async e=>{e.preventDefault();r.className='';r.textContent='Traduction en cours…';try{let x=await fetch('/translate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sentence:s.value,method:m.value})}),d=await x.json();if(!x.ok)throw Error(d.detail||'Erreur');r.textContent=d.translation||'(aucune traduction)'}catch(e){r.className='error';r.textContent=e.message}}</script></main></body></html>"""
