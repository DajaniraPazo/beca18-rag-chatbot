"""Script to generate the beca18_rag_chatbot.ipynb notebook."""
import json, os

cells = []
_id = 0

def md(source: str) -> dict:
    global _id; _id += 1
    return {"cell_type": "markdown", "id": f"md{_id:03d}", "metadata": {}, "source": source}

def code(source: str) -> dict:
    global _id; _id += 1
    return {"cell_type": "code", "execution_count": None, "id": f"c{_id:03d}",
            "metadata": {}, "outputs": [], "source": source}

# ── TITLE ──────────────────────────────────────────────────────────────────
cells.append(md(
"""# 🎓 Beca 18 RAG Chatbot — Normativa PRONABEC 2026

**Sistema de Generación Aumentada por Recuperación (RAG)** para responder preguntas
sobre la Resolución Directoral Ejecutiva N.° 033-2026-MINEDU/VMGI-PRONABEC.

| Componente | Detalle |
|---|---|
| Extracción | `pypdf` + marcadores `[PAGE N]` |
| Tokenización | `tiktoken cl100k_base` |
| Segmentación | `RecursiveCharacterTextSplitter` (400/60) |
| Embeddings | `gemini-embedding-001` · 768 dim |
| Base vectorial | `ChromaDB` · distancia coseno |
| Generación | `gemini-2.5-flash` · solo en contexto |
| Interfaz | `ipywidgets` interactivo |

> ⚠️ **Requiere:** archivo `.env` con `GEMINI_API_KEY=<tu_clave>` en la raíz del repo.
> Obten tu clave gratis en https://aistudio.google.com/app/apikey"""))

# ── STEP 0: SETUP ──────────────────────────────────────────────────────────
cells.append(md("## Paso 0 — Configuración del entorno"))

cells.append(code(
"""# Instalar dependencias
%pip install -q \\
    pypdf==4.3.1 \\
    tiktoken==0.8.0 \\
    "langchain-text-splitters==0.3.3" \\
    "google-genai==1.15.0" \\
    "chromadb==0.6.3" \\
    "ipywidgets==8.1.5" \\
    "tqdm==4.67.1" \\
    "python-dotenv==1.1.0"
print("✓ Dependencias instaladas")"""))

cells.append(code(
"""import os, sys, importlib.metadata
from dotenv import load_dotenv

# Buscar .env en el directorio actual y en el directorio padre (para Colab)
for path in [".", "..", "/content"]:
    if load_dotenv(os.path.join(path, ".env")):
        break

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY no encontrada.\\n"
        "Crea un archivo .env con: GEMINI_API_KEY=tu_clave\\n"
        "Obtén tu clave en: https://aistudio.google.com/app/apikey"
    )
print("✓ API key cargada correctamente")
print(f"  Clave (primeros 8 caracteres): {GEMINI_API_KEY[:8]}...")"""))

cells.append(code(
"""# Verificar versiones de paquetes instalados
PACKAGES = [
    "pypdf", "tiktoken", "langchain-text-splitters",
    "google-genai", "chromadb", "ipywidgets", "tqdm", "python-dotenv",
]
print(f"Python: {sys.version.split()[0]}")
print("-" * 35)
for pkg in PACKAGES:
    try:
        ver = importlib.metadata.version(pkg)
        print(f"  {pkg:<30} {ver}")
    except importlib.metadata.PackageNotFoundError:
        print(f"  {pkg:<30} NO ENCONTRADO")"""))

# ── STEP 1: PDF EXTRACTION ──────────────────────────────────────────────────
cells.append(md(
"""## Paso 1 — Extracción de texto del PDF

Se extrae el texto página por página con `pypdf`, se inserta un marcador `[PAGE N]`
al inicio de cada página y se aplica limpieza ligera:
- Colapsar espacios múltiples.
- Eliminar líneas en blanco excesivas.
- Preservar los números de página para citar fuentes."""))

cells.append(code(
"""import os, re, requests
from pypdf import PdfReader

PDF_PATH  = "data/beca18_reglamento.pdf"
PDF_URL_1 = ("https://cdn.www.gob.pe/uploads/document/file/7778068/"
             "7778068-rde-n-033-2026-minedu-vmgi-pronabec.pdf")
PDF_URL_2 = ("https://cdn.www.gob.pe/uploads/document/file/7778068/"
             "rde-n-033-2026-minedu-vmgi-pronabec.pdf")

os.makedirs("data", exist_ok=True)

def _try_download(url: str, dest: str) -> bool:
    try:
        r = requests.get(url, timeout=30,
                         headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        if r.status_code == 200:
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return os.path.getsize(dest) > 10_000
    except Exception:
        pass
    return False

if not (os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 10_000):
    print("PDF no encontrado localmente. Intentando descarga automática...")
    downloaded = False
    for url in [PDF_URL_1, PDF_URL_2]:
        if _try_download(url, PDF_PATH):
            print(f"✓ PDF descargado desde:\\n  {url}")
            downloaded = True
            break
    if not downloaded:
        # Intento con google.colab (sólo en Colab)
        try:
            from google.colab import files
            print("\\n⚠ Descarga automática fallida.")
            print("Sube el PDF desde tu dispositivo (ventana emergente):")
            uploaded = files.upload()
            if uploaded:
                fn = list(uploaded.keys())[0]
                os.rename(fn, PDF_PATH)
                print(f"✓ PDF guardado en {PDF_PATH}")
            else:
                raise FileNotFoundError("No se subió ningún archivo.")
        except ImportError:
            raise FileNotFoundError(
                f"Coloca el PDF en: {PDF_PATH}\\n"
                f"Descárgalo de: https://www.gob.pe/institucion/pronabec/"
                f"normas-legales/7778068-033-2026-minedu-vmgi-pronabec"
            )
else:
    print(f"✓ PDF encontrado: {PDF_PATH}  ({os.path.getsize(PDF_PATH):,} bytes)")"""))

cells.append(code(
"""def clean_page_text(text: str) -> str:
    \"\"\"Limpieza ligera de texto extraído por pypdf.\"\"\"
    text = re.sub(r'[ \\t]+', ' ', text)          # colapsar espacios horizontales
    text = re.sub(r'\\n{3,}', '\\n\\n', text)       # máximo 2 saltos de línea
    text = re.sub(r' \\n', '\\n', text)             # espacio antes de \\n
    return text.strip()

reader = PdfReader(PDF_PATH)
pages_text: list[str] = []

for i, page in enumerate(reader.pages, start=1):
    raw = page.extract_text() or ""
    cleaned = clean_page_text(raw)
    if cleaned:
        pages_text.append(f"[PAGE {i}]\\n{cleaned}")

FULL_TEXT = "\\n\\n".join(pages_text)

char_count  = len(FULL_TEXT)
word_count  = len(FULL_TEXT.split())

print(f"✓ Páginas extraídas : {len(pages_text)}")
print(f"  Total caracteres  : {char_count:,}")
print(f"  Total palabras    : {word_count:,}")
print()
print("--- Primeros 600 caracteres ---")
print(FULL_TEXT[:600])"""))

# ── STEP 2: TOKENIZATION & CHUNKING ─────────────────────────────────────────
cells.append(md(
"""## Paso 2 — Tokenización y segmentación

### Justificación de los parámetros de segmentación

| Parámetro | Valor | Razón |
|---|---|---|
| `chunk_size` | **400 tokens** | Muy por debajo del límite de 8 192 tokens del modelo de embedding, permite fragmentos semánticamente coherentes (≈5-8 oraciones). |
| `chunk_overlap` | **60 tokens** | ≈15 % del tamaño del fragmento: preserva el contexto en los límites de corte sin duplicar demasiado contenido. |
| `separators` | `["\\n\\n", "\\n", ". ", " "]` | Corta primero por párrafos, luego por líneas, luego por oraciones, y como último recurso por espacios. |

Con un total de ≈ X tokens y fragmentos de 400 tokens, se generan ≈ X/340 fragmentos
(considerando el overlap). Cada fragmento cabe con holgura en la ventana de 8 192 tokens
del modelo `gemini-embedding-001`."""))

cells.append(code(
"""import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Contar tokens totales ──────────────────────────────────────────────────
enc = tiktoken.get_encoding("cl100k_base")
total_tokens = len(enc.encode(FULL_TEXT))

print(f"Total tokens (cl100k_base) : {total_tokens:,}")
print(f"Límite embedding model     : 8 192 tokens")
print(f"Fragmentos estimados       : ~{total_tokens // 340:,}")
print()

# ── Segmentar ─────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,
    separators=["\\n\\n", "\\n", ". ", " "],
    length_function=len,
)

raw_chunks: list[str] = splitter.split_text(FULL_TEXT)

# ── Adjuntar metadatos ────────────────────────────────────────────────────
CHUNK_META: list[dict] = []
for chunk in raw_chunks:
    m = re.search(r'\\[PAGE (\\d+)\\]', chunk)
    CHUNK_META.append({
        "document" : "RDE-033-2026-MINEDU-VMGI-PRONABEC",
        "topic"    : "Beca 18 y Becas Especiales – Convocatoria 2026",
        "language" : "Spanish",
        "page"     : m.group(1) if m else "N/A",
    })

avg_chars = sum(len(c) for c in raw_chunks) / len(raw_chunks)
print(f"Total de fragmentos         : {len(raw_chunks):,}")
print(f"Longitud promedio (chars)   : {avg_chars:.0f}")
print()
print("--- Fragmento 0 ---")
print(raw_chunks[0][:300])"""))

# ── STEP 3: EMBEDDINGS ───────────────────────────────────────────────────────
cells.append(md(
"""## Paso 3 — Embeddings

Modelo: **`gemini-embedding-001`** · 768 dimensiones

- `embed_documents(texts)` → usa `task_type=RETRIEVAL_DOCUMENT` (para indexación).
- `embed_query(text)` → usa `task_type=RETRIEVAL_QUERY` (para búsqueda).

Se implementa **retroceso exponencial** para manejar el límite de velocidad
del nivel gratuito (~60 req/min)."""))

cells.append(code(
"""import time, math
from google import genai
from google.genai import types

GEMINI_CLIENT    = genai.Client(api_key=GEMINI_API_KEY)
EMBEDDING_MODEL  = "gemini-embedding-001"
EMBEDDING_DIM    = 768

def _embed_batch(texts: list[str], task_type: str,
                 max_retries: int = 6) -> list[list[float]]:
    \"\"\"Incrusta una lista de textos con retroceso exponencial.\"\"\"
    for attempt in range(max_retries):
        try:
            resp = GEMINI_CLIENT.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in resp.embeddings]
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = min(60, (2 ** attempt) + 0.5 * attempt)
            print(f"  ⏳ Rate-limit / error ({exc.__class__.__name__}). "
                  f"Esperando {wait:.1f}s (intento {attempt + 1}/{max_retries})...")
            time.sleep(wait)
    return []  # unreachable

def embed_documents(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    \"\"\"Incrusta documentos para indexación (RETRIEVAL_DOCUMENT).\"\"\"
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_embeddings.extend(_embed_batch(batch, "RETRIEVAL_DOCUMENT"))
        if i + batch_size < len(texts):
            time.sleep(1.1)   # ≈ 55 req/min — dentro del límite gratuito
    return all_embeddings

def embed_query(text: str) -> list[float]:
    \"\"\"Incrusta una consulta para búsqueda (RETRIEVAL_QUERY).\"\"\"
    return _embed_batch([text], "RETRIEVAL_QUERY")[0]

# ── Prueba rápida ──────────────────────────────────────────────────────────
print("Probando embed_query...")
_test_emb = embed_query("¿Cuáles son los requisitos para postular a Beca 18?")
assert len(_test_emb) == EMBEDDING_DIM, f"Dimensión inesperada: {len(_test_emb)}"
print(f"✓ Dimensión del embedding de consulta : {len(_test_emb)}")
print(f"  Primeros 5 valores : {_test_emb[:5]}")"""))

# ── STEP 4: CHROMADB ─────────────────────────────────────────────────────────
cells.append(md(
"""## Paso 4 — Base de datos vectorial (ChromaDB)

- **Persistencia**: `chroma_db_beca18/` (excluido de git).
- **Distancia**: coseno.
- **Indexación idempotente**: si la colección ya tiene documentos, se omite el paso de incrustación."""))

cells.append(code(
"""import chromadb
from tqdm.auto import tqdm

DB_PATH         = "chroma_db_beca18"
COLLECTION_NAME = "beca18_docs"

chroma_client = chromadb.PersistentClient(path=DB_PATH)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

existing = collection.count()
print(f"Documentos en la colección : {existing}")

if existing == 0:
    print(f"Indexando {len(raw_chunks):,} fragmentos...")
    BATCH = 50
    for i in tqdm(range(0, len(raw_chunks), BATCH), desc="Indexando"):
        b_texts = raw_chunks[i : i + BATCH]
        b_meta  = CHUNK_META[i : i + BATCH]
        b_ids   = [f"chunk_{j}" for j in range(i, i + len(b_texts))]
        b_embs  = embed_documents(b_texts)
        collection.add(
            ids=b_ids,
            documents=b_texts,
            embeddings=b_embs,
            metadatas=b_meta,
        )
        if i + BATCH < len(raw_chunks):
            time.sleep(1.1)
    print(f"\\n✓ Indexación completa. Documentos almacenados: {collection.count():,}")
else:
    print(f"✓ Colección existente cargada. Documentos: {existing:,}")"""))

# ── STEP 5: SEMANTIC SEARCH ──────────────────────────────────────────────────
cells.append(md("## Paso 5 — Búsqueda semántica"))

cells.append(code(
"""def semantic_search(question: str, k: int = 5) -> list[dict]:
    \"\"\"Recupera los k fragmentos más relevantes para una pregunta.

    Returns:
        Lista de dicts con claves: ``text``, ``metadata``, ``distance``.
    \"\"\"
    q_emb = embed_query(question)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

# ── Prueba ────────────────────────────────────────────────────────────────
TEST_Q = "¿Cuáles son los requisitos de rendimiento académico para Beca 18 Ordinaria?"
print(f"Consulta: {TEST_Q}\\n")
search_results = semantic_search(TEST_Q, k=3)
for i, r in enumerate(search_results, 1):
    print(f"--- Resultado {i} | distancia: {r['distance']:.4f} | "
          f"página: {r['metadata'].get('page','N/A')} ---")
    print(r["text"][:280])
    print()"""))

# ── STEP 6: GROUNDED GENERATION ─────────────────────────────────────────────
cells.append(md(
"""## Paso 6 — Generación con contexto (respuesta fundamentada)

El prompt de sistema exige que el modelo:
1. Responda **exclusivamente** con el contexto recuperado.
2. Cite el número de página `[Página X]` cuando esté disponible.
3. Responda **"El documento no contiene información sobre este tema"** si el contexto es insuficiente."""))

cells.append(code(
"""SYSTEM_PROMPT = \"\"\"Eres un asistente experto en normativa educativa peruana, \
especializado en el programa Beca 18 y Becas Especiales del PRONABEC (Convocatoria 2026).

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE con base en los fragmentos de contexto proporcionados.
   No uses ningún conocimiento externo ni paramétrico.
2. Si la información solicitada NO está en el contexto, responde exactamente:
   "El documento no contiene información sobre este tema."
3. Cuando cites información, indica el número de página entre corchetes, por ejemplo [Página 15].
4. Si varias fuentes aportan información complementaria, consolídala de forma coherente.
5. Responde en español con un tono formal y preciso.\"\"\"

def answer_with_context(question: str, k: int = 5) -> dict:
    \"\"\"Genera una respuesta fundamentada usando fragmentos recuperados.

    Returns:
        Dict con claves ``answer`` (str) y ``sources`` (list[dict]).
    \"\"\"
    sources = semantic_search(question, k=k)

    context_parts = []
    for i, s in enumerate(sources, 1):
        page = s["metadata"].get("page", "N/A")
        context_parts.append(f"[Fragmento {i} – Página {page}]:\\n{s['text']}")
    context_block = "\\n\\n".join(context_parts)

    prompt = (
        f"Contexto del reglamento oficial:\\n\\n{context_block}\\n\\n"
        f"Pregunta del usuario: {question}\\n\\n"
        "Responde basándote ÚNICAMENTE en el contexto anterior."
    )

    resp = GEMINI_CLIENT.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
        ),
    )
    return {"answer": resp.text, "sources": sources}"""))

cells.append(code(
"""# ── 5 preguntas sobre el tema + 1 fuera de tema ───────────────────────────
TEST_QUESTIONS = [
    # 1) Requisitos de elegibilidad
    "¿Cuáles son los requisitos específicos para postular a la Beca 18 Ordinaria?",
    # 2) Modalidades de la beca
    "¿Cuáles son todas las modalidades de becas disponibles en la convocatoria 2026?",
    # 3) Beneficios económicos
    "¿Qué beneficios económicos y académicos recibe un becario adjudicado?",
    # 4) Obligaciones del estudiante
    "¿Cuáles son las principales obligaciones del becario durante sus estudios?",
    # 5) Pérdida de la beca
    "¿Cuáles son los impedimentos y condiciones para perder o quedar descalificado de la beca?",
    # 6) Fuera de tema
    "¿Cuál es la temperatura promedio anual en Lima, Perú?",
]

for q in TEST_QUESTIONS:
    print("=" * 65)
    print(f"PREGUNTA: {q}")
    print("=" * 65)
    result = answer_with_context(q, k=5)
    print("RESPUESTA:")
    print(result["answer"])
    print(f"\\n(Fuentes utilizadas: {len(result['sources'])} fragmentos)\\n")"""))

# ── STEP 7: INTERACTIVE CHAT ─────────────────────────────────────────────────
cells.append(md(
"""## Paso 7 — Interfaz de chat interactiva (ipywidgets)

- **Caja de texto** para ingresar la pregunta.
- **Botones** "Preguntar" y "Borrar".
- **Slider** para ajustar *k* (número de fragmentos recuperados).
- **Acordeón** expandible que muestra los fragmentos fuente con página y distancia."""))

cells.append(code(
"""import ipywidgets as widgets
from IPython.display import display, clear_output

# ── Widgets ───────────────────────────────────────────────────────────────
w_question = widgets.Text(
    placeholder="Escribe tu pregunta sobre Beca 18...",
    layout=widgets.Layout(width="65%"),
)
w_k = widgets.IntSlider(
    value=5, min=1, max=10, step=1,
    description="k (chunks):",
    style={"description_width": "80px"},
    layout=widgets.Layout(width="260px"),
)
w_ask   = widgets.Button(description="Preguntar",  button_style="primary", icon="search")
w_clear = widgets.Button(description="Borrar",     button_style="warning",  icon="trash")
w_out   = widgets.Output(layout=widgets.Layout(min_height="80px"))
w_acc   = widgets.Accordion(children=[], selected_index=None)

# ── Handlers ──────────────────────────────────────────────────────────────
def _on_ask(_btn):
    question = w_question.value.strip()
    if not question:
        return
    with w_out:
        clear_output(wait=True)
        print("🔍 Buscando respuesta...")
    try:
        result = answer_with_context(question, k=w_k.value)
        with w_out:
            clear_output(wait=True)
            print(f"❓ {question}\\n")
            print("📄 Respuesta:\\n")
            print(result["answer"])
        src_widgets = []
        for i, s in enumerate(result["sources"], 1):
            page = s["metadata"].get("page", "N/A")
            dist = s["distance"]
            src_widgets.append(widgets.HTML(
                f"<div style='font-family:monospace;font-size:12px;padding:8px'>"
                f"<b>Página:</b> {page} &nbsp;|&nbsp; "
                f"<b>Distancia coseno:</b> {dist:.4f}<br><br>"
                f"<pre style='white-space:pre-wrap;background:#f5f5f5;"
                f"padding:6px;border-radius:4px'>{s['text'][:600]}</pre></div>"
            ))
        w_acc.children = tuple(src_widgets)
        for i, s in enumerate(result["sources"]):
            w_acc.set_title(
                i,
                f"Fuente {i+1}  |  Pág. {s['metadata'].get('page','?')}"
                f"  |  dist={s['distance']:.4f}"
            )
        w_acc.selected_index = None
    except Exception as exc:
        with w_out:
            clear_output(wait=True)
            print(f"❌ Error: {exc}")

def _on_clear(_btn):
    w_question.value = ""
    with w_out:
        clear_output()
    w_acc.children = ()

w_ask.on_click(_on_ask)
w_clear.on_click(_on_clear)

# ── Layout ─────────────────────────────────────────────────────────────────
ui = widgets.VBox([
    widgets.HTML("<h2 style='margin-bottom:8px'>🎓 Chatbot Beca 18 – PRONABEC 2026</h2>"),
    widgets.HBox([w_question, w_ask, w_clear],
                 layout=widgets.Layout(align_items="center", gap="8px")),
    widgets.HBox([w_k]),
    widgets.HTML("<hr style='margin:8px 0'>"),
    w_out,
    widgets.HTML("<b style='font-size:13px'>📚 Fragmentos recuperados (expandir para ver):</b>"),
    w_acc,
])
display(ui)"""))

# ── NOTEBOOK OBJECT ──────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

os.makedirs("notebooks", exist_ok=True)
out_path = os.path.join("notebooks", "beca18_rag_chatbot.ipynb")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(notebook, fh, ensure_ascii=False, indent=1)

print(f"✓ Notebook generado: {out_path}")
print(f"  Celdas totales: {len(cells)}")
