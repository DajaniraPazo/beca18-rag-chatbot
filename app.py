"""
Beca 18 RAG Chatbot — Streamlit web app
Grounded on RDE N.° 033-2026-MINEDU/VMGI-PRONABEC
"""
import os
import re
import time
import requests
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Chatbot Beca 18 – PRONABEC 2026",
    page_icon="🎓",
    layout="wide",
)

# ── API Key ───────────────────────────────────────────────────────────────────
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    st.error(
        "**GEMINI_API_KEY no configurada.**\n\n"
        "En Streamlit Cloud: Settings → Secrets → agrega `GEMINI_API_KEY = \"tu_clave\"`\n\n"
        "Obtén tu clave gratis en https://aistudio.google.com/app/apikey"
    )
    st.stop()

# ── PDF download & cache ──────────────────────────────────────────────────────
PDF_PATH = Path("data/beca18_reglamento.pdf")
PDF_URLS = [
    "https://cdn.www.gob.pe/uploads/document/file/7778068/"
    "7778068-rde-n-033-2026-minedu-vmgi-pronabec.pdf",
    "https://www.gob.pe/institucion/pronabec/normas-legales/"
    "7778068-033-2026-minedu-vmgi-pronabec",
]

@st.cache_resource(show_spinner="📥 Descargando el reglamento PDF…")
def get_pdf_path() -> Path:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 10_000:
        return PDF_PATH
    for url in PDF_URLS:
        try:
            r = requests.get(
                url, timeout=60,
                headers={"User-Agent": "Mozilla/5.0"},
                stream=True,
            )
            if r.status_code == 200:
                with open(PDF_PATH, "wb") as fh:
                    for chunk in r.iter_content(8192):
                        fh.write(chunk)
                if PDF_PATH.stat().st_size > 10_000:
                    return PDF_PATH
        except Exception:
            continue
    raise FileNotFoundError(
        "No se pudo descargar el PDF automáticamente.\n"
        "Coloca beca18_reglamento.pdf en data/ y reinicia la app."
    )

# ── Text extraction ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📄 Extrayendo texto del PDF…")
def extract_full_text() -> str:
    from pypdf import PdfReader
    path = get_pdf_path()
    pages: list[str] = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
        if raw:
            pages.append(f"[PAGE {i}]\n{raw}")
    return "\n\n".join(pages)

# ── Chunking ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="✂️ Dividiendo en fragmentos…")
def get_chunks() -> tuple[list[str], list[dict]]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    full_text = extract_full_text()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )
    chunks = splitter.split_text(full_text)
    metas = []
    for chunk in chunks:
        m = re.search(r"\[PAGE (\d+)\]", chunk)
        metas.append({
            "document": "RDE-033-2026-MINEDU-VMGI-PRONABEC",
            "topic": "Beca 18 y Becas Especiales Convocatoria 2026",
            "language": "Spanish",
            "page": m.group(1) if m else "N/A",
        })
    return chunks, metas

# ── Gemini client ─────────────────────────────────────────────────────────────
from google import genai
from google.genai import types

_gemini = genai.Client(api_key=GEMINI_API_KEY)
EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL   = "gemini-2.5-flash"

def _embed_batch(texts: list[str], task_type: str, max_retries: int = 6) -> list[list[float]]:
    for attempt in range(max_retries):
        try:
            resp = _gemini.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in resp.embeddings]
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(60, 2 ** attempt))
    return []

def embed_documents(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        out.extend(_embed_batch(texts[i : i + batch_size], "RETRIEVAL_DOCUMENT"))
        if i + batch_size < len(texts):
            time.sleep(1.1)
    return out

def embed_query(text: str) -> list[float]:
    return _embed_batch([text], "RETRIEVAL_QUERY")[0]

# ── ChromaDB ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🗄️ Construyendo índice vectorial (primera vez: ~5 min)…")
def get_collection():
    import chromadb

    chunks, metas = get_chunks()
    client = chromadb.PersistentClient(path="chroma_db_beca18")
    col = client.get_or_create_collection(
        name="beca18_docs",
        metadata={"hnsw:space": "cosine"},
    )
    if col.count() == 0:
        BATCH = 50
        progress = st.progress(0, text="Indexando fragmentos…")
        total = len(range(0, len(chunks), BATCH))
        for step, i in enumerate(range(0, len(chunks), BATCH)):
            batch = chunks[i : i + BATCH]
            col.add(
                ids=[f"chunk_{j}" for j in range(i, i + len(batch))],
                documents=batch,
                embeddings=embed_documents(batch),
                metadatas=metas[i : i + BATCH],
            )
            progress.progress((step + 1) / total,
                              text=f"Indexando… {i + len(batch)}/{len(chunks)}")
            if i + BATCH < len(chunks):
                time.sleep(1.1)
        progress.empty()
    return col

# ── RAG core ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Eres un asistente experto en normativa educativa peruana, especializado en "
    "el programa Beca 18 y Becas Especiales del PRONABEC (Convocatoria 2026).\n\n"
    "REGLAS ESTRICTAS:\n"
    "1. Responde ÚNICAMENTE con base en el contexto proporcionado. "
    "No uses conocimiento externo.\n"
    "2. Si la información NO está en el contexto, responde exactamente: "
    '"El documento no contiene información sobre este tema."\n'
    "3. Cita el número de página entre corchetes [Página X].\n"
    "4. Responde en español con tono formal y preciso."
)

def answer(question: str, k: int = 5) -> tuple[str, list[dict]]:
    col = get_collection()
    q_emb = embed_query(question)
    results = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    sources = [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    context = "\n\n".join(
        f"[Fragmento {i+1} – Página {s['metadata'].get('page', 'N/A')}]:\n{s['text']}"
        for i, s in enumerate(sources)
    )
    prompt = (
        f"Contexto del reglamento oficial:\n\n{context}\n\n"
        f"Pregunta: {question}\n\n"
        "Responde basándote ÚNICAMENTE en el contexto anterior."
    )
    resp = _gemini.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
        ),
    )
    return resp.text, sources

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎓 Chatbot Beca 18 — Normativa PRONABEC 2026")
st.caption(
    "Respuestas basadas exclusivamente en la "
    "**RDE N.° 033-2026-MINEDU/VMGI-PRONABEC**. "
    "El modelo no usa conocimiento externo al documento."
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    k_val = st.slider("Fragmentos a recuperar (k)", min_value=1, max_value=10, value=5)
    st.divider()
    st.markdown("**💡 Preguntas de ejemplo**")
    EXAMPLES = [
        "¿Cuáles son los requisitos para postular a Beca 18 Ordinaria?",
        "¿Qué modalidades de becas existen en la convocatoria 2026?",
        "¿Qué beneficios académicos y económicos recibe el becario?",
        "¿Cuáles son las obligaciones del becario durante sus estudios?",
        "¿Cuándo se pierde o se queda descalificado de la beca?",
        "¿Cuántas becas se ofrecen en total y cómo se distribuyen?",
    ]
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True, key=f"btn_{ex[:20]}"):
            st.session_state["queued_question"] = ex

    st.divider()
    st.markdown(
        "📄 [Reglamento oficial]"
        "(https://www.gob.pe/institucion/pronabec/normas-legales/"
        "7778068-033-2026-minedu-vmgi-pronabec)"
    )
    st.markdown(
        "💻 [Código fuente]"
        "(https://github.com/DajaniraPazo/beca18-rag-chatbot)"
    )

# Boot: trigger index build once
with st.spinner("Iniciando sistema RAG…"):
    try:
        get_collection()
        st.session_state["ready"] = True
    except Exception as exc:
        st.error(f"Error al inicializar: {exc}")
        st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} fragmentos fuente"):
                for i, s in enumerate(msg["sources"], 1):
                    pg   = s["metadata"].get("page", "?")
                    dist = s["distance"]
                    st.markdown(f"**Fuente {i}** | Página {pg} | distancia: {dist:.4f}")
                    st.code(s["text"][:450], language=None)

# Resolve input: sidebar button or chat input
user_input = (
    st.session_state.pop("queued_question", None)
    or st.chat_input("Escribe tu pregunta sobre Beca 18…")
)

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Buscando en el reglamento…"):
            try:
                response_text, sources = answer(user_input, k=k_val)
            except Exception as exc:
                response_text = f"❌ Error al generar respuesta: {exc}"
                sources = []
        st.markdown(response_text)
        if sources:
            with st.expander(f"📚 {len(sources)} fragmentos fuente"):
                for i, s in enumerate(sources, 1):
                    pg   = s["metadata"].get("page", "?")
                    dist = s["distance"]
                    st.markdown(f"**Fuente {i}** | Página {pg} | distancia: {dist:.4f}")
                    st.code(s["text"][:450], language=None)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources,
    })
