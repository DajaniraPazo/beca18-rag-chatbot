# Beca 18 RAG Chatbot — Normativa PRONABEC 2026

## Proposito del proyecto

Sistema de **Generacion Aumentada por Recuperacion (RAG)** que responde preguntas
sobre la normativa oficial del programa Beca 18 (Resolucion Directoral Ejecutiva
N.° 033-2026-MINEDU/VMGI-PRONABEC) recuperando fragmentos relevantes del PDF
fuente e introduciendolos como contexto en un modelo de lenguaje Gemini.
El sistema responde **exclusivamente** desde el documento — no usa conocimiento
parametrico del modelo — y rechaza preguntas ajenas al reglamento.

## Descripcion del documento fuente

| Campo | Detalle |
|---|---|
| Titulo | RDE N.° 033-2026-MINEDU/VMGI-PRONABEC |
| Contenido | Bases del Concurso Beca 18 y Becas Especiales Convocatoria 2026 |
| Institucion | PRONABEC – Ministerio de Educacion del Peru |
| URL | https://www.gob.pe/institucion/pronabec/normas-legales/7778068-033-2026-minedu-vmgi-pronabec |

---

## Pipeline RAG (resumen)

El PDF es procesado página a página con `pypdf` (marcadores `[PAGE N]`) → el texto
es tokenizado con `tiktoken` y dividido en fragmentos de 400 tokens / 60 de overlap
con `RecursiveCharacterTextSplitter` → cada fragmento es incrustado con
`gemini-embedding-001` (768 dim, tarea `RETRIEVAL_DOCUMENT`) y almacenado en
`ChromaDB` (distancia coseno, indexación idempotente) → ante cada pregunta del
usuario, la consulta se incrusta (`RETRIEVAL_QUERY`) y se recuperan los *k*
fragmentos más cercanos → éstos se inyectan como contexto en `gemini-2.5-flash`
junto a un prompt de sistema estricto que exige citar páginas y rechazar preguntas
fuera del documento → la respuesta y las fuentes se presentan en una interfaz
`ipywidgets` interactiva.

---

## Requisitos

- Python 3.10+
- Clave API de Google Gemini (gratuita): https://aistudio.google.com/app/apikey

---

## Instalación y configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/<tu-usuario>/beca18-rag-chatbot.git
cd beca18-rag-chatbot
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar la clave API

```bash
cp .env.example .env
# Edita .env y reemplaza your_gemini_api_key_here con tu clave real
```

El archivo `.env` NUNCA debe ser versionado (ya está en `.gitignore`).

### 4. Obtener el PDF

El PDF **no se incluye** en el repositorio por su tamaño.
Descárgalo manualmente desde:

```
https://www.gob.pe/institucion/pronabec/normas-legales/7778068-033-2026-minedu-vmgi-pronabec
```

y colócalo en `data/beca18_reglamento.pdf`.

> El notebook también intenta descargarlo automáticamente en la celda del Paso 1.

---

## Ejecutar el notebook

### Localmente (Jupyter Lab / Notebook)

```bash
jupyter lab notebooks/beca18_rag_chatbot.ipynb
```

Ejecuta las celdas de arriba a abajo (`Kernel → Restart & Run All`).

### Google Colab

1. Sube el notebook a [colab.research.google.com](https://colab.research.google.com).
2. En la primera celda, crea el archivo `.env`:
   ```python
   %%writefile .env
   GEMINI_API_KEY=tu_clave_aqui
   ```
3. Ejecuta todas las celdas en orden.

---

## Usar la interfaz de chat

La celda del **Paso 7** muestra:

| Elemento | Funcion |
|---|---|
| Caja de texto | Escribe tu pregunta sobre Beca 18 |
| Boton "Preguntar" | Lanza la busqueda y generacion |
| Boton "Borrar" | Limpia la pregunta y la respuesta |
| Slider *k* | Ajusta cuantos fragmentos se recuperan (1-10) |
| Acordeon "Fuentes" | Expande para ver texto, pagina y distancia de cada fuente |

Ejemplos de preguntas:
- Cuales son los requisitos para postular a Beca 18 Ordinaria?
- Que modalidades de becas existen en la convocatoria 2026?
- Cuales son los beneficios academicos del becario?
- Que ocurre si el becario desaprueba un curso?

---

## Estructura del repositorio

```
beca18-rag-chatbot/
├── data/
│   ├── .gitkeep                    # Instrucciones para descargar el PDF
│   └── beca18_reglamento.pdf       # No versionado (data/*.pdf en .gitignore)
├── notebooks/
│   └── beca18_rag_chatbot.ipynb    # Notebook principal (Pasos 0-7)
├── scripts/
│   └── create_notebook.py          # Script auxiliar que genero el notebook
├── video/
│   └── link.txt                    # Enlace al video explicativo
├── .env.example                    # Plantilla para GEMINI_API_KEY
├── .gitignore                      # Excluye .env, chroma_db_*/, data/*.pdf
├── requirements.txt                # Dependencias con versiones fijadas
└── README.md
```

---

## Seguridad

- La clave API solo se carga desde `.env` via `python-dotenv`.
- `.env` y `chroma_db_*/` estan excluidos del repositorio.
- `data/*.pdf` tambien esta en `.gitignore`.
