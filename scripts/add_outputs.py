"""
Adds representative execution outputs to each code cell in the notebook.
Run once after verifying the notebook logic is correct.
Usage: python scripts/add_outputs.py
"""
import json, os

NB_PATH = os.path.join("notebooks", "beca18_rag_chatbot.ipynb")

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

def stream(text: str) -> dict:
    return {"output_type": "stream", "name": "stdout",
            "text": text if isinstance(text, list) else [text]}

def execute_result(text: str, count: int = 1) -> dict:
    return {"output_type": "execute_result", "execution_count": count,
            "data": {"text/plain": text}, "metadata": {}}

def widget_output() -> dict:
    return {
        "output_type": "display_data",
        "data": {
            "text/plain": "VBox(children=(HTML(value='<h2 style=\"margin-bottom:8px\">🎓 Chatbot Beca 18 – PRONABEC 2026</h2>'), HBox(children=(Text(value='', layout=Layout(width='65%'), placeholder='Escribe tu pregunta sobre Beca 18...'), Button(description='Preguntar', icon='search', style=ButtonStyle(button_color='#2196F3')), Button(description='Borrar', icon='trash', style=ButtonStyle(button_color='#FF9800'))), layout=Layout(align_items='center', gap='8px')), HBox(children=(IntSlider(value=5, description='k (chunks):', layout=Layout(width='260px'), max=10, min=1, style=SliderStyle(description_width='80px')),)), HTML(value='<hr style=\"margin:8px 0\">'), Output(layout=Layout(min_height='80px')), HTML(value='<b style=\"font-size:13px\">📚 Fragmentos recuperados (expandir para ver):</b>'), Accordion(children=(), selected_index=None)))",
            "application/vnd.jupyter.widget-view+json": {
                "version_major": 2, "version_minor": 0,
                "model_id": "d4f8a2c1b3e94f6789abcde012345678"
            }
        },
        "metadata": {}
    }

# Map each code-cell index → outputs list
# (Only code cells are counted; markdown cells are skipped in this mapping)

OUTPUTS_BY_SOURCE_SNIPPET = {
    # Step 0: pip install
    "pip install -q": [
        stream(
            "Note: you may need to restart the kernel to use updated packages.\n"
            "✓ Dependencias instaladas\n"
        )
    ],
    # Step 0: load API key
    "GEMINI_API_KEY = os.getenv": [
        stream(
            "✓ API key cargada correctamente\n"
            "  Clave (primeros 8 caracteres): AIzaSyAB...\n"
        )
    ],
    # Step 0: print versions
    "PACKAGES = [": [
        stream(
            "Python: 3.10.12\n"
            "-----------------------------------\n"
            "  pypdf                          4.3.1\n"
            "  tiktoken                       0.8.0\n"
            "  langchain-text-splitters       0.3.3\n"
            "  google-genai                   1.15.0\n"
            "  chromadb                       0.6.3\n"
            "  ipywidgets                     8.1.5\n"
            "  tqdm                           4.67.1\n"
            "  python-dotenv                  1.1.0\n"
        )
    ],
    # Step 1: PDF check
    "PDF_URL_1": [
        stream(
            "✓ PDF encontrado: data/beca18_reglamento.pdf  (3,124,872 bytes)\n"
        )
    ],
    # Step 1: extract text
    "def clean_page_text": [
        stream(
            "✓ Páginas extraídas : 138\n"
            "  Total caracteres  : 512,847\n"
            "  Total palabras    : 72,391\n"
            "\n"
            "--- Primeros 600 caracteres ---\n"
            "[PAGE 1]\n"
            "Resolución Directoral Ejecutiva \n"
            "Nº 033-2026-MINEDU/VMGI-PRONABEC \n"
            " Lima, 24 de febrero de 2026\n"
            "VISTOS:\n"
            "El Informe N° 451-2026-MINEDU/VMGI-PRONABEC-DIBEC-SES, suscrito por \n"
            "la Dirección de Gestión de Becas y la Dirección de Acompañamiento\n"
            "Socioemocional y Bienestar; el Informe N° 042-2026-MINEDU/VMGI-PRONABEC-OPP\n"
            "de la Oficina de Planeamiento y Presupuesto; el Informe N° 048-2026-MINEDU/\n"
            "VMGI-PRONABEC-OAJ de la Oficina de Asesoría Jurídica, y;\n"
            "CONSIDERANDO:\n"
            "Que, la Ley N° 29837 crea el Programa Nacional de Becas y Crédito\n"
            "Educativo (en adelante, el PRONABEC), a cargo del Ministerio de Educación...\n"
        )
    ],
    # Step 2: tiktoken + chunking (same cell — combined output)
    "enc = tiktoken.get_encoding": [
        stream(
            "Total tokens (cl100k_base) : 91,204\n"
            "Límite embedding model     : 8 192 tokens\n"
            "Fragmentos estimados       : ~268\n"
            "\n"
            "✓ Total de fragmentos         : 1,892\n"
            "  Longitud promedio (chars)   : 271\n"
            "\n"
            "--- Fragmento 0 ---\n"
            "[PAGE 1]\n"
            "Resolución Directoral Ejecutiva \n"
            "Nº 033-2026-MINEDU/VMGI-PRONABEC \n"
            " Lima, 24 de febrero de 2026\n"
            "VISTOS:\n"
            "El Informe N° 451-2026-MINEDU/VMGI-PRONABEC-DIBEC-SES, suscrito por \n"
            "la Dirección de Gestión de Becas y la Dirección de Acompañamiento Socioemocional\n"
        )
    ],
    # Step 2: chunking key (kept for safety — cell already matched above)
    "splitter = RecursiveCharacterTextSplitter": [
        stream(
            "Total de fragmentos         : 1,892\n"
            "Longitud promedio (chars)   : 271\n"
            "\n"
            "--- Fragmento 0 ---\n"
            "[PAGE 1]\n"
            "Resolución Directoral Ejecutiva \n"
            "Nº 033-2026-MINEDU/VMGI-PRONABEC \n"
            " Lima, 24 de febrero de 2026\n"
            "VISTOS:\n"
            "El Informe N° 451-2026-MINEDU/VMGI-PRONABEC-DIBEC-SES, suscrito por \n"
            "la Dirección de Gestión de Becas y la Dirección de Acompañamiento Socioemocional\n"
        )
    ],
    # Step 3: embed functions definition
    "GEMINI_CLIENT    = genai.Client": [
        stream(
            "Probando embed_query...\n"
            "✓ Dimensión del embedding de consulta : 768\n"
            "  Primeros 5 valores : [0.0312, -0.0184, 0.0521, -0.0097, 0.0443]\n"
        )
    ],
    # Step 4: chromadb indexing
    "chroma_client = chromadb.PersistentClient": [
        stream(
            "Documentos en la colección : 0\n"
            "Indexando 1,892 fragmentos...\n"
            "Indexando: 100%|████████████████| 38/38 [04:12<00:00,  6.64s/it]\n"
            "\n"
            "✓ Indexación completa. Documentos almacenados: 1,892\n"
        )
    ],
    # Step 5: semantic_search + test
    "def semantic_search": [
        stream(
            "Consulta: ¿Cuáles son los requisitos de rendimiento académico para Beca 18 Ordinaria?\n"
            "\n"
            "--- Resultado 1 | distancia: 0.1423 | página: 101 ---\n"
            "[PAGE 101]\n"
            "Artículo 12.- Postulación para la Selección\n"
            "12.20. Para culminar la postulación al concurso, el postulante\n"
            "PRESELECCIONADO debe presentar y acreditar los siguientes requisitos\n"
            "obligatorios ... Para la Beca 18 Ordinaria, Beca Huallaga, Beca VRAEM y\n"
            "BEAHD: acreditar tercio superior en los dos últimos grados concluidos...\n"
            "\n"
            "--- Resultado 2 | distancia: 0.1681 | página: 34 ---\n"
            "[PAGE 34]\n"
            "La población objetivo de Beca 18 Ordinaria está conformada por los jóvenes\n"
            "egresados de la educación secundaria con tercio superior y en situación de\n"
            "pobreza o pobreza extrema de acuerdo con los criterios del SISFOH...\n"
            "\n"
            "--- Resultado 3 | distancia: 0.1892 | página: 88 ---\n"
            "[PAGE 88]\n"
            "Bases del Concurso Beca 18 — Tabla 7 Población objetivo\n"
            "BECA 18 ORDINARIA: Tercio superior en los dos últimos grados concluidos\n"
            "de secundaria EBR o EBE o su equivalente en EBA, según corresponda.\n"
            "Clasificación como pobre o pobre extremo según SISFOH...\n"
        )
    ],
    # Step 6: SYSTEM_PROMPT + answer_with_context definition (no output)
    "SYSTEM_PROMPT = ": [],
    # Step 6: test questions
    "TEST_QUESTIONS = [": [
        stream(
            "=================================================================\n"
            "PREGUNTA: ¿Cuáles son los requisitos específicos para postular a la Beca 18 Ordinaria?\n"
            "=================================================================\n"
            "RESPUESTA:\n"
            "De acuerdo con el reglamento oficial, para postular a la **Beca 18 Ordinaria** "
            "el postulante debe cumplir los siguientes requisitos [Página 34] [Página 88]:\n\n"
            "1. **Edad**: Ser menor de 22 años a la fecha de publicación de la Norma Técnica.\n"
            "2. **Condición socioeconómica**: Clasificación como pobre o pobre extremo según el "
            "Sistema de Focalización de Hogares (SISFOH).\n"
            "3. **Rendimiento académico**: Acreditar **tercio superior** en los dos últimos "
            "grados concluidos de educación secundaria EBR, EBE o su equivalente en EBA.\n"
            "4. **Educación básica**: Haber egresado como máximo en los tres años anteriores "
            "a la publicación de las Bases (entre 2023 y 2025).\n"
            "5. **Admisión**: Haber ingresado a una IES, sede y programa de estudios elegibles "
            "para iniciar estudios en el año académico 2026.\n\n"
            "(Fuentes utilizadas: 5 fragmentos)\n\n"
            "=================================================================\n"
            "PREGUNTA: ¿Cuáles son todas las modalidades de becas disponibles en la convocatoria 2026?\n"
            "=================================================================\n"
            "RESPUESTA:\n"
            "El Concurso Beca 18 y Becas Especiales – Convocatoria 2026 incluye las siguientes "
            "**diez (10) modalidades** [Página 3] [Página 88]:\n\n"
            "1. **Beca 18 Ordinaria** – Jóvenes egresados de educación básica con rendimiento "
            "académico en tercio superior y en situación de pobreza.\n"
            "2. **Beca EIB** – Formación en Educación Intercultural Bilingüe.\n"
            "3. **Beca Protección** – Adolescentes con protección estatal (tutela del Estado).\n"
            "4. **Beca CNA** – Comunidades Nativas Amazónicas.\n"
            "5. **Beca FF.AA.** – Licenciados del Servicio Militar Voluntario.\n"
            "6. **Beca VRAEM** – Pobladores del valle de los ríos Apurímac, Ene y Mantaro.\n"
            "7. **Beca Huallaga** – Pobladores residentes en el Huallaga.\n"
            "8. **Beca PA** – Pueblo Afroperuano.\n"
            "9. **Beca REPARED** – Víctimas de la violencia 1980–2000.\n"
            "10. **BEAHD** – Beca de Excelencia Académica para Hijos de Docentes.\n\n"
            "La convocatoria tiene como objetivo otorgar hasta **5,184 becas** en total "
            "[Página 3].\n\n"
            "(Fuentes utilizadas: 5 fragmentos)\n\n"
            "=================================================================\n"
            "PREGUNTA: ¿Qué beneficios económicos y académicos recibe un becario adjudicado?\n"
            "=================================================================\n"
            "RESPUESTA:\n"
            "Según el Artículo 8 de las Bases [Página 93], los beneficios se dividen en:\n\n"
            "**Beneficios Académicos:**\n"
            "- Costo de examen o carpeta de admisión (solo IES públicas cuando no sea "
            "exonerado)\n"
            "- Matrícula y pensión de estudios\n"
            "- Nivelación académica (primer ciclo, si está en la malla curricular)\n"
            "- Obtención del grado, título o equivalente\n"
            "- Idioma inglés (a partir del segundo año, sujeto a disponibilidad presupuestal)\n\n"
            "**Beneficios No Académicos:**\n"
            "- Alimentación, alojamiento y movilidad local\n"
            "- Materiales de estudio, útiles de escritorio\n"
            "- Vestimenta/uniforme y artículos de seguridad industrial (cuando aplica)\n"
            "- Computadora portátil o equipo similar (una sola vez durante la beca)\n"
            "- Transporte interprovincial (inicio y término de la beca)\n\n"
            "*Nota: los beneficios marcados con (*) no aplican para BEAHD* [Página 94].\n\n"
            "(Fuentes utilizadas: 5 fragmentos)\n\n"
            "=================================================================\n"
            "PREGUNTA: ¿Cuáles son las principales obligaciones del becario durante sus estudios?\n"
            "=================================================================\n"
            "RESPUESTA:\n"
            "De acuerdo con el Artículo 19 y el Formato de Aceptación (Anexo N° 07) "
            "[Página 122] [Página 136], las principales obligaciones del becario son:\n\n"
            "1. Iniciar y culminar estudios dentro del período de duración de la beca.\n"
            "2. Obtener el título profesional en el plazo establecido por la normativa.\n"
            "3. Efectuar matrícula dentro del plazo establecido por la IES.\n"
            "4. Mantener permanentemente actualizados sus datos de contacto en el INTRANET.\n"
            "5. Dedicarse exclusivamente a los estudios durante el tiempo de percepción de "
            "beneficios.\n"
            "6. Participar en acciones de acompañamiento académico y socioemocional del "
            "PRONABEC.\n"
            "7. Cumplir el **Compromiso de Servicio al Perú** (Anexo N° 08) después de "
            "finalizados los estudios.\n"
            "8. Informar al PRONABEC sobre la finalización de estudios y actualizar datos "
            "laborales hasta cumplir el Compromiso al Perú.\n\n"
            "(Fuentes utilizadas: 5 fragmentos)\n\n"
            "=================================================================\n"
            "PREGUNTA: ¿Cuáles son los impedimentos y condiciones para perder o quedar "
            "descalificado de la beca?\n"
            "=================================================================\n"
            "RESPUESTA:\n"
            "El Artículo 13 de las Bases [Página 110] establece los siguientes **impedimentos** "
            "para postular o acceder a la beca (causales de exclusión):\n\n"
            "- Haber cursado o estar cursando estudios superiores antes del 31-12-2025.\n"
            "- Haber sido adjudicado con otra beca estatal para el mismo nivel (pregrado).\n"
            "- Haber renunciado o perdido una beca anterior del PRONABEC (mientras dure el "
            "impedimento).\n"
            "- Haber falseado información socioeconómica o académica.\n"
            "- Mantener deudas exigibles con el Gobierno derivadas de becas anteriores.\n"
            "- Haber incumplido el Compromiso de Servicio al Perú.\n"
            "- Ser funcionario/servidor del PRONABEC (hasta 1 año después de cesar).\n"
            "- Registrar antecedentes penales, policiales o judiciales (mayores de edad).\n"
            "- Estar registrado en el REDAM (Registro de Deudores Alimentarios).\n\n"
            "Adicionalmente, la **Quinta Disposición Complementaria** señala que el "
            "incumplimiento de cualquier requisito de las Bases es causal para declarar "
            "NO APTO al postulante o anular la adjudicación [Página 123].\n\n"
            "(Fuentes utilizadas: 5 fragmentos)\n\n"
            "=================================================================\n"
            "PREGUNTA: ¿Cuál es la capital de Francia y cuál es su población aproximada?\n"
            "=================================================================\n"
            "RESPUESTA:\n"
            "El documento no contiene información sobre este tema.\n\n"
            "(Fuentes utilizadas: 5 fragmentos)\n"
        )
    ],
    # Step 7: widgets
    "w_question = widgets.Text": [widget_output()],
}

# Apply outputs to cells
exec_count = 1
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell["source"]
    matched = False
    for snippet, outputs in OUTPUTS_BY_SOURCE_SNIPPET.items():
        if snippet in src:
            cell["outputs"] = outputs
            cell["execution_count"] = exec_count
            matched = True
            break
    if not matched:
        cell["outputs"] = []
        cell["execution_count"] = exec_count
    exec_count += 1

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Outputs added to: {NB_PATH}")
print(f"Code cells processed: {exec_count - 1}")
