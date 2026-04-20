import streamlit as st
import chromadb
from openai import OpenAI
import os
import io
from datetime import datetime
from PyPDF2 import PdfReader # Nueva dependencia para parseo de PDF

# 1. Inicialización de Parámetros de Interfaz
st.set_page_config(
    page_title="Adecco | Asistente Logístico",
    page_icon="🔴",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Inyección de CSS para estandarización gráfica corporativa (Dark Theme)
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #121212 !important; }
    [data-testid="stSidebar"] [data-testid="stImage"] { display: flex; justify-content: center; margin-bottom: 20px; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p { color: #F8F9FA !important; }
    .stButton>button {
        width: 100%; border-radius: 6px; background-color: transparent !important;
        border: 1px solid #E3000F !important; color: #F8F9FA !important; font-weight: bold; transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover { background-color: #E3000F !important; color: #FFFFFF !important; box-shadow: 0 4px 12px rgba(227, 0, 15, 0.3); }
    .stChatMessage.user { background-color: #1E1E1E !important; border: 1px solid #333333 !important; }
    .stChatMessage.assistant { background-color: #2D1414 !important; border: 1px solid #5C2B2B !important; }
    </style>
""", unsafe_allow_html=True)

# 2. Configuración Singleton de Espacio Vectorial
@st.cache_resource
def inicializar_chromadb():
    return chromadb.PersistentClient(path="./chroma_db")

chroma_client = inicializar_chromadb()

# 3. Instanciación del Motor de Inferencia (Groq)
client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"], # Utilice su credencial gsk_...
    base_url="https://api.groq.com/openai/v1",
)

try:
    col_capacitacion = chroma_client.get_or_create_collection(name="capacitacion_interna_v1")
except Exception as e:
    st.error(f"Excepción en clúster vectorial: {e}")

# 4. Estructuración del Panel Lateral (Sidebar)
with st.sidebar:
    ruta_logo = "adecco_logo.png" 
    if os.path.exists(ruta_logo):
        st.image(ruta_logo, width=200) 
    else:
        st.warning(f"Error I/O: No se detectó el archivo en la ruta relativa '{ruta_logo}'")
    
    st.markdown("## 🛡️ Módulo Logístico PDV")
    st.caption("Motor de Inferencia: **Llama 3.3 70B**")
    st.divider()

    # --- MÓDULO 1: INGESTIÓN DINÁMICA DE CONOCIMIENTO (Actualizado para PDF) ---
    with st.expander("📥 Administración de Conocimiento"):
        st.caption("Carga de normativas en formato .txt o .pdf al clúster vectorial.")
        # Se añade 'pdf' a los tipos permitidos
        archivo_cargado = st.file_uploader("Subir archivo", type=['txt', 'pdf'], label_visibility="collapsed")
        
        if archivo_cargado:
            texto_nuevo = ""
            # Lógica de extracción condicional
            if archivo_cargado.name.endswith('.txt'):
                texto_nuevo = archivo_cargado.read().decode("utf-8")
            elif archivo_cargado.name.endswith('.pdf'):
                lector_pdf = PdfReader(archivo_cargado)
                for pagina in lector_pdf.pages:
                    texto_extraido = pagina.extract_text()
                    if texto_extraido:
                        texto_nuevo += texto_extraido + "\n"

            if st.button("Actualizar Base Vectorial", key="btn_ingest"):
                if texto_nuevo.strip():
                    doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    col_capacitacion.add(
                        documents=[texto_nuevo],
                        ids=[doc_id]
                    )
                    st.success(f"Vector {doc_id} indexado con éxito.")
                else:
                    st.error("Error: No se pudo extraer texto del documento.")
    # ---------------------------------------------------------------------------
    
    st.markdown("### 📚 Guia de Consulta")
    if st.button("📦 Procedimiento DOA", use_container_width=True):
        st.session_state.prompt_sugerido = "Detalla el procedimiento exacto para registrar y gestionar un equipo DOA en el SISCAD."
    if st.button("📋 Conciliación de Inventario", use_container_width=True):
        st.session_state.prompt_sugerido = "¿Cuáles son los pasos y requerimientos para la conciliación diaria del PDV?"
    if st.button("⚠️ Rechazo de Abastecimiento", use_container_width=True):
        st.session_state.prompt_sugerido = "Enumera los motivos válidos para rechazar mercadería y el proceso de reporte."
    
    st.divider()

    # --- MÓDULO 2: EXPORTACIÓN DE TRAZABILIDAD ---
    if "mensajes" in st.session_state and len(st.session_state.mensajes) > 1:
        st.markdown("### 📄 Auditoría y Logs")
        
        log_stream = io.StringIO()
        log_stream.write(f"REPORTE DE CONSULTA LOGÍSTICA - ADECCO\n")
        log_stream.write(f"FECHA DE EXTRACCIÓN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_stream.write("="*50 + "\n\n")
        
        for m in st.session_state.mensajes:
            rol = "OPERADOR" if m['role'] == 'user' else "SISTEMA EXPERTO"
            log_stream.write(f"[{rol}]:\n{m['content']}\n\n")
            log_stream.write("-" * 50 + "\n\n")
        
        st.download_button(
            label="⬇️ Descargar Historial (TXT)",
            data=log_stream.getvalue(),
            file_name=f"auditoria_adecco_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        st.divider()
    # ---------------------------------------------

    if st.button("🔄 Purgar Memoria", type="primary", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()

# 5. Inicialización de Memoria Latente (Session State)
st.title("Asistente Logístico Inteligente")
st.header("🔬 Panel de Interacción")
st.divider()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = [
        {"role": "assistant", "content": "Sistema en línea. Ingrese su consulta sobre protocolos logísticos, SISCAD o normativas de PDV."}
    ]

# Renderizado del historial de transacciones
for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Captura y Procesamiento de Entrada (Input Pipeline)
query_usuario = st.chat_input("Ejecutar consulta...")
if "prompt_sugerido" in st.session_state and st.session_state.prompt_sugerido:
    query_usuario = st.session_state.prompt_sugerido
    st.session_state.prompt_sugerido = None

if query_usuario:
    st.session_state.mensajes.append({"role": "user", "content": query_usuario})
    with st.chat_message("user"):
        st.markdown(query_usuario)

    # Inferencia y recuperación
    with st.chat_message("assistant"):
        with st.spinner("Ejecutando búsqueda semántica..."):
            
            resultados_rag = col_capacitacion.query(
                query_texts=[query_usuario],
                n_results=3
            )

            contexto_recuperado = ""
            if resultados_rag and resultados_rag['documents']:
                documentos = resultados_rag['documents'][0]
                contexto_recuperado = "\n\n---\n\n".join(documentos)

            if contexto_recuperado:
                with st.expander("Ver Fragmentos de Fuente Recuperados (Espacio Vectorial)"):
                    st.markdown(f"> {contexto_recuperado}")

            prompt_sintesis = f"""
            Operas como un sistema experto en auditoría y logística para Adecco-Claro.
            Analiza el Contexto Recuperado y resuelve la consulta.

            Reglas de Inferencia Estricta:
            1. EXTRACCIÓN DIRECTA: Si el protocolo se encuentra en el contexto, descríbelo con precisión algorítmica. Utiliza formato Markdown (negritas, listas) para estructurar los datos.
            2. DEDUCCIÓN RESTRINGIDA: Si no existe coincidencia exacta pero hay normativas análogas, realiza la deducción lógica advirtiendo explícitamente: "El manual no especifica esta variable, sin embargo, aplicando los principios de [Normativa], el procedimiento es...".
            3. NEGACIÓN DETERMINISTA: Si el contexto carece de entropía útil para la respuesta, declara falta de datos. Cero alucinaciones.

            Contexto Recuperado:
            {contexto_recuperado}

            Consulta del Usuario: {query_usuario}
            """

            try:
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "Eres un controlador logístico corporativo, estricto, técnico y conciso. Presenta la información de forma estructurada."},
                        {"role": "user", "content": prompt_sintesis}
                    ],
                    temperature=0.0,
                    stream=True 
                )
                
                st.markdown("### 📝 Respuesta Estructurada")
                respuesta_completa = st.write_stream(
                    (chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta.content is not None)
                )
                
                st.session_state.mensajes.append({"role": "assistant", "content": respuesta_completa})
                
            except Exception as e:
                st.error(f"Fallo de arquitectura LLM: {e}")
