import streamlit as st
import chromadb
from openai import OpenAI
import os

# 0. Extracción Segura de Credenciales (Previene hardcoding)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Error Crítico: Variable de entorno 'GROQ_API_KEY' no configurada en el sistema local.")
    st.stop()

# --- PARCHE DE CONCURRENCIA PARA CHROMADB ---
@st.cache_resource
def inicializar_chromadb():
    """Instancia el cliente de base de datos de forma persistente en memoria (Singleton)"""
    return chromadb.PersistentClient(path="./chroma_db")

chroma_client = inicializar_chromadb()
# --------------------------------------------

# 1. Configuración del Cliente de Inferencia (Groq)
client = OpenAI(
    api_key=api_key, 
    base_url="https://api.groq.com/openai/v1",
)

# Conexión exclusiva al espacio vectorial de Adecco
try:
    col_capacitacion = chroma_client.get_or_create_collection(name="capacitacion_interna_v1")
except Exception as e:
    st.error(f"Fallo en la resolución de la colección vectorial: {e}")

# 2. Interfaz de Usuario
st.title("Asistente de Capacitación Adecco")
st.markdown("---")

query_usuario = st.text_input("Ingrese su consulta sobre los procesos de capacitación:")

if query_usuario:
    with st.spinner("Buscando en la base de conocimiento interna..."):
        
        # A. Recuperación de Información (RAG Retrieval) directa
        resultados_rag = col_capacitacion.query(
            query_texts=[query_usuario],
            n_results=3
        )

        contexto_recuperado = ""
        if resultados_rag and resultados_rag['documents']:
            documentos = resultados_rag['documents'][0]
            contexto_recuperado = "\n\n---\n\n".join(documentos)
        else:
            st.warning("El espacio latente no devolvió vectores cercanos a la consulta.")

        # B. Síntesis y Generación de Respuesta
        prompt_sintesis = f"""
        Eres un asistente corporativo experto de Adecco. Responde a la consulta basándote ÚNICAMENTE en el siguiente contexto extraído de los manuales de capacitación de la empresa.
        Si la respuesta no se encuentra en el contexto, declara explícitamente que no posees la información en tus manuales. No alucines procesos externos.

        Contexto Recuperado:
        {contexto_recuperado}

        Consulta del Usuario: {query_usuario}
        """

        try:
            # Inferencia de alta velocidad con Llama 3
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Eres un instructor corporativo riguroso, claro y preciso."},
                    {"role": "user", "content": prompt_sintesis}
                ],
                temperature=0.1 
            )
            
            st.markdown("### Respuesta Oficial")
            st.write(completion.choices[0].message.content)
            
        except Exception as e:
            st.error(f"Excepción en la generación LLM: {e}")