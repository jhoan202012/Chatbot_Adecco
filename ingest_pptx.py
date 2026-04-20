import os
import uuid
import chromadb
import time
import base64
from io import BytesIO
from pdf2image import convert_from_path
from PIL import Image
from openai import OpenAI

# 0. Extracción Segura de Credenciales
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Excepción de Entorno: Variable 'GROQ_API_KEY' no detectada. Abortando ejecución.")

# 1. Configuración de API de Inferencia (Groq)
client = OpenAI(
    api_key=api_key, 
    base_url="https://api.groq.com/openai/v1",
)

def codificar_imagen_base64(img: Image) -> str:
    """Convierte un objeto PIL Image a una cadena base64 para consumo de la API REST."""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def procesar_diapositivas_restantes(pdf_path: str, slide_inicio: int) -> list[dict]:
    """
    Rasteriza el PDF y extrae semántica utilizando modelo multimodal.
    Inicia la iteración desde slide_inicio para evitar duplicidad vectorial.
    """
    print(f"Iniciando rasterización de: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"El archivo {pdf_path} no existe.")

    # Nota arquitectónica: Ruta absoluta válida únicamente para ejecución local.
    imagenes = convert_from_path(pdf_path, poppler_path=r"D:\Chatbot_Adecco\poppler\Library\bin")
    documentos_procesados = []

    prompt_extraccion = (
        "Analiza esta diapositiva técnica de capacitación corporativa. "
        "1. Si contiene un diagrama de flujo, describe secuencialmente los nodos, decisiones y actores. "
        "2. Si es una captura de pantalla, describe la interfaz y la acción ilustrada. "
        "3. Si es texto, transcríbelo estructuradamente. "
        "Entrega únicamente la información técnica, sin preámbulos."
    )

    for i in range(slide_inicio - 1, len(imagenes)):
        slide_num = i + 1
        print(f"Procesando inferencia para diapositiva {slide_num}/{len(imagenes)}...")
        
        img = imagenes[i]
        img_base64 = codificar_imagen_base64(img)
        
        exito = False
        intentos = 0
        
        while not exito and intentos < 3:
            try:
                respuesta = client.chat.completions.create(
                    model='meta-llama/llama-4-scout-17b-16e-instruct',
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_extraccion},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.0
                )
                
                metadata = {
                    "source": os.path.basename(pdf_path),
                    "slide_number": slide_num,
                    "content_type": "multimodal_extraction"
                }
                
                documentos_procesados.append({
                    "page_content": respuesta.choices[0].message.content,
                    "metadata": metadata
                })
                
                exito = True
                time.sleep(5) 
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    print(f"  [!] Cuota de ráfaga excedida. Suspendiendo hilo por 15 segundos...")
                    time.sleep(15)
                    intentos += 1
                else:
                    print(f"  [X] Error crítico en diapositiva {slide_num}: {e}")
                    break

    return documentos_procesados


def indexar_documentos_capacitacion(documentos: list[dict], persist_directory: str = "./chroma_db"):
    """Inyecta los fragmentos semánticos adicionales en el espacio latente de ChromaDB."""
    print(f"\nIniciando conexión con ChromaDB en {persist_directory}...")
    cliente_chroma = chromadb.PersistentClient(path=persist_directory)
    
    coleccion = cliente_chroma.get_or_create_collection(
        name="capacitacion_interna_v1",
        metadata={"hnsw:space": "cosine"}
    )

    textos = [doc["page_content"] for doc in documentos]
    metadatos = [doc["metadata"] for doc in documentos]
    ids = [str(uuid.uuid4()) for _ in documentos]

    print(f"Indexando {len(textos)} vectores nuevos en la colección...")
    coleccion.add(documents=textos, metadatas=metadatos, ids=ids)
    print("Transacción vectorial de completitud finalizada con éxito.")


if __name__ == "__main__":
    archivo_objetivo = "datos.pdf" 
    
    try:
        datos_extraidos = procesar_diapositivas_restantes(archivo_objetivo, slide_inicio=19)
        
        if datos_extraidos:
            indexar_documentos_capacitacion(datos_extraidos)
        else:
            print("Abortando indexación: No se extrajeron datos válidos.")
            
    except Exception as e:
        print(f"Excepción crítica en el pipeline: {e}")