# API REST para servir el modelo de clasificación de comentarios
# Requisitos: pip install fastapi uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import re
import unicodedata
from pathlib import Path

# ----------------------------
# Configuración de la API
# ----------------------------
app = FastAPI(
    title="Clasificador de Comentarios API",
    description="API para clasificar comentarios de negocios como positivos o negativos",
    version="1.0.0"
)

# ----------------------------
# Cargar modelos entrenados
# ----------------------------
MODEL_PATH = Path(__file__).parent / "modelo.joblib"
VECTORIZER_PATH = Path(__file__).parent / "tfidf.joblib"

try:
    modelo = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Modelo y vectorizador cargados correctamente")
except FileNotFoundError as e:
    print(f"❌ Error: No se encontraron los archivos del modelo. Ejecuta primero el script de entrenamiento.")
    print(f"   Detalles: {e}")
    modelo = None
    vectorizer = None

# ----------------------------
# Función de limpieza (idéntica al entrenamiento)
# ----------------------------
def limpiar(s: str) -> str:
    """Pipeline de limpieza avanzado:
    1. Convertir a minúsculas
    2. Eliminar URLs
    3. Eliminar emails
    4. Eliminar menciones (@usuario)
    5. Eliminar hashtags (conservar texto)
    6. Eliminar emojis
    7. Normalizar acentos (opcional, aquí los conservamos)
    8. Eliminar caracteres especiales
    9. Normalizar espacios
    """
    # 1. Minúsculas
    s = s.lower()
    
    # 2. Eliminar URLs (http, https, www)
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    
    # 3. Eliminar emails
    s = re.sub(r'\S+@\S+', ' ', s)
    
    # 4. Eliminar menciones (@usuario)
    s = re.sub(r'@\w+', ' ', s)
    
    # 5. Eliminar hashtags pero conservar el texto (#recomendado -> recomendado)
    s = re.sub(r'#(\w+)', r'\1', s)
    
    # 6. Eliminar emojis (rangos Unicode principales)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticonos
        "\U0001F300-\U0001F5FF"  # símbolos y pictogramas
        "\U0001F680-\U0001F6FF"  # transporte y símbolos de mapas
        "\U0001F1E0-\U0001F1FF"  # banderas
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # símbolos y emoticonos suplementarios
        "\U00002600-\U000026FF"  # símbolos misceláneos
        "\U00002700-\U000027BF"
        "]+", flags=re.UNICODE
    )
    s = emoji_pattern.sub(' ', s)
    
    # 7. Opcional: Normalizar acentos (descomponer y quitar diacríticos)
    # s = unicodedata.normalize('NFD', s)
    # s = ''.join(char for char in s if unicodedata.category(char) != 'Mn')
    # Aquí conservamos los acentos para mejor contexto en español
    
    # 8. Conservar solo letras (con acentos), números y espacios
    s = re.sub(r"[^a-záéíóúñü0-9\s]", " ", s)
    
    # 9. Normalizar espacios múltiples
    s = re.sub(r"\s+", " ", s).strip()
    
    return s

# ----------------------------
# Modelos de datos (schemas)
# ----------------------------
class ComentarioRequest(BaseModel):
    texto: str = Field(..., description="Comentario a clasificar", min_length=1, max_length=5000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "texto": "Excelente servicio, muy recomendable 😊"
            }
        }

class ComentariosBatchRequest(BaseModel):
    textos: List[str] = Field(..., description="Lista de comentarios a clasificar", min_items=1, max_items=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "textos": [
                    "Excelente servicio, muy recomendable",
                    "Pésima experiencia, no lo recomiendo"
                ]
            }
        }

class PrediccionResponse(BaseModel):
    texto_original: str
    texto_limpio: str
    prediccion: str = Field(..., description="positivo o negativo")
    confianza: Optional[float] = Field(None, description="Nivel de confianza (si el modelo lo soporta)")

class PrediccionesBatchResponse(BaseModel):
    total: int
    predicciones: List[PrediccionResponse]
    resumen: dict

# ----------------------------
# Endpoints de la API
# ----------------------------
@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "mensaje": "API de Clasificación de Comentarios de Negocios",
        "version": "1.0.0",
        "estado": "activo" if modelo and vectorizer else "sin modelo",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch"
        }
    }

@app.get("/health")
async def health_check():
    """Verificar el estado de la API y el modelo"""
    return {
        "status": "healthy" if modelo and vectorizer else "unhealthy",
        "modelo_cargado": modelo is not None,
        "vectorizador_cargado": vectorizer is not None,
        "tipo_modelo": type(modelo).__name__ if modelo else None
    }

@app.post("/predict", response_model=PrediccionResponse)
async def predecir_comentario(request: ComentarioRequest):
    """
    Clasificar un comentario individual como positivo o negativo.
    
    Args:
        request: Objeto con el texto del comentario
        
    Returns:
        Predicción con el texto original, limpio y clasificación
    """
    if not modelo or not vectorizer:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Entrena el modelo primero.")
    
    try:
        # Limpiar texto
        texto_limpio = limpiar(request.texto)
        
        if not texto_limpio:
            raise HTTPException(status_code=400, detail="El texto limpio está vacío después del preprocesamiento")
        
        # Vectorizar y predecir
        X = vectorizer.transform([texto_limpio])
        prediccion = modelo.predict(X)[0]
        
        # Obtener probabilidad si el modelo lo soporta
        confianza = None
        if hasattr(modelo, 'decision_function'):
            # LinearSVC usa decision_function
            decision = modelo.decision_function(X)[0]
            # Convertir a probabilidad aproximada usando función sigmoide
            import numpy as np
            confianza = float(1 / (1 + np.exp(-decision)))
            if prediccion == 0:  # negativo
                confianza = 1 - confianza
        elif hasattr(modelo, 'predict_proba'):
            # LogisticRegression tiene predict_proba
            proba = modelo.predict_proba(X)[0]
            confianza = float(max(proba))
        
        return PrediccionResponse(
            texto_original=request.texto,
            texto_limpio=texto_limpio,
            prediccion="positivo" if prediccion == 1 else "negativo",
            confianza=confianza
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la predicción: {str(e)}")

@app.post("/predict/batch", response_model=PrediccionesBatchResponse)
async def predecir_comentarios_batch(request: ComentariosBatchRequest):
    """
    Clasificar múltiples comentarios en una sola petición.
    
    Args:
        request: Lista de textos a clasificar
        
    Returns:
        Lista de predicciones con resumen estadístico
    """
    if not modelo or not vectorizer:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Entrena el modelo primero.")
    
    try:
        predicciones = []
        
        for texto in request.textos:
            # Limpiar texto
            texto_limpio = limpiar(texto)
            
            if not texto_limpio:
                predicciones.append(PrediccionResponse(
                    texto_original=texto,
                    texto_limpio="",
                    prediccion="desconocido",
                    confianza=None
                ))
                continue
            
            # Vectorizar y predecir
            X = vectorizer.transform([texto_limpio])
            prediccion = modelo.predict(X)[0]
            
            # Obtener probabilidad si el modelo lo soporta
            confianza = None
            if hasattr(modelo, 'decision_function'):
                import numpy as np
                decision = modelo.decision_function(X)[0]
                confianza = float(1 / (1 + np.exp(-decision)))
                if prediccion == 0:
                    confianza = 1 - confianza
            elif hasattr(modelo, 'predict_proba'):
                proba = modelo.predict_proba(X)[0]
                confianza = float(max(proba))
            
            predicciones.append(PrediccionResponse(
                texto_original=texto,
                texto_limpio=texto_limpio,
                prediccion="positivo" if prediccion == 1 else "negativo",
                confianza=confianza
            ))
        
        # Generar resumen
        total_positivos = sum(1 for p in predicciones if p.prediccion == "positivo")
        total_negativos = sum(1 for p in predicciones if p.prediccion == "negativo")
        total_desconocidos = sum(1 for p in predicciones if p.prediccion == "desconocido")
        
        return PrediccionesBatchResponse(
            total=len(predicciones),
            predicciones=predicciones,
            resumen={
                "positivos": total_positivos,
                "negativos": total_negativos,
                "desconocidos": total_desconocidos,
                "porcentaje_positivos": round(total_positivos / len(predicciones) * 100, 2) if predicciones else 0,
                "porcentaje_negativos": round(total_negativos / len(predicciones) * 100, 2) if predicciones else 0
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar las predicciones: {str(e)}")

# ----------------------------
# Ejecutar la API
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Iniciando API de Clasificación de Comentarios...")
    print("📖 Documentación interactiva disponible en: http://localhost:8000/docs")
    print("📊 Interfaz alternativa en: http://localhost:8000/redoc")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
