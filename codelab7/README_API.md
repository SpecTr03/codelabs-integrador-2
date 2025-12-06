# Guía de uso de la API de Clasificación de Comentarios

## 📋 Requisitos

```bash
pip install fastapi uvicorn requests
```

## 🚀 Inicio rápido

### 1. Entrenar el modelo (si no lo has hecho)
```bash
python clasificador-comentarios-negocio.py
```
Esto generará los archivos `modelo.joblib` y `tfidf.joblib`.

### 2. Iniciar la API
```bash
# Opción 1: Ejecutar directamente
python api.py

# Opción 2: Usar uvicorn con recarga automática
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 3. Probar la API
```bash
# En otra terminal
python test_api.py
```

## 📖 Endpoints disponibles

### 1. `/` - Información general
```bash
curl http://localhost:8000/
```

### 2. `/health` - Estado del servicio
```bash
curl http://localhost:8000/health
```

### 3. `/predict` - Predicción individual
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texto": "Excelente servicio, muy recomendable"}'
```

**Respuesta:**
```json
{
  "texto_original": "Excelente servicio, muy recomendable",
  "texto_limpio": "excelente servicio muy recomendable",
  "prediccion": "positivo",
  "confianza": 0.9876
}
```

### 4. `/predict/batch` - Predicción en lote
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "textos": [
      "Excelente servicio",
      "Pésima experiencia",
      "Producto de calidad"
    ]
  }'
```

**Respuesta:**
```json
{
  "total": 3,
  "predicciones": [
    {
      "texto_original": "Excelente servicio",
      "texto_limpio": "excelente servicio",
      "prediccion": "positivo",
      "confianza": 0.9876
    },
    ...
  ],
  "resumen": {
    "positivos": 2,
    "negativos": 1,
    "desconocidos": 0,
    "porcentaje_positivos": 66.67,
    "porcentaje_negativos": 33.33
  }
}
```

## 🌐 Documentación interactiva

Una vez iniciada la API, visita:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 Ejemplo en Python

```python
import requests

# Predicción individual
response = requests.post(
    "http://localhost:8000/predict",
    json={"texto": "Excelente servicio 😊"}
)
print(response.json())

# Predicción en lote
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "textos": [
            "Muy buena atención",
            "Pésima experiencia",
            "Producto de calidad"
        ]
    }
)
print(response.json())
```

## 🔧 Ejemplo en JavaScript

```javascript
// Predicción individual
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    texto: 'Excelente servicio 😊'
  })
})
  .then(response => response.json())
  .then(data => console.log(data));

// Predicción en lote
fetch('http://localhost:8000/predict/batch', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    textos: [
      'Muy buena atención',
      'Pésima experiencia',
      'Producto de calidad'
    ]
  })
})
  .then(response => response.json())
  .then(data => console.log(data));
```

## 📊 Características de la API

- ✅ **Preprocesamiento automático**: Elimina URLs, emails, emojis, menciones
- ✅ **Predicción individual y en lote**: Hasta 100 comentarios por petición
- ✅ **Confianza de predicción**: Incluye nivel de confianza cuando está disponible
- ✅ **Resumen estadístico**: Para predicciones en lote
- ✅ **Validación de entrada**: Longitud máxima, tipos de datos
- ✅ **Documentación OpenAPI**: Swagger UI y ReDoc
- ✅ **Manejo de errores robusto**: Códigos de estado HTTP apropiados

## 🐳 Despliegue con Docker (opcional)

Crea un `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py modelo.joblib tfidf.joblib ./

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Construir y ejecutar:
```bash
docker build -t clasificador-api .
docker run -p 8000:8000 clasificador-api
```

## ⚡ Rendimiento

- Predicción individual: ~10-20ms
- Predicción en lote (100 comentarios): ~100-200ms
- Recomendado: Usar predicción en lote para mejor throughput

## 🛡️ Seguridad

Para producción, considera:
- Agregar autenticación (API keys, JWT)
- Limitar rate limiting
- Validar CORS según necesidades
- Usar HTTPS
- Monitoreo y logging

## 📝 Notas

- La API carga el modelo al iniciar
- Asegúrate de tener `modelo.joblib` y `tfidf.joblib` en el mismo directorio
- El preprocesamiento de texto debe ser idéntico al usado en el entrenamiento
