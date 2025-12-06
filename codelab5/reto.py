import json
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Cargar el modelo nano (rápido y liviano)
model = YOLO("yolov8n.pt")

# Abrir el video
cap = cv2.VideoCapture("video_personas.mp4")

if not cap.isOpened():
    print("Error: No se pudo abrir el video")
    exit()

detecciones_video = []
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Realizar detección en el frame actual
    results = model(frame)
    
    detecciones_frame = []
    for r in results[0].boxes:
        clase = model.names[int(r.cls)]
        
        # Filtrar solo personas
        if clase == "person":
            obj = {
                "clase": clase,
                "score": float(r.conf),
                "bbox": r.xyxy.tolist()[0]  # [x1,y1,x2,y2]
            }
            detecciones_frame.append(obj)
    
    # Agregar las detecciones del frame con su número
    if detecciones_frame:  # Solo agregar si hay detecciones
        detecciones_video.append({
            "frame": frame_num,
            "detecciones": detecciones_frame
        })
    
    frame_num += 1
    print(f"Procesado frame {frame_num}")

cap.release()

# Guardar todas las detecciones en JSON
with open("resultados-video_personas.json", "w") as f:
    json.dump(detecciones_video, f, indent=4)

print(f"Proceso completado. Total de frames procesados: {frame_num}")
print("Detecciones guardadas en resultados-video_personas.json")

# Crear gráfico del número de personas detectadas por frame
frames_con_detecciones = [item["frame"] for item in detecciones_video]
num_personas_por_frame = [len(item["detecciones"]) for item in detecciones_video]

plt.figure(figsize=(12, 6))
plt.plot(frames_con_detecciones, num_personas_por_frame, marker='o', linestyle='-', linewidth=2, markersize=4)
plt.xlabel('Número de Frame')
plt.ylabel('Número de Personas Detectadas')
plt.title('Detección de Personas por Frame')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('grafico_detecciones.png', dpi=300)
plt.show()

print("Gráfico guardado como grafico_detecciones.png")