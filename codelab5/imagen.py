from ultralytics import YOLO  
import time
  
# Cargar el modelo nano (rápido y liviano)  
model = YOLO("yolov8n.pt")  
  
# --- Detección en una imagen ---  
image_path = "jinjer-band.jpg"  # cámbialo por tu imagen  
t2 = time.time()  
results = model(image_path)  
t3 = time.time()
  
# Mostrar resultados  
for r in results:  
    print(r.names)  # clases disponibles  
    print(r.boxes)  # cajas detectadas  
  
# Guardar imagen con anotaciones  
results[0].save(filename="resultado_yolo.jpg")