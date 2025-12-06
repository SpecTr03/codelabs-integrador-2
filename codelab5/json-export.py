import json  
import cv2  
from ultralytics import YOLO  
  
# Cargar el modelo nano (rápido y liviano)  
model = YOLO("yolov8n.pt")  
  
frame = cv2.imread("jinjer-band.jpg")  
results = model(frame)  
  
detecciones = []  
for r in results[0].boxes:  
    obj = {  
        "clase": model.names[int(r.cls)],  
        "score": float(r.conf),  
        "bbox": r.xyxy.tolist()[0]  # [x1,y1,x2,y2]  
    }  
    detecciones.append(obj)  
  
with open("resultados.json", "w") as f:  
    json.dump(detecciones, f, indent=4)