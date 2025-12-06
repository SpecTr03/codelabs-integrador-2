import cv2  
from ultralytics import YOLO  
  
# Cargar el modelo nano (rápido y liviano)  
model = YOLO("yolov8n.pt")  
  
cap = cv2.VideoCapture(0)  
  
while True:  
    ret, frame = cap.read()  
    if not ret:  
        break  
  
    results = model(frame)  
    annotated = results[0].plot()  # dibuja las detecciones  
  
    cv2.imshow("Detección YOLOv8", annotated)  
  
    if cv2.waitKey(1) & 0xFF == ord("q"):  
        break  
  
cap.release()  
cv2.destroyAllWindows()