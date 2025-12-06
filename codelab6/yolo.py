from ultralytics import YOLO

model_yolo = YOLO("yolov8n.pt")
# Ultralytics returns a list of Results; take the first item for a single image
results = model_yolo("chilena.jpg")[0]
results.show()