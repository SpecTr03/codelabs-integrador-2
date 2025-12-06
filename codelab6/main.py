from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import time


def iou(box_a, box_b):
    """Compute Intersection over Union between two boxes [x1, y1, x2, y2]."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    return inter_area / float(box_a_area + box_b_area - inter_area + 1e-9)


def best_box_for_class(boxes, labels, scores, target_class, class_names):
    """Return best-scoring box for target_class name, or (None, None) if absent."""
    best_box = None
    best_score = -1.0
    for box, lab, sc in zip(boxes, labels, scores):
        if class_names[int(lab)] == target_class and float(sc) > best_score:
            best_box = box
            best_score = float(sc)
    return best_box, best_score

# Cargar SSD300-VGG16 con pesos preentrenados por defecto (COCO) y ponerlo en modo evaluación para inferencia
weights = SSD300_VGG16_Weights.DEFAULT
model = ssd300_vgg16(weights=weights).eval()
yolo_model = YOLO("yolov8n.pt")

# Transforms correctos (no redimensionan fijo a 300x300, se encargan de escalar de forma consistente). Prepara la imagen igual que durante el entrenamiento (resize, normalización, etc.) para que el modelo la entienda bien
preprocess = weights.transforms()

# Imagen
img = Image.open("chilena.jpg").convert("RGB")

# Convierte la imagen en un tensor preprocesado y le agrega la dimensión de batch (1 imagen)
x = preprocess(img).unsqueeze(0)

# Ejecuta el modelo en modo inferencia (sin gradientes) y obtiene las predicciones de la única imagen del batch
with torch.no_grad():
    t0 = time.time()  
    out = model(x)[0]  
    t1 = time.time()  
  
print("SSD:", t1-t0, "seg")

# Boxes ya están en escala original 🎉
boxes, labels, scores = out["boxes"], out["labels"], out["scores"]
categories = weights.meta["categories"]

# Ejemplo: calcula IoU entre las dos primeras detecciones (si existen)
if len(boxes) >= 2:
    iou_val = iou(boxes[0].tolist(), boxes[1].tolist())
    print(f"IoU entre detección 0 y 1: {iou_val:.3f}")

# --- SSD vs YOLO IoU para el mismo objeto (ejemplo: "person") ---
target_class = "person"

ssd_box, ssd_score = best_box_for_class(boxes, labels, scores, target_class, categories)

yolo_result = yolo_model("chilena.jpg")[0]
yolo_boxes = yolo_result.boxes.xyxy
yolo_labels = yolo_result.boxes.cls
yolo_scores = yolo_result.boxes.conf
yolo_names = yolo_result.names
yolo_box, yolo_score = best_box_for_class(yolo_boxes, yolo_labels, yolo_scores, target_class, yolo_names)

if ssd_box is not None and yolo_box is not None:
    iou_ssd_yolo = iou(ssd_box.tolist(), yolo_box.tolist())
    print(f"IoU {target_class} SSD vs YOLO: {iou_ssd_yolo:.3f} (ssd_score={ssd_score:.2f}, yolo_score={yolo_score:.2f})")
else:
    print(f"No se encontró la clase {target_class} en ambos modelos para comparar IoU.")

# Visualizar
fig, ax = plt.subplots(1, figsize=(8, 6))
ax.imshow(img)
for box, lab, sc in zip(boxes, labels, scores):
    if sc > 0.5:  # filtrar confianza
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                                 edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1, f"{categories[int(lab)]}: {sc:.2f}",
            color="white", fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.5})
plt.axis("off")
plt.show()