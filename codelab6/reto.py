from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import time
import pandas as pd


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


def calculate_mean_iou(ssd_boxes, yolo_boxes, threshold=0.3):
    """Calculate mean IoU between all SSD and YOLO box pairs with IoU > threshold."""
    if len(ssd_boxes) == 0 or len(yolo_boxes) == 0:
        return 0.0
    
    ious = []
    for ssd_box in ssd_boxes:
        for yolo_box in yolo_boxes:
            iou_val = iou(ssd_box, yolo_box)
            if iou_val > threshold:
                ious.append(iou_val)
    
    return sum(ious) / len(ious) if ious else 0.0


def process_image(image_path, model, yolo_model, preprocess, confidence_threshold=0.5):
    """Process a single image with both SSD and YOLO models."""
    print(f"\n{'='*60}")
    print(f"Procesando: {image_path}")
    print(f"{'='*60}")
    
    # Cargar imagen
    img = Image.open(image_path).convert("RGB")
    
    # SSD prediction
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        t0_ssd = time.time()
        out = model(x)[0]
        t1_ssd = time.time()
    
    ssd_time = t1_ssd - t0_ssd
    boxes, _, scores = out["boxes"], out["labels"], out["scores"]
    
    # Filtrar por confianza
    ssd_mask = scores > confidence_threshold
    ssd_boxes_filtered = boxes[ssd_mask]
    ssd_count = len(ssd_boxes_filtered)
    
    print(f"SSD: {ssd_time:.4f} seg | Objetos detectados: {ssd_count}")
    
    # YOLO prediction
    t0_yolo = time.time()
    yolo_result = yolo_model(image_path)[0]
    t1_yolo = time.time()
    
    yolo_time = t1_yolo - t0_yolo
    yolo_boxes = yolo_result.boxes.xyxy
    yolo_scores = yolo_result.boxes.conf
    
    # Filtrar por confianza
    yolo_mask = yolo_scores > confidence_threshold
    yolo_boxes_filtered = yolo_boxes[yolo_mask]
    yolo_count = len(yolo_boxes_filtered)
    
    print(f"YOLO: {yolo_time:.4f} seg | Objetos detectados: {yolo_count}")
    
    # Calculate mean IoU between all detections
    mean_iou = calculate_mean_iou(
        [box.tolist() for box in ssd_boxes_filtered],
        [box.tolist() for box in yolo_boxes_filtered]
    )
    
    print(f"IoU medio entre predicciones: {mean_iou:.4f}")
    
    return {
        'imagen': image_path.split('/')[-1].split('\\')[-1],
        'tiempo_ssd': ssd_time,
        'tiempo_yolo': yolo_time,
        'objetos_ssd': ssd_count,
        'objetos_yolo': yolo_count,
        'iou_medio': mean_iou
    }


# Cargar modelos
print("Cargando modelos...")
weights = SSD300_VGG16_Weights.DEFAULT
model = ssd300_vgg16(weights=weights).eval()
yolo_model = YOLO("yolov8n.pt")
preprocess = weights.transforms()

# Lista de imágenes a procesar
images = ["chilena.jpg", "galaxia-andromeda.jpg", "metallica.jpeg"]

# Procesar todas las imágenes
results = []
for image_path in images:
    result = process_image(image_path, model, yolo_model, preprocess)
    results.append(result)

# Crear tabla comparativa
print(f"\n{'='*60}")
print("TABLA COMPARATIVA")
print(f"{'='*60}\n")

df = pd.DataFrame(results)

# Formatear la tabla
df['tiempo_ssd'] = df['tiempo_ssd'].apply(lambda x: f"{x:.4f}s")
df['tiempo_yolo'] = df['tiempo_yolo'].apply(lambda x: f"{x:.4f}s")
df['iou_medio'] = df['iou_medio'].apply(lambda x: f"{x:.4f}")

# Renombrar columnas para mejor visualización
df.columns = ['Imagen', 'Tiempo SSD', 'Tiempo YOLO', 'Objetos SSD', 'Objetos YOLO', 'IoU Medio']

print(df.to_string(index=False))
print()

# Guardar tabla en CSV
df.to_csv('comparacion_ssd_yolo.csv', index=False)
print("✓ Tabla guardada en: comparacion_ssd_yolo.csv")