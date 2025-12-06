
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from time import time

# Lista de imágenes a procesar
imagenes = ['images/daredevil.jpg', 'images/manchester.jpg', 'images/hinchas.jpeg']

detector = MTCNN()  # puedes ajustar min_face_size
umbral_list = [0.6, 0.95]  # lista de umbrales a comparar

for nombre_img in imagenes:
    print(f"\nProcesando: {nombre_img}")
    img = cv2.imread(nombre_img)
    if img is None:
        print(f"No se pudo cargar la imagen: {nombre_img}")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Medir tiempo de inferencia promedio en 5 corridas
    tiempos = []
    resultados = None
    for _ in range(5):
        t0 = time()
        res = detector.detect_faces(img_rgb)
        t1 = time()
        tiempos.append((t1 - t0)*1000)
        resultados = res  # guardar el último resultado para mostrar
    tiempo_promedio = sum(tiempos) / len(tiempos)

    print(f"Detected: {len(resultados)} rostro(s) • tiempo promedio: {tiempo_promedio:.1f} ms (5 corridas)")
    for r in resultados:
        print(r['confidence'], r['box'], r['keypoints'].keys())
    # Comparar diferentes umbrales
    for thr in umbral_list:
        filtrados = [r for r in resultados if r['confidence'] >= thr]
        print(f"Con thr={thr} quedan {len(filtrados)} rostros")
    vis = img_rgb.copy()
    for r in resultados:
        x, y, w, h = r['box']
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
        # Dibujar landmarks con colores diferentes
        # Alignment nos sirve para normalizar los rostros, esto facilita el las comparaciones y reconocimientos.
        keypoints = r['keypoints']
        # Ojos
        cv2.circle(vis, keypoints['left_eye'], 4, (0,0,255), -1)   # rojo
        cv2.circle(vis, keypoints['right_eye'], 4, (0,0,255), -1)  # rojo
        # Nariz
        cv2.circle(vis, keypoints['nose'], 4, (0,255,255), -1)     # amarillo
        # Boca
        cv2.circle(vis, keypoints['mouth_left'], 4, (0,255,0), -1) # verde
        cv2.circle(vis, keypoints['mouth_right'], 4, (0,255,0), -1)# verde
    plt.imshow(vis)
    plt.title(nombre_img)
    plt.axis('off')
    plt.show()

def iou(a, b):
    # a,b en formato [x,y,w,h]
    ax1, ay1, aw, ah = a; ax2, ay2 = ax1+aw, ay1+ah
    bx1, by1, bw, bh = b; bx2, by2 = bx1+bw, by1+bh
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1)*max(0, iy2-iy1)
    union = aw*ah + bw*bh - inter
    return inter/union if union>0 else 0.0

# Si tienes una caja "ground truth" gt_box, compara:
# print(iou(res[0]['box'], gt_box))