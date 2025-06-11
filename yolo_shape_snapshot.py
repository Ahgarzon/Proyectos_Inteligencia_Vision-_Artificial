"""
YOLOv8-Seg  ▸  Máscaras precisas  ▸  Clasificación de formas geométricas
-----------------------------------------------------------------------
• Captura UN frame de la webcam.
• Corre YOLOv8n-seg (pre-entrenado en COCO, 640×640).
• Cada máscara se simplifica a un contorno y se clasifica por nº de vértices:
      - 3   → Triángulo
      - 4   → Cuadrado / Rectángulo
      - >4  → Círculo aprox.
• Dibuja resultado y espera una tecla para salir.

Cambio clave:  NO hay contornos iniciales → la segmentación la da YOLO.
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------- CONFIGURA AQUÍ ------------------ #
MODEL_PATH   = "yolov8n-seg.pt"   # usa el peso nano-seg (6 MB). Cambia a s/m/l si quieres
CONF_THRESH  = 0.25               # confianza mínima por instancia
MIN_MASK_PX  = 600                # descarta máscaras muy pequeñas
# --------------------------------------------------- #

# Cargar modelo
model = YOLO(MODEL_PATH)          # automáticamente se descarga la primera vez

# Capturar instantánea
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("No se capturó frame")

# Inference (una sola imagen)
#  – Source puede ser un ndarray BGR directamente con stream=False
results = model.predict(frame, conf=CONF_THRESH, stream=False, verbose=False)[0]

# Procesar cada máscara
names = model.model.names         # diccionario id→clase COCO (por si quieres imprimir)
for j, mask_tensor in enumerate(results.masks.data):
    # Cada máscara → numpy bool (H, W)
    mask = mask_tensor.cpu().numpy().astype(np.uint8)
    if mask.sum() < MIN_MASK_PX:
        continue  # demasiado pequeña

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        continue
    cnt = max(cnts, key=cv2.contourArea)

    # Clasificación de la forma
    peri   = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    verts  = len(approx)

    if verts == 3:
        shape, color = "Triángulo", (0, 255, 255)
    elif verts == 4:
        shape, color = "Cuadrado",  (0, 255,   0)
    else:
        shape, color = "Círculo",   (255,  0,   0)

    # Bounding box original de YOLO
    x1, y1, x2, y2 = map(int, results.boxes.xyxy[j].cpu().tolist())
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, shape, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Mostrar
cv2.imshow("YOLOv8-Seg + Shape Classifier", frame)
print("Pulsa una tecla sobre la ventana para salir")
cv2.waitKey(0)
cv2.destroyAllWindows()
