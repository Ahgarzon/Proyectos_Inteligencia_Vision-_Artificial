"""
Instantánea  ▸  Contornos rápidos  ▸  SAM (box-prompt)  ▸  Clasificación de figura
-------------------------------------------------------------------------------
• Captura UN frame de la webcam.
• Busca contornos grandes y filtra por área + relación aspecto.
• Cada caja se manda a SAM (predict(box=...)).
• Clasifica la forma (triángulo, cuadrado / rectángulo, círculo).
"""

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# ---------------- CONFIGURA AQUÍ ---------------- #
CKPT          = "sam_vit_b_01ec64.pth"   # ruta al checkpoint
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MIN_AREA      = 800                      # área mínima para contorno inicial
MAX_AREA      = 400_000                  # área máxima (descarta paredes, etc.)
MAX_AR_DIFF   = 4.0                      # relación aspecto máx. ancho/alto
EDGE_THRESH   = 60                       # umbral Canny
BLUR_K        = 3                        # kernel blur para suavizar
# ------------------------------------------------ #

# 1. Cargar SAM
sam = sam_model_registry["vit_b"](checkpoint=CKPT).to(DEVICE)
predictor = SamPredictor(sam)

# 2. Capturar una instantánea
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se puede abrir la cámara.")
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("No se pudo capturar el frame.")

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
predictor.set_image(rgb)

# 3. Contornos rápidos para candidatos
gray  = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                         (BLUR_K, BLUR_K), 0)
edges = cv2.Canny(gray, EDGE_THRESH, EDGE_THRESH * 3)
edges = cv2.dilate(edges, None, iterations=1)

cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for c in cnts:
    area = cv2.contourArea(c)
    if not (MIN_AREA <= area <= MAX_AREA):
        continue
    x, y, w, h = cv2.boundingRect(c)
    ar = max(w, h) / (min(w, h) + 1e-5)
    if ar > MAX_AR_DIFF:
        continue
    boxes.append([x, y, x + w, y + h])

if not boxes:
    print("No se detectaron candidatos.")
    cv2.imshow("Snapshot", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    raise SystemExit

# 4. Procesar cada box con SAM y clasificar
for (x0, y0, x1, y1) in boxes:
    box_np = np.array([x0, y0, x1, y1])           # (4,)  x0,y0,x1,y1
    masks, _, _ = predictor.predict(
        box=box_np[None, :],                      # añade batch dim
        multimask_output=False
    )
    seg = masks[0]                               # (H,W) bool
    mask_u8 = seg.astype("uint8") * 255

    cnts_seg, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_seg:
        continue
    cnt  = max(cnts_seg, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        continue

    peri   = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    verts  = len(approx)

    if verts == 3:
        shape, color = "Triángulo", (0, 255, 255)
    elif verts == 4:
        shape, color = "Cuadrado",  (0, 255,   0)
    else:
        shape, color = "Círculo",   (255,  0,   0)

    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    cv2.putText(frame, shape, (x0, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# 5. Mostrar resultado
cv2.imshow("Instantánea clasificada", frame)
print("Cierra la ventana o pulsa cualquier tecla sobre ella para salir.")
cv2.waitKey(0)
cv2.destroyAllWindows()
