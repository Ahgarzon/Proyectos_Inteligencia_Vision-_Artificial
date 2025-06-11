"""
YOLOv8-SEGM en vivo  ·  Formas + Tamaño  ·  Snapshot 3-D PLY + visor matplotlib
Teclas:
    c  →   snapshot_cloud.ply  +  ventana 3-D interactiva
    q  →   salir
"""

import cv2, time, numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# -------------------- PARÁMETROS -------------------- #
MODEL_PATH  = "yolov8n-seg.pt"
CONF        = 0.25
MIN_PIXELS  = 800
ARUCO_MM    = 50.0      # lado real del marcador
ARUCO_ID    = 0
FONT        = cv2.FONT_HERSHEY_SIMPLEX
VALID_CLASSES = {"sports ball", "bowl"}   # rueda ≈ sports ball/bowl
# ---------------------------------------------------- #

def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray):
    rgb8 = (rgb * 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb8):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

def show_cloud(xyz: np.ndarray, rgb: np.ndarray):
    fig = plt.figure("Snapshot cloud")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1, depthshade=False)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    plt.show()

def shape_and_color(cnt):
    peri  = cv2.arcLength(cnt, True)
    area  = cv2.contourArea(cnt)
    circ  = 4*np.pi*area / (peri**2 + 1e-5)
    verts = len(cv2.approxPolyDP(cnt, 0.04*peri, True))
    if circ > .80:            return "Círculo",   (255,   0,   0)
    if verts == 3:            return "Triángulo", (  0, 255, 255)
    if verts == 4:
        x, y, w, h = cv2.boundingRect(cnt)
        return ("Cuadrado" if .9 <= w/h <= 1.1 else "Rectángulo",
                (  0, 255,   0))
    return "Polígono", (200, 200, 0)

# --------------- Cargar modelos -------------------- #
model  = YOLO(MODEL_PATH)
names  = model.model.names
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
try:    # OpenCV ≥4.7
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    detect = detector.detectMarkers
except AttributeError:                       # OpenCV 4.6
    detect = lambda img: cv2.aruco.detectMarkers(img, aruco_dict)

# --------------- Bucle en vivo --------------------- #
cap = cv2.VideoCapture(0);  assert cap.isOpened()
print("Teclas  c = snapshot  ·  q = salir")
px_per_mm, t_prev = None, 0

while True:
    ok, frame = cap.read()
    if not ok: break
    vis = frame.copy()

    # 1) ArUco ⇒ escala
    corners, ids, _ = detect(frame)
    if ids is not None and ARUCO_ID in ids:
        pts = corners[np.where(ids.flatten()==ARUCO_ID)[0][0]].reshape(4,2)
        avg_side = sum(np.linalg.norm(pts[i]-pts[(i+1)%4]) for i in range(4))/4
        px_per_mm = avg_side / ARUCO_MM
        cv2.polylines(vis,[pts.astype(int)],True,(0,0,255),2)
        cv2.putText(vis,"Aruco OK",(int(pts[0][0]),int(pts[0][1])-10),
                    FONT,0.5,(0,0,255),1)

    # 2) YOLOv8-Seg
    res = model.predict(frame, conf=CONF, stream=False, verbose=False)[0]
    for mask_t, box, cls in zip(res.masks.data, res.boxes.xyxy, res.boxes.cls):
        if names[int(cls)] not in VALID_CLASSES: continue
        mask = mask_t.cpu().numpy().astype(np.uint8)
        if mask.sum() < MIN_PIXELS: continue

        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea)          # ← cnt AHORA SÍ existe
        shape, col = shape_and_color(cnt)

        x1,y1,x2,y2 = map(int, box.cpu().tolist())
        w_px, h_px  = x2-x1, y2-y1
        label = (f"{shape}: {w_px/px_per_mm:.1f}×{h_px/px_per_mm:.1f} mm"
                 if px_per_mm else f"{shape}: {w_px}px×{h_px}px")

        cv2.rectangle(vis, (x1,y1), (x2,y2), col, 2)
        cv2.putText(vis, label, (x1, max(20,y1-8)), FONT, .6, col, 2)

    # 3) FPS
    t_now = time.time(); fps = 1/(t_now-t_prev) if t_prev else 0; t_prev = t_now
    cv2.putText(vis, f"{fps:.1f} FPS", (10,25), FONT, .7, (0,255,0), 2)

    # 4) Mostrar
    cv2.imshow("YOLOv8 live", vis)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break
    if key == ord('c'):
        H,W,_ = frame.shape; cx,cy = W/2, H/2
        pts, cols = [], []
        for mask_t in res.masks.data:
            m = mask_t.cpu().numpy().astype(np.uint8)
            ys, xs = np.where(m == 1)
            if xs.size == 0: continue
            scale = px_per_mm or 1
            xs_mm = (xs - cx) / scale
            ys_mm = (ys - cy) / scale
            zs_mm = np.zeros_like(xs_mm, float)    # sin profundidad real
            pts.append(np.stack([xs_mm, -ys_mm, zs_mm], 1))
            cols.append(frame[ys, xs][:, ::-1]/255.)
        if pts:
            xyz = np.concatenate(pts); rgb = np.concatenate(cols)
            write_ply("snapshot_cloud.ply", xyz, rgb)
            print("✅ snapshot_cloud.ply guardado.")
            show_cloud(xyz, rgb)
        else:
            print("⚠️  No hay máscaras para snapshot.")

cap.release(); cv2.destroyAllWindows()
