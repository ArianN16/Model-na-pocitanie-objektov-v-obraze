import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics import settings
settings.update({"sync": False})

# =========================
# KONFIGURÁCIA (ODPORÚČANÝ ŠTART)
# =========================
MODEL_PATH = "cesta k modelu umelej inteligencie"
IMAGE_PATH = "cesta k snímku"

CONF_THRES = 0.3  # nižšie -> lepšie chytí prekryté
IMGSZ = 960              
DEVICE = 0

# postprocess pre count
MIN_MASK_AREA = 900    
MASK_IOU_SUPPRESS = 0.8 # potlačenie duplicitných masiek (0.5–0.7)

# vizuál
DRAW_IDS = True
FILL_MASKS = True
ALPHA = 0.35

# viewer
BASE_WINDOW_NAME = "YOLOv8s-seg viewer"
MAX_START_WIDTH = 1400
MIN_ZOOM = 0.1
MAX_ZOOM = 10.0
ZOOM_STEP = 1.1


# =========================
# POMOCNÉ FUNKCIE
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return (inter / uni) if uni > 0 else 0.0


# =========================
# NAČÍTANIE MODELU + INFERENCE
# =========================
model = YOLO(MODEL_PATH)
results = model(source=IMAGE_PATH, conf=CONF_THRES, imgsz=IMGSZ, device=DEVICE)
result = results[0]

# =========================
# NAČÍTANIE OBRÁZKA
# =========================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Neviem načítať obrázok: {IMAGE_PATH}")

draw = img.copy()
overlay = img.copy()

# =========================
# MASK NMS (odstránenie duplicitných masiek) + MIN_MASK_AREA filter
# =========================
kept_polys = []
centroids = []

if result.masks is not None:
    # binárne masky v internom rozlíšení (na IoU medzi maskami)
    m = (result.masks.data > 0.5).cpu().numpy().astype(np.uint8)  # [N,H,W]
    N = m.shape[0]

    suppressed = set()
    kept_idxs = []

    for i in range(N):
        if i in suppressed:
            continue
        kept_idxs.append(i)
        for j in range(i + 1, N):
            if j in suppressed:
                continue
            if mask_iou(m[i], m[j]) > MASK_IOU_SUPPRESS:
                suppressed.add(j)

    # z kept indexov sprav polygóny v originálnej mierke a filtruj podľa plochy
    for i in kept_idxs:
        pts = result.masks.xy[i].astype(np.int32)
        if pts.shape[0] < 3:
            continue

        area = cv2.contourArea(pts)  # px^2 v ORIGINÁLNEJ fotke
        if area < MIN_MASK_AREA:
            continue

        kept_polys.append(pts)

        M = cv2.moments(pts)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = int(pts[0][0]), int(pts[0][1])
        centroids.append((cx, cy))

num_objects = len(kept_polys)

print(f"\n📷 {IMAGE_PATH}")
print(f"➡️  Počet detekovaných objektov: {num_objects}")


# =========================
# KRESLENIE
# =========================
if FILL_MASKS and kept_polys:
    for pts in kept_polys:
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
    draw = cv2.addWeighted(overlay, ALPHA, draw, 1 - ALPHA, 0)

for pts in kept_polys:
    cv2.polylines(draw, [pts], True, (0, 0, 255), 2)

if DRAW_IDS:
    for i, (cx, cy) in enumerate(centroids, start=1):
        cv2.putText(draw, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(draw, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

# badge
badge_text = f"COUNT: {num_objects}"
font = cv2.FONT_HERSHEY_SIMPLEX
scale, thickness = 1.2, 3
(text_w, text_h), _ = cv2.getTextSize(badge_text, font, scale, thickness)
pad = 12
cv2.rectangle(draw, (10, 10), (10 + text_w + 2 * pad, 10 + text_h + 2 * pad), (0, 0, 0), -1)
cv2.putText(draw, badge_text, (10 + pad, 10 + pad + text_h), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


# =========================
# INTERAKTÍVNY VIEWER (zoom/pan + title s count)
# =========================
state = {"zoom": 1.0, "pan_x": 0, "pan_y": 0, "dragging": False, "last_x": 0, "last_y": 0}

def window_title():
    return f"{BASE_WINDOW_NAME} | obj={num_objects} | conf={CONF_THRES} imgsz={IMGSZ} area>={MIN_MASK_AREA} iou>{MASK_IOU_SUPPRESS} | {IMAGE_PATH}"

def render():
    h, w = draw.shape[:2]
    zoom = state["zoom"]

    scaled_w = max(1, int(w * zoom))
    scaled_h = max(1, int(h * zoom))
    scaled = cv2.resize(draw, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    viewport_w = min(MAX_START_WIDTH, scaled_w)
    viewport_h = min(int(MAX_START_WIDTH * (h / w)), scaled_h)
    viewport_w = max(400, viewport_w)
    viewport_h = max(300, viewport_h)

    max_pan_x = max(0, scaled_w - viewport_w)
    max_pan_y = max(0, scaled_h - viewport_h)
    state["pan_x"] = clamp(state["pan_x"], 0, max_pan_x)
    state["pan_y"] = clamp(state["pan_y"], 0, max_pan_y)

    x0, y0 = state["pan_x"], state["pan_y"]
    view = scaled[y0:y0 + viewport_h, x0:x0 + viewport_w].copy()

    info = "drag pan | +/- or W/S zoom | R reset | ESC/Q"
    cv2.putText(view, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(view, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    # update title (OpenCV >= 4.5 typicky)
    try:
        cv2.setWindowTitle(WINDOW_NAME, window_title())
    except Exception:
        pass

    cv2.imshow(WINDOW_NAME, view)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEWHEEL:
        old_zoom = state["zoom"]
        if flags > 0:
            state["zoom"] = clamp(state["zoom"] * ZOOM_STEP, MIN_ZOOM, MAX_ZOOM)
        else:
            state["zoom"] = clamp(state["zoom"] / ZOOM_STEP, MIN_ZOOM, MAX_ZOOM)

        ratio = state["zoom"] / old_zoom
        state["pan_x"] = int(state["pan_x"] * ratio + x * (ratio - 1))
        state["pan_y"] = int(state["pan_y"] * ratio + y * (ratio - 1))
        render()

    elif event == cv2.EVENT_LBUTTONDOWN:
        state["dragging"] = True
        state["last_x"], state["last_y"] = x, y

    elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
        dx, dy = x - state["last_x"], y - state["last_y"]
        state["pan_x"] -= dx
        state["pan_y"] -= dy
        state["last_x"], state["last_y"] = x, y
        render()

    elif event == cv2.EVENT_LBUTTONUP:
        state["dragging"] = False

WINDOW_NAME = window_title()
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

h0, w0 = draw.shape[:2]
if w0 > MAX_START_WIDTH:
    state["zoom"] = MAX_START_WIDTH / w0

render()

while True:
    key = cv2.waitKey(20) & 0xFF
    if key in [27, ord("q"), ord("Q")]:
        break

    if key in [ord("r"), ord("R")]:
        state["zoom"] = (MAX_START_WIDTH / w0) if w0 > MAX_START_WIDTH else 1.0
        state["pan_x"] = 0
        state["pan_y"] = 0
        render()

    if key in [ord("+"), ord("="), ord("w"), ord("W")]:
        state["zoom"] = clamp(state["zoom"] * ZOOM_STEP, MIN_ZOOM, MAX_ZOOM)
        render()

    if key in [ord("-"), ord("_"), ord("s"), ord("S")]:
        state["zoom"] = clamp(state["zoom"] / ZOOM_STEP, MIN_ZOOM, MAX_ZOOM)
        render()

cv2.destroyAllWindows()
