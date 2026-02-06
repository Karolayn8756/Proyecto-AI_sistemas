import base64
import time
import cv2
import requests
from collections import deque
import math

IA_URL = "http://127.0.0.1:8001/infer"

# Suavizado simple para que no tiemble (EMA)
class EMA2D:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.prev = None

    def update(self, x, y):
        if self.prev is None:
            self.prev = (x, y)
        px, py = self.prev
        nx = (1 - self.alpha) * px + self.alpha * x
        ny = (1 - self.alpha) * py + self.alpha * y
        self.prev = (nx, ny)
        return nx, ny

ema = EMA2D(alpha=0.35)

def bgr_to_b64jpg(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return base64.b64encode(buf).decode("utf-8")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_call = 0
call_every = 1/12  # 12 FPS hacia IA (estable)

last_data = None

# ===== Fruit Ninja: "espada" (trail) + velocidad =====
trail = deque(maxlen=12)   # más grande = más largo el rastro
prev_tip = None
prev_tip_t = None
tip_speed = 0.0
CUT_SPEED = 900  # umbral px/seg para considerar "corte" (ajustable)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # modo espejo (1 = horizontal)

    if not ret:
        break

    now = time.time()

    # Llamada a IA (throttle)
    if now - last_call >= call_every:
        last_call = now
        img_b64 = bgr_to_b64jpg(frame)
        try:
            r = requests.post(IA_URL, json={"image_b64": img_b64}, timeout=1.5)
            last_data = r.json()
        except Exception:
            # si falla, seguimos mostrando cámara sin crashear
            last_data = None

    data = last_data

    # Dibujar overlay si hay datos
    if data and data.get("hand_landmarks"):
        h, w = frame.shape[:2]
        lm = data["hand_landmarks"]

        # punta del índice = landmark 8
        ix = int(lm[8]["x"] * w)
        iy = int(lm[8]["y"] * h)

        # suavizado
        sx, sy = ema.update(ix, iy)
        sx, sy = int(sx), int(sy)

        # punto del dedo
        cv2.circle(frame, (sx, sy), 10, (0, 255, 0), -1)
        cv2.putText(frame, "INDEX TIP", (sx + 12, sy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ===== trail + speed =====
        tip = (sx, sy)
        trail.append(tip)

        if prev_tip is not None and prev_tip_t is not None:
            dt = max(now - prev_tip_t, 1e-6)
            dx = tip[0] - prev_tip[0]
            dy = tip[1] - prev_tip[1]
            tip_speed = math.sqrt(dx*dx + dy*dy) / dt

        prev_tip = tip
        prev_tip_t = now

        # dibujar rastro (espada)
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i-1], trail[i], (0, 255, 255), 4)

        # velocidad
        cv2.putText(frame, f"Speed: {tip_speed:.0f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # indicador de corte
        if tip_speed >= CUT_SPEED:
            cv2.putText(frame, "CUT!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

    else:
        # si no hay mano, resetea para que no se quede pegado
        trail.clear()
        prev_tip = None
        prev_tip_t = None
        tip_speed = 0.0

    # Smile / latency
    if data:
        smile = data.get("smile_score")
        latency = data.get("latency_ms")
    else:
        smile, latency = None, None

    if smile is not None:
        cv2.putText(frame, f"SmileScore: {smile:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if latency is not None:
        cv2.putText(frame, f"IA Latency: {latency} ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, "ESC para salir", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Client - Camera + IA Overlay", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
