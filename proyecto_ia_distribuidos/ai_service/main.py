import base64
import time
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

# ---------- Mediapipe robusto ----------
try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError(
        "No se pudo importar mediapipe. Instálalo en tu venv: pip install mediapipe"
    ) from e

if not hasattr(mp, "solutions"):
    raise RuntimeError(
        "Tu 'mediapipe' NO tiene mp.solutions.\n"
        "Causas típicas:\n"
        "1) Tienes un archivo llamado 'mediapipe.py' en tu proyecto (renómbralo).\n"
        "2) Mediapipe no soporta tu versión de Python (recomendado: Python 3.10/3.11).\n"
        "Solución rápida: crea venv con Python 3.11 y reinstala requirements."
    )

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="IA Service - Hands + Face (2 faces)")

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

face = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,          # 👈 IMPORTANTE: 2 caras
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

class FrameIn(BaseModel):
    image_b64: str  # JPEG base64 (sin "data:image/...")

def b64_to_bgr(img_b64: str) -> np.ndarray:
    raw = base64.b64decode(img_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("No se pudo decodificar la imagen.")
    return bgr

def hand_landmarks(bgr: np.ndarray) -> Optional[List[Dict[str, float]]]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0].landmark
    return [{"x": float(p.x), "y": float(p.y), "z": float(p.z)} for p in lm]

def _bbox_from_landmarks(lm: List[Any]) -> Dict[str, float]:
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    x1, x2 = float(min(xs)), float(max(xs))
    y1, y2 = float(min(ys)), float(max(ys))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

def smile_score_from_facemesh(lm: List[Any]) -> Optional[float]:
    """
    Score estable:
      - mouth_width / face_width  (sonrisa suele abrir comisuras)
      - y un poquito de "apertura" para diferenciar boca neutra vs sonrisa real
    """
    # Índices FaceMesh comunes
    # boca: 61 (izq), 291 (der)
    # mejillas: 234 (izq cara), 454 (der cara)
    # labios: 13 (arriba), 14 (abajo) para apertura
    try:
        left_mouth = lm[61]
        right_mouth = lm[291]
        left_face = lm[234]
        right_face = lm[454]
        upper_lip = lm[13]
        lower_lip = lm[14]
    except Exception:
        return None

    mouth_w = np.hypot(right_mouth.x - left_mouth.x, right_mouth.y - left_mouth.y)
    face_w  = np.hypot(right_face.x - left_face.x, right_face.y - left_face.y)
    mouth_open = np.hypot(lower_lip.x - upper_lip.x, lower_lip.y - upper_lip.y)

    if face_w <= 1e-6:
        return None

    # mezcla suave (evita hiper-sensibilidad)
    score = (mouth_w / face_w) + (0.35 * mouth_open)
    return float(score)

def faces_info(bgr: np.ndarray) -> List[Dict[str, Any]]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = face.process(rgb)
    if not res.multi_face_landmarks:
        return []

    out = []
    for fl in res.multi_face_landmarks[:2]:
        lm = fl.landmark
        bbox = _bbox_from_landmarks(lm)
        score = smile_score_from_facemesh(lm)
        cx = (bbox["x1"] + bbox["x2"]) * 0.5
        out.append({"bbox": bbox, "cx": cx, "smile_score": score})

    # ordenar por posición en X (izquierda -> derecha)
    out.sort(key=lambda d: d.get("cx", 0.5))
    return out

@app.post("/infer")
def infer(frame: FrameIn) -> Dict[str, Any]:
    t0 = time.time()
    bgr = b64_to_bgr(frame.image_b64)

    hand = hand_landmarks(bgr)
    faces = faces_info(bgr)

    return {
        "hand_landmarks": hand,
        "faces": faces,  # lista de 0..2 caras, ordenadas izq->der
        "latency_ms": int((time.time() - t0) * 1000),
    }
