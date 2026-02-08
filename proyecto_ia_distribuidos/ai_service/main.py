import base64
import time
from dataclasses import dataclass
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
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# -------------------- CONFIG (ajusta si quieres) --------------------
HOLD_MS = 450            # sostiene último resultado bueno si falla detección (ms)
EMA_ALPHA = 0.35         # suavizado (0.2 más suave / 0.5 más reactivo)
BASELINE_FRAMES = 18     # ~ 1 segundo si envías ~18 fps
UNKNOWN_HOLD_MS = 300    # para RPS: si sale unknown, sostener último gesto válido un ratito
# -------------------------------------------------------------------

class FrameIn(BaseModel):
    image_b64: str  # JPEG base64 (sin "data:image/...")
    session_id: str = "default"   # 👈 ideal: único por jugador/juego
    calibrate: bool = False       # 👈 si True, recalibra baseline sonrisa (neutral)

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
    Score base (luego lo suavizamos y lo calibramos con baseline):
      - mouth_width / face_width
      - + un poco de apertura de boca
    """
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

    score = (mouth_w / face_w) + (0.30 * mouth_open)
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

    out.sort(key=lambda d: d.get("cx", 0.5))
    return out

# -------------------- RPS (más robusto) --------------------
def _dist(a: Dict[str, float], b: Dict[str, float]) -> float:
    return float(((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5)

FINGERS = {
    "index":  (5, 6, 8),   # mcp, pip, tip
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20),
}

def finger_extended(lm: List[Dict[str, float]], mcp: int, pip: int, tip: int) -> bool:
    # extendido si tip está notablemente más lejos del mcp que el pip (tolerancia 15%)
    return _dist(lm[tip], lm[mcp]) > _dist(lm[pip], lm[mcp]) * 1.15

def classify_rps(lm: Optional[List[Dict[str, float]]]) -> Optional[str]:
    if not lm:
        return None

    ext = {}
    for name, (mcp, pip, tip) in FINGERS.items():
        ext[name] = finger_extended(lm, mcp, pip, tip)

    count = sum(ext.values())

    if count <= 1:
        return "rock"
    if count >= 4:
        return "paper"

    if ext["index"] and ext["middle"] and (not ext["ring"]) and (not ext["pinky"]):
        return "scissors"

    return "unknown"

def hand_point(lm: Optional[List[Dict[str, float]]]) -> Optional[Dict[str, float]]:
    if not lm:
        return None
    # punta del índice (ideal para Fruit Ninja)
    p = lm[8]
    return {"x": float(p["x"]), "y": float(p["y"])}

# -------------------- Estado por sesión --------------------
def ema(prev: Optional[float], x: float, alpha: float = EMA_ALPHA) -> float:
    return x if prev is None else (alpha * x + (1 - alpha) * prev)

@dataclass
class SessionState:
    # hold
    last_hand: Any = None
    last_faces: Any = None
    last_ts: float = 0.0

    # rps hold
    last_rps: Optional[str] = None
    last_rps_ts: float = 0.0

    # smoothing
    ema_hand_x: Optional[float] = None
    ema_hand_y: Optional[float] = None
    ema_smile_left: Optional[float] = None
    ema_smile_right: Optional[float] = None

    # baseline (neutral)
    base_left: Optional[float] = None
    base_right: Optional[float] = None
    base_frames: int = 0
    base_sum_left: float = 0.0
    base_sum_right: float = 0.0

SESSIONS: Dict[str, SessionState] = {}

def get_state(sid: str) -> SessionState:
    st = SESSIONS.get(sid)
    if not st:
        st = SessionState()
        SESSIONS[sid] = st
    return st

def apply_smile_baseline(st: SessionState, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    faces vienen ordenadas izq->der. Aplicamos EMA y baseline por slot:
      faces[0] = izquierda, faces[1] = derecha
    """
    # recalibrar si el cliente lo pide (calibrate=True)
    # (la bandera se maneja en /infer, aquí solo aplicamos baseline ya acumulado)

    # EMA por cara
    if len(faces) >= 1 and faces[0].get("smile_score") is not None:
        st.ema_smile_left = ema(st.ema_smile_left, float(faces[0]["smile_score"]))
        faces[0]["smile_ema"] = st.ema_smile_left
    if len(faces) >= 2 and faces[1].get("smile_score") is not None:
        st.ema_smile_right = ema(st.ema_smile_right, float(faces[1]["smile_score"]))
        faces[1]["smile_ema"] = st.ema_smile_right

    # aplicar baseline si ya existe
    if len(faces) >= 1 and faces[0].get("smile_ema") is not None and st.base_left is not None:
        faces[0]["smile_level"] = max(0.0, float(faces[0]["smile_ema"]) - st.base_left)
    if len(faces) >= 2 and faces[1].get("smile_ema") is not None and st.base_right is not None:
        faces[1]["smile_level"] = max(0.0, float(faces[1]["smile_ema"]) - st.base_right)

    return faces

def update_baseline(st: SessionState, faces: List[Dict[str, Any]]):
    """
    Baseline: promedia ~1s de sonrisa neutral para cada cara.
    Se guarda en st.base_left / st.base_right.
    """
    if st.base_frames >= BASELINE_FRAMES:
        return

    left = faces[0].get("smile_ema") if len(faces) >= 1 else None
    right = faces[1].get("smile_ema") if len(faces) >= 2 else None

    if left is not None:
        st.base_sum_left += float(left)
    if right is not None:
        st.base_sum_right += float(right)

    st.base_frames += 1

    if st.base_frames >= BASELINE_FRAMES:
        # si solo hay 1 cara, base_right puede quedar None y no pasa nada
        st.base_left = (st.base_sum_left / BASELINE_FRAMES) if st.base_sum_left > 0 else None
        st.base_right = (st.base_sum_right / BASELINE_FRAMES) if st.base_sum_right > 0 else None

@app.post("/infer")
def infer(frame: FrameIn) -> Dict[str, Any]:
    t0 = time.time()
    bgr = b64_to_bgr(frame.image_b64)

    st = get_state(frame.session_id)
    now = time.time()

    # si piden recalibrar (para Smile Battle), reinicia baseline
    if frame.calibrate:
        st.base_left = None
        st.base_right = None
        st.base_frames = 0
        st.base_sum_left = 0.0
        st.base_sum_right = 0.0

    hand = hand_landmarks(bgr)
    faces = faces_info(bgr)

    # HOLD (evita parpadeo)
    if hand is None and st.last_hand is not None and (now - st.last_ts) * 1000 < HOLD_MS:
        hand = st.last_hand
    if (not faces) and st.last_faces is not None and (now - st.last_ts) * 1000 < HOLD_MS:
        faces = st.last_faces

    # guardar últimos buenos
    if hand is not None:
        st.last_hand = hand
        st.last_ts = now
    if faces:
        st.last_faces = faces
        st.last_ts = now

    # RPS
    rps = classify_rps(hand)
    if rps == "unknown":
        # sostener el último gesto válido un ratito (evita “tijeras->piedra” por 1 frame malo)
        if st.last_rps and (now - st.last_rps_ts) * 1000 < UNKNOWN_HOLD_MS:
            rps = st.last_rps
    elif rps in ("rock", "paper", "scissors"):
        st.last_rps = rps
        st.last_rps_ts = now

    # Hand point suavizado (Fruit Ninja)
    hp = hand_point(hand)
    if hp is not None:
        st.ema_hand_x = ema(st.ema_hand_x, hp["x"])
        st.ema_hand_y = ema(st.ema_hand_y, hp["y"])
        hp["x_ema"] = st.ema_hand_x
        hp["y_ema"] = st.ema_hand_y

    # Smile: EMA + baseline
    faces = apply_smile_baseline(st, faces)
    if st.base_left is None and st.base_right is None:
        # si aún no hay baseline, vamos acumulando (cuando tengas caras)
        update_baseline(st, faces)

    return {
        "hand_landmarks": hand,
        "hand_point": hp,     # 👈 útil para Fruit Ninja (usa x_ema/y_ema si están)
        "rps": rps,           # 👈 "rock" | "paper" | "scissors" | "unknown" | None
        "faces": faces,       # 👈 ahora incluye smile_ema y smile_level (si baseline listo)
        "baseline_ready": (st.base_frames >= BASELINE_FRAMES),
        "latency_ms": int((time.time() - t0) * 1000),
    }
