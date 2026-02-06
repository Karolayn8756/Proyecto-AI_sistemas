import os
import time
import base64
import random
from collections import deque

import cv2
import numpy as np

# --- HTTP ---
try:
    import requests
except Exception:
    requests = None

# =========================
# URLs
# =========================
IA_URL = "http://127.0.0.1:8001/infer"
GAME_SERVER_URL = "http://127.0.0.1:8000"
GAME_ID = "rps"

# =========================
# Ventana
# =========================
WIN = "RPS KAWAII (VS IA)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

CAM_W, CAM_H = 1280, 720
CAM_INDEX = 0

# =========================
# Match config
# =========================
BEST_OF = 5  # gana el primero en 3
IA_FPS = 10
ai_every = 1.0 / IA_FPS

# Fases por ronda (SOLO dentro del juego)
PHASE_READY = 1.2
PHASE_1 = 0.8
PHASE_2 = 0.8
PHASE_3 = 0.8
PHASE_GO = 0.35
PHASE_SHOW = 1.4

HIST_GESTURE = 20

# =========================
# Assets
# =========================
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "RPS")

IMG_MENU = "menurps.png"
IMG_RULES = "rules texto.png"
IMG_RULES_LIST = "rules rpslista.png"
IMG_AREYOUREADY = "areyouready rps.png"
IMG_LETSGO = "Lestogorps.png"
IMG_GAME = "game rps.png"
IMG_GAMEOVER = "gameover rps.png"
FONT_TTF = "font.ttf"

# Persist name
NAMES_FILE = os.path.join(os.path.dirname(__file__), "rps_name.txt")

# =========================
# Helpers
# =========================
def clamp(x, a, b):
    return max(a, min(b, x))

def load_name():
    try:
        if os.path.exists(NAMES_FILE):
            t = open(NAMES_FILE, "r", encoding="utf-8").read().strip()
            return t if t else "PLAYER"
    except Exception:
        pass
    return "PLAYER"

def save_name(name):
    try:
        open(NAMES_FILE, "w", encoding="utf-8").write(name.strip() + "\n")
    except Exception:
        pass

def load_img(name):
    path = os.path.join(ASSET_DIR, name)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img

def alpha_blit(bg_bgr, fg_rgba, x, y, w=None, h=None):
    """Pega PNG con alpha sobre fondo BGR."""
    if fg_rgba is None:
        return
    fg = fg_rgba
    if w is not None and h is not None:
        fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_AREA)

    fh, fw = fg.shape[:2]
    bh, bw = bg_bgr.shape[:2]

    x = int(x); y = int(y)
    if x >= bw or y >= bh:
        return
    x2 = min(bw, x + fw)
    y2 = min(bh, y + fh)
    if x2 <= x or y2 <= y:
        return

    fg = fg[:(y2 - y), :(x2 - x)]
    if fg.shape[2] == 3:
        bg_bgr[y:y2, x:x2] = fg
        return

    alpha = fg[:, :, 3] / 255.0
    alpha = alpha[..., None]
    fg_rgb = fg[:, :, :3]

    roi = bg_bgr[y:y2, x:x2].astype(np.float32)
    out = (alpha * fg_rgb.astype(np.float32) + (1 - alpha) * roi)
    bg_bgr[y:y2, x:x2] = out.astype(np.uint8)

def draw_text(img, text, x, y, scale=1.0, color=(255,255,255), thick=2):
    cv2.putText(img, text, (int(x), int(y)),
                cv2.FONT_HERSHEY_DUPLEX, float(scale),
                (0,0,0), int(thick)+3, cv2.LINE_AA)
    cv2.putText(img, text, (int(x), int(y)),
                cv2.FONT_HERSHEY_DUPLEX, float(scale),
                color, int(thick), cv2.LINE_AA)

def bgr_to_b64jpg(bgr, quality=85):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def post_json(url, payload, timeout=1.2):
    if requests is None:
        return None
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code != 200:
            return {"_http_error": r.status_code, "_text": r.text}
        return r.json()
    except Exception as e:
        return {"_exception": str(e)}

def send_score(player, score, extra=None):
    if requests is None:
        return
    try:
        payload = {"player": player, "game": GAME_ID, "score": int(score), "extra": extra or {}}
        requests.post(f"{GAME_SERVER_URL}/score", json=payload, timeout=1.0)
    except Exception:
        pass

# =========================
# IA parsing + hand utils
# =========================
def infer_hand_landmarks(frame_bgr):
    img_b64 = bgr_to_b64jpg(frame_bgr)
    if not img_b64:
        return None, None
    resp = post_json(IA_URL, {"image_b64": img_b64}, timeout=1.2)
    if not isinstance(resp, dict):
        return None, None
    return resp.get("hand_landmarks"), resp.get("latency_ms")

def draw_landmarks(frame_bgr, lm, color=(0, 0, 255)):
    """Dibuja puntos (como tu ejemplo). lm: lista de 21 puntos con x,y normalizados."""
    if not lm or not isinstance(lm, list):
        return
    h, w = frame_bgr.shape[:2]

    # conexiones típicas (MediaPipe Hand)
    edges = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
    ]

    pts = []
    for p in lm[:21]:
        try:
            x = int(clamp(float(p["x"]), 0.0, 1.0) * w)
            y = int(clamp(float(p["y"]), 0.0, 1.0) * h)
            pts.append((x, y))
        except Exception:
            return

    for a, b in edges:
        if a < len(pts) and b < len(pts):
            cv2.line(frame_bgr, pts[a], pts[b], (255,255,255), 2, cv2.LINE_AA)

    for (x, y) in pts:
        cv2.circle(frame_bgr, (x, y), 6, color, -1, cv2.LINE_AA)

def fingers_up(lm):
    """
    Devuelve [thumb, index, middle, ring, pinky] como boolean.
    Usamos comparaciones más robustas.
    """
    pts = [(float(p["x"]), float(p["y"]), float(p.get("z", 0.0))) for p in lm[:21]]

    def up(tip, pip):
        return pts[tip][1] < pts[pip][1] - 0.01  # margen

    # dedos
    index = up(8, 6)
    middle = up(12, 10)
    ring = up(16, 14)
    pinky = up(20, 18)

    # pulgar: separación horizontal (depende mano)
    thumb = abs(pts[4][0] - pts[2][0]) > 0.045

    return [thumb, index, middle, ring, pinky]

def classify_rps(lm):
    """
    Mejor detección de TIJERA:
    - Tijera: index y middle arriba, ring y pinky abajo
    - Papel: 4 o 5 dedos arriba
    - Piedra: 0-1 dedos arriba
    """
    if lm is None:
        return "none"
    thumb, idx, mid, ring, pinky = fingers_up(lm)

    up_count = sum([thumb, idx, mid, ring, pinky])

    # PAPEL: casi todo arriba
    if up_count >= 4:
        return "paper"

    # TIJERA: idx+mid arriba, ring+pinky abajo (thumb no importa tanto)
    if idx and mid and (not ring) and (not pinky):
        return "scissors"

    # PIEDRA: casi nada arriba
    if up_count <= 1:
        return "rock"

    # fallback
    if idx and mid:
        return "scissors"
    return "rock"

GESTURES = ["rock", "paper", "scissors"]
EMOJI = {"rock":"✊", "paper":"✋", "scissors":"✌️", "none":"—"}
NAMEG = {"rock":"PIEDRA", "paper":"PAPEL", "scissors":"TIJERA", "none":"SIN MANO"}

def decide_winner(player_move, ai_move):
    if player_move == "none":
        return "ai"
    if player_move == ai_move:
        return "tie"
    if (player_move == "rock" and ai_move == "scissors") or \
       (player_move == "paper" and ai_move == "rock") or \
       (player_move == "scissors" and ai_move == "paper"):
        return "player"
    return "ai"

# =========================
# Game state machine (FIX BUG)
# =========================
STATE_NAME = "name"
STATE_MENU = "menu"
STATE_RULES = "rules"
STATE_READY = "ready"
STATE_LETSGO = "letsgo"
STATE_PLAY = "play"
STATE_GAMEOVER = "gameover"

def needed_wins():
    return (BEST_OF // 2) + 1

def stable_gesture(hist: deque):
    if not hist:
        return "none"
    vals = [g for g in hist if g != "none"]
    if not vals:
        return "none"
    return max(set(vals), key=vals.count)

# =========================
# Main
# =========================
def main():
    # Load assets
    img_menu = load_img(IMG_MENU)
    img_rules = load_img(IMG_RULES)
    img_rules_list = load_img(IMG_RULES_LIST)
    img_ready = load_img(IMG_AREYOUREADY)
    img_letsgo = load_img(IMG_LETSGO)
    img_game = load_img(IMG_GAME)
    img_gameover = load_img(IMG_GAMEOVER)

    # Camera
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir la cámara. Revisa permisos o CAM_INDEX.")

    player_name = load_name()
    buf = player_name

    # Match vars
    wins_p = 0
    wins_ai = 0
    round_idx = 1

    gesture_hist = deque(maxlen=HIST_GESTURE)
    last_ai_send = 0.0
    last_latency_ms = None

    # Round vars
    phase = "ready"
    phase_start = 0.0
    frozen_player = None
    frozen_ai = None
    frozen_result = None
    freeze_until = 0.0  # <- FIX para que no rebote

    # State
    state = STATE_NAME

    def reset_round():
        nonlocal phase, phase_start, frozen_player, frozen_ai, frozen_result, freeze_until
        gesture_hist.clear()
        phase = "ready"
        phase_start = time.time()
        frozen_player = None
        frozen_ai = None
        frozen_result = None
        freeze_until = 0.0

    def reset_match():
        nonlocal wins_p, wins_ai, round_idx
        wins_p = 0
        wins_ai = 0
        round_idx = 1
        reset_round()

    def match_done():
        return wins_p >= needed_wins() or wins_ai >= needed_wins()

    reset_match()

    while True:
        ok, cam = cap.read()
        if not ok or cam is None:
            continue
        cam = cv2.flip(cam, 1)
        cam = cv2.resize(cam, (CAM_W, CAM_H))
        h, w = cam.shape[:2]

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        # =========================
        # STATE: NAME (tu pantalla bonita)
        # =========================
        if state == STATE_NAME:
            # usamos fondo MENU si existe, si no, un fondo oscuro
            frame = np.zeros_like(cam)
            if img_menu is not None:
                bg = cv2.resize(img_menu, (w, h), interpolation=cv2.INTER_AREA)
                if bg.shape[2] == 4:
                    frame[:] = 20
                    alpha_blit(frame, bg, 0, 0, w, h)
                else:
                    frame = bg
            else:
                frame[:] = (40, 30, 40)

            # caja input
            draw_text(frame, "Escribe tu nombre y ENTER para iniciar", int(w*0.18), int(h*0.25), scale=1.0, color=(255,255,255), thick=2)

            x1, y1 = int(w*0.18), int(h*0.33)
            x2, y2 = int(w*0.82), int(h*0.43)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), -1)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)

            draw_text(frame, f"{buf}_", x1+20, y2-20, scale=1.2, color=(255, 230, 140), thick=2)

            draw_text(frame, "ENTER iniciar | ESC salir | Backspace borrar", int(w*0.18), int(h*0.92), scale=0.9, color=(255,255,255), thick=2)

            cv2.imshow(WIN, frame)

            if key in (13, 10):
                player_name = (buf.strip() or "PLAYER")
                save_name(player_name)
                state = STATE_READY
                # prepara transición limpia
                ready_t0 = time.time()
            elif key in (8, 127):
                buf = buf[:-1]
            elif 32 <= key <= 126:
                if len(buf) < 16:
                    buf += chr(key)
            continue

        # =========================
        # STATE: READY / LETSGO (solo 1 vez)
        # =========================
        if state == STATE_READY:
            frame = np.zeros_like(cam)
            if img_ready is not None:
                bg = cv2.resize(img_ready, (w, h), interpolation=cv2.INTER_AREA)
                if bg.shape[2] == 4:
                    frame[:] = 0
                    alpha_blit(frame, bg, 0, 0, w, h)
                else:
                    frame = bg
            else:
                frame[:] = (25, 25, 25)
                draw_text(frame, "ARE YOU READY?", int(w*0.30), int(h*0.50), scale=2.0, color=(255,255,255), thick=3)

            cv2.imshow(WIN, frame)

            # pasa automático a LETSGO después de 0.9s (sin que vuelva luego)
            if time.time() - ready_t0 >= 0.9:
                state = STATE_LETSGO
                lets_t0 = time.time()
            continue

        if state == STATE_LETSGO:
            frame = np.zeros_like(cam)
            if img_letsgo is not None:
                bg = cv2.resize(img_letsgo, (w, h), interpolation=cv2.INTER_AREA)
                if bg.shape[2] == 4:
                    frame[:] = 0
                    alpha_blit(frame, bg, 0, 0, w, h)
                else:
                    frame = bg
            else:
                frame[:] = (30, 30, 30)
                draw_text(frame, "LET'S GO!", int(w*0.35), int(h*0.50), scale=2.2, color=(255,255,255), thick=3)

            cv2.imshow(WIN, frame)

            # pasa automático a PLAY después de 0.9s
            if time.time() - lets_t0 >= 0.9:
                state = STATE_PLAY
                reset_match()
            continue

        # =========================
        # STATE: PLAY (nunca vuelve a READY/LETSGO)
        # =========================
        if state == STATE_PLAY:
            # base frame: tu fondo "game"
            frame = np.zeros_like(cam)
            if img_game is not None:
                bg = cv2.resize(img_game, (w, h), interpolation=cv2.INTER_AREA)
                if bg.shape[2] == 4:
                    frame[:] = 0
                    alpha_blit(frame, bg, 0, 0, w, h)
                else:
                    frame = bg
            else:
                frame[:] = (255, 140, 200)

            # camera box centrada (para que no salga “champú” raro)
            cam_box_w = int(w * 0.62)
            cam_box_h = int(h * 0.52)
            cam_x1 = (w - cam_box_w)//2
            cam_y1 = int(h * 0.18)
            cam_x2 = cam_x1 + cam_box_w
            cam_y2 = cam_y1 + cam_box_h

            # borde
            cv2.rectangle(frame, (cam_x1-4, cam_y1-4), (cam_x2+4, cam_y2+4), (0,0,0), -1)
            cv2.rectangle(frame, (cam_x1-2, cam_y1-2), (cam_x2+2, cam_y2+2), (255,255,255), 2)

            cam_small = cv2.resize(cam, (cam_box_w, cam_box_h))
            frame[cam_y1:cam_y2, cam_x1:cam_x2] = cam_small

            # inference loop (solo si no estamos congelados)
            now = time.time()
            lm = None

            if now >= freeze_until and (now - last_ai_send >= ai_every):
                last_ai_send = now
                lm, lat = infer_hand_landmarks(cam)
                if lat is not None:
                    last_latency_ms = lat
                g = classify_rps(lm)
                gesture_hist.append(g)

            # dibujar landmarks encima de la cámara (si hay)
            if lm is not None:
                # dibujamos en cam_small -> convertimos coords normalizadas al box
                # (más simple: dibujamos sobre una copia y volvemos a pegar)
                tmp = cam_small.copy()
                draw_landmarks(tmp, lm, color=(0, 0, 255))
                frame[cam_y1:cam_y2, cam_x1:cam_x2] = tmp

            # scoreboard (colores legibles)
            draw_text(frame, f"{player_name}: {wins_p}", int(w*0.06), int(h*0.10), scale=1.25, color=(255,255,255), thick=2)
            draw_text(frame, f"Ronda {round_idx}/{BEST_OF}", int(w*0.42), int(h*0.10), scale=1.10, color=(255,255,255), thick=2)
            draw_text(frame, f"IA: {wins_ai}", int(w*0.86), int(h*0.10), scale=1.25, color=(255,255,255), thick=2)

            if last_latency_ms is not None:
                draw_text(frame, f"IA {int(last_latency_ms)}ms", int(w*0.78), int(h*0.06), scale=0.85, color=(255,255,255), thick=2)

            # fases timing
            elapsed = now - phase_start

            if now < freeze_until:
                # aún congelado, solo mostrar
                pass
            else:
                # avanzar fase normalmente
                if phase == "ready" and elapsed >= PHASE_READY:
                    phase = "one"; phase_start = now
                elif phase == "one" and elapsed >= PHASE_1:
                    phase = "two"; phase_start = now
                elif phase == "two" and elapsed >= PHASE_2:
                    phase = "three"; phase_start = now
                elif phase == "three" and elapsed >= PHASE_3:
                    phase = "go"; phase_start = now
                elif phase == "go" and elapsed >= PHASE_GO:
                    # FREEZE (FIX)
                    frozen_player = stable_gesture(gesture_hist)
                    frozen_ai = random.choice(GESTURES)
                    frozen_result = decide_winner(frozen_player, frozen_ai)

                    if frozen_result == "player":
                        wins_p += 1
                    elif frozen_result == "ai":
                        wins_ai += 1

                    phase = "show"; phase_start = now
                    freeze_until = now + PHASE_SHOW  # <- evita rebote

                elif phase == "show" and elapsed >= PHASE_SHOW:
                    # siguiente ronda / terminar
                    if match_done():
                        state = STATE_GAMEOVER
                        # guarda score solo una vez aquí
                        score = wins_p * 100
                        extra = {"wins_player": wins_p, "wins_ai": wins_ai, "best_of": BEST_OF}
                        send_score(player_name, score, extra=extra)
                    else:
                        round_idx += 1
                        reset_round()

            # overlays de fase (sin ???)
            def center_big(text, y, color):
                draw_text(frame, text, int(w*0.25), y, scale=2.0, color=color, thick=3)

            if phase == "ready":
                center_big("ALÍSTATE…", int(h*0.60), (255,255,255))
                draw_text(frame, "Pon tu mano frente a la cámara", int(w*0.28), int(h*0.67), scale=1.0, color=(255,255,255), thick=2)
            elif phase == "one":
                center_big("1  PIEDRA", int(h*0.60), (255,255,255))
            elif phase == "two":
                center_big("2  PAPEL", int(h*0.60), (255,255,255))
            elif phase == "three":
                center_big("3  TIJERA", int(h*0.60), (255,255,255))
            elif phase == "go":
                center_big("¡YA!", int(h*0.60), (255,255,255))
            elif phase == "show":
                pm = frozen_player or "none"
                am = frozen_ai or "rock"
                res = frozen_result or "tie"

                draw_text(frame, f"Tú: {EMOJI.get(pm)}  {NAMEG.get(pm)}", int(w*0.10), int(h*0.83), scale=1.2, color=(255,255,255), thick=2)
                draw_text(frame, f"IA: {EMOJI.get(am)}  {NAMEG.get(am)}", int(w*0.55), int(h*0.83), scale=1.2, color=(255,255,255), thick=2)

                if res == "player":
                    msg = f"PUNTO PARA {player_name}"
                elif res == "ai":
                    msg = "PUNTO PARA LA IA"
                else:
                    msg = "EMPATE"
                draw_text(frame, msg, int(w*0.22), int(h*0.93), scale=1.4, color=(255,255,255), thick=3)

            # controls
            draw_text(frame, "ESC salir | R reiniciar | N nombre", int(w*0.08), int(h*0.98), scale=0.85, color=(255,255,255), thick=2)

            cv2.imshow(WIN, frame)

            if key in (ord("n"), ord("N")):
                state = STATE_NAME
                buf = player_name
            if key in (ord("r"), ord("R")):
                reset_match()

            continue

        # =========================
        # STATE: GAMEOVER (tu imagen bonita)
        # =========================
        if state == STATE_GAMEOVER:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            if img_gameover is not None:
                bg = cv2.resize(img_gameover, (w, h), interpolation=cv2.INTER_AREA)
                if bg.shape[2] == 4:
                    frame[:] = 0
                    alpha_blit(frame, bg, 0, 0, w, h)
                else:
                    frame = bg
            else:
                frame[:] = (20, 20, 20)

            if wins_p > wins_ai:
                title = f"GANA {player_name}"
            elif wins_ai > wins_p:
                title = "GANA LA IA"
            else:
                title = "EMPATE"

            draw_text(frame, "RESULTADO FINAL", int(w*0.30), int(h*0.22), scale=1.4, color=(255,255,255), thick=3)
            draw_text(frame, title, int(w*0.25), int(h*0.35), scale=1.8, color=(255,255,255), thick=4)
            draw_text(frame, f"{player_name} {wins_p}  -  {wins_ai} IA", int(w*0.30), int(h*0.50), scale=1.2, color=(255,255,255), thick=3)
            draw_text(frame, f"Score guardado: {wins_p*100}", int(w*0.32), int(h*0.62), scale=1.1, color=(255,255,255), thick=2)

            draw_text(frame, "R reiniciar | N nombre | ESC salir", int(w*0.22), int(h*0.92), scale=1.0, color=(255,255,255), thick=2)

            cv2.imshow(WIN, frame)

            if key in (ord("r"), ord("R")):
                state = STATE_PLAY
                reset_match()
            if key in (ord("n"), ord("N")):
                state = STATE_NAME
                buf = player_name
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
