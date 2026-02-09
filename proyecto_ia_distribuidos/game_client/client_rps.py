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

# --- MUSICA (opcional) ---
try:
    import pygame
    PYGAME_OK = True
except Exception:
    pygame = None
    PYGAME_OK = False


# =========================
# URLs
# =========================
IA_URL = "http://127.0.0.1:8001/infer"
GAME_SERVER_URL = "http://127.0.0.1:8000"
GAME_ID = "rps"

# =========================
# Ventana
# =========================
WIN = "PIEDRA PAPEL TIJERA (VS IA)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

CAM_W, CAM_H = 1280, 720
CAM_INDEX = 0

# =========================
# Match config
# =========================
BEST_OF = 5  # gana el primero en 3
IA_FPS = 10
ai_every = 1.0 / IA_FPS

# Fases por ronda (dentro del juego)
PHASE_READY = 1.0
PHASE_1 = 0.75
PHASE_2 = 0.75
PHASE_3 = 0.75
PHASE_GO = 0.30
PHASE_SHOW = 1.25

HIST_GESTURE = 18

# =========================
# Assets
# =========================
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "RPS")

IMG_START = "RPS_inicio.png"
IMG_NAME = "RPS_NOMBRE.png"
IMG_RULES = "Reglas.png"
IMG_HANDPOS = "tu mano debe estar.png"
IMG_LISTO = "RPS_Listo.png"
IMG_GAME_BG = "camara_ai.png"          # tu fondo del juego con los marcos
IMG_GAMEOVER_BG = "RESULTADO_Final.png"

MUSIC_FILE = "rps.mp3"
BEEP_FILE = "BEEP.mp3"  # <-- ESTE ES EL QUE QUIERES

IMG_AI_ROCK = "manopiedra.png"
IMG_AI_PAPER = "manopapel.png"
IMG_AI_SCISSORS = "manotijeras.png"

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
            return t if t else ""
    except Exception:
        pass
    return ""

def save_name(name):
    try:
        open(NAMES_FILE, "w", encoding="utf-8").write(name.strip() + "\n")
    except Exception:
        pass

def load_img(name):
    path = os.path.join(ASSET_DIR, name)
    if not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

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
    # Sombra negra para legibilidad
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
# IA parsing + hand draw
# =========================
def infer_from_ai(frame_bgr, session_id="default"):
    """
    Devuelve:
      - hand_landmarks (lista de 21 o None)
      - rps ("rock"/"paper"/"scissors"/"unknown"/None)
      - latency_ms
    """
    img_b64 = bgr_to_b64jpg(frame_bgr)
    if not img_b64:
        return None, None, None

    resp = post_json(IA_URL, {"image_b64": img_b64, "session_id": session_id}, timeout=1.2)
    if not isinstance(resp, dict):
        return None, None, None

    return resp.get("hand_landmarks"), resp.get("rps"), resp.get("latency_ms")

def draw_landmarks(frame_bgr, lm, color=(0, 0, 255)):
    """Dibuja puntos + conexiones (MediaPipe Hand)."""
    if not lm or not isinstance(lm, list):
        return
    h, w = frame_bgr.shape[:2]

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


# =========================
# RPS logic
# =========================
GESTURES = ["rock", "paper", "scissors"]
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

def stable_gesture(hist: deque):
    if not hist:
        return "none"
    vals = [g for g in hist if g in ("rock", "paper", "scissors")]
    if not vals:
        return "none"
    return max(set(vals), key=vals.count)


# =========================
# States
# =========================
STATE_START = "start"
STATE_NAME = "name"
STATE_RULES = "rules"
STATE_HANDPOS = "handpos"
STATE_LISTO = "listo"
STATE_PLAY = "play"
STATE_GAMEOVER = "gameover"

def needed_wins():
    return (BEST_OF // 2) + 1


# =========================
# Music (BG + Beep)
# =========================
BEEP_CHANNEL = None

def music_start(asset_dir):
    """Arranca música de fondo con volumen moderado."""
    global BEEP_CHANNEL
    if not PYGAME_OK:
        return

    try:
        # Pre-init para menos delay en sonidos
        pygame.mixer.pre_init(44100, -16, 2, 256)
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        pygame.mixer.set_num_channels(8)
        BEEP_CHANNEL = pygame.mixer.Channel(1)

        music_path = os.path.join(asset_dir, MUSIC_FILE)
        if os.path.exists(music_path):
            pygame.mixer.music.load(music_path)
            pygame.mixer.music.set_volume(0.22)  # <-- BAJAMOS la música para oír el beep
            pygame.mixer.music.play(-1)
    except Exception:
        pass

def music_stop():
    if not PYGAME_OK:
        return
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass

def load_beep(asset_dir):
    """Carga el beep como Sound (no interrumpe la música)."""
    if not PYGAME_OK:
        return None
    try:
        beep_path = os.path.join(asset_dir, BEEP_FILE)
        if os.path.exists(beep_path):
            s = pygame.mixer.Sound(beep_path)
            s.set_volume(1.0)  # <-- BEEP FUERTE
            return s
    except Exception:
        pass
    return None

def beep_play(beep_sound):
    """Reproduce el beep 1 vez, usando canal dedicado."""
    global BEEP_CHANNEL
    if not PYGAME_OK or beep_sound is None:
        return
    try:
        if BEEP_CHANNEL is not None:
            # evita que se encime feo: corta el beep anterior y vuelve a sonar
            if BEEP_CHANNEL.get_busy():
                BEEP_CHANNEL.stop()
            BEEP_CHANNEL.play(beep_sound)
        else:
            beep_sound.play()
    except Exception:
        pass


# =========================
# Main
# =========================
def main():
    # Load assets
    img_start = load_img(IMG_START)
    img_name = load_img(IMG_NAME)
    img_rules = load_img(IMG_RULES)
    img_handpos = load_img(IMG_HANDPOS)
    img_listo = load_img(IMG_LISTO)
    img_game_bg = load_img(IMG_GAME_BG)
    img_gameover_bg = load_img(IMG_GAMEOVER_BG)

    img_ai_rock = load_img(IMG_AI_ROCK)
    img_ai_paper = load_img(IMG_AI_PAPER)
    img_ai_scissors = load_img(IMG_AI_SCISSORS)

    ai_imgs = {
        "rock": img_ai_rock,
        "paper": img_ai_paper,
        "scissors": img_ai_scissors,
    }

    # Start music + beep
    music_start(ASSET_DIR)
    beep_sound = load_beep(ASSET_DIR)

    # Camera
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir la camara. Revisa permisos o CAM_INDEX.")

    # Name
    saved = load_name()
    player_name = saved.strip()
    buf = player_name

    # Match vars
    wins_p = 0
    wins_ai = 0
    round_idx = 1

    # IA vars
    gesture_hist = deque(maxlen=HIST_GESTURE)
    last_ai_send = 0.0
    last_latency_ms = None

    # Round vars
    phase = "ready"
    phase_start = 0.0
    frozen_player = None
    frozen_ai = None
    frozen_result = None
    freeze_until = 0.0

    state = STATE_START

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

    # Para que ENTER no se repita raro (si lo mantienes apretado)
    enter_lock = False

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

        # ENTER lock
        if key in (13, 10):
            if enter_lock:
                key = 0
            else:
                enter_lock = True
        else:
            enter_lock = False

        # =========================
        # START
        # =========================
        if state == STATE_START:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            if img_start is not None:
                bg = cv2.resize(img_start, (w, h), interpolation=cv2.INTER_AREA)
                if bg.shape[2] == 4:
                    frame[:] = 0
                    alpha_blit(frame, bg, 0, 0, w, h)
                else:
                    frame = bg
            else:
                frame[:] = (30, 30, 40)
                draw_text(frame, "PIEDRA PAPEL TIJERA", int(w*0.18), int(h*0.30), 2.0)

            draw_text(frame, "ENTER CONTINUAR", int(w*0.36), int(h*0.82), 1.2)
            draw_text(frame, "ESC SALIR", int(w*0.42), int(h*0.90), 1.0)

            cv2.imshow(WIN, frame)

            if key in (13, 10):
                state = STATE_NAME
                if not buf:
                    buf = ""
            continue

        # =========================
        # NAME (obligatoria)
        # =========================
        if state == STATE_NAME:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            if img_name is not None:
                bg = cv2.resize(img_name, (w, h), interpolation=cv2.INTER_AREA)
                if bg.shape[2] == 4:
                    frame[:] = 0
                    alpha_blit(frame, bg, 0, 0, w, h)
                else:
                    frame = bg
            else:
                frame[:] = (60, 50, 70)

            x1, y1 = int(w*0.20), int(h*0.44)
            x2, y2 = int(w*0.80), int(h*0.54)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)

            draw_text(frame, "ESCRIBE TU NOMBRE (OBLIGATORIO)", int(w*0.20), int(h*0.36), 1.0)
            draw_text(frame, f"{buf}_", x1 + 18, y2 - 18, 1.4, (255, 230, 140), 2)

            draw_text(frame, "ENTER CONTINUAR | Q INICIO | ESC SALIR", int(w*0.18), int(h*0.92), 0.9)

            if not buf.strip():
                draw_text(frame, "NO PUEDE ESTAR VACIO", int(w*0.34), int(h*0.62), 1.0, (0, 70, 255), 2)

            cv2.imshow(WIN, frame)

            if key in (ord("q"), ord("Q")):
                state = STATE_START
            elif key in (8, 127):
                buf = buf[:-1]
            elif key in (13, 10):
                if buf.strip():
                    player_name = buf.strip()
                    save_name(player_name)
                    state = STATE_RULES
            elif 32 <= key <= 126:
                if len(buf) < 16:
                    buf += chr(key)
            continue

        # =========================
        # RULES
        # =========================
        if state == STATE_RULES:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            if img_rules is not None:
                bg = cv2.resize(img_rules, (w, h), interpolation=cv2.INTER_AREA)
                if bg.shape[2] == 4:
                    frame[:] = 0
                    alpha_blit(frame, bg, 0, 0, w, h)
                else:
                    frame = bg
            else:
                frame[:] = (20, 20, 20)
                draw_text(frame, "REGLAS", int(w*0.42), int(h*0.25), 2.0)

            draw_text(frame, "ENTER CONTINUAR | Q INICIO", int(w*0.30), int(h*0.92), 1.0)
            cv2.imshow(WIN, frame)

            if key in (ord("q"), ord("Q")):
                state = STATE_START
            elif key in (13, 10):
                state = STATE_HANDPOS
            continue

        # =========================
        # HAND POSITION
        # =========================
        if state == STATE_HANDPOS:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            if img_handpos is not None:
                bg = cv2.resize(img_handpos, (w, h), interpolation=cv2.INTER_AREA)
                if bg.shape[2] == 4:
                    frame[:] = 0
                    alpha_blit(frame, bg, 0, 0, w, h)
                else:
                    frame = bg
            else:
                frame[:] = (10, 10, 10)
                draw_text(frame, "MANO APUNTANDO HACIA ARRIBA", int(w*0.16), int(h*0.45), 1.4)

            draw_text(frame, "ENTER CONTINUAR | Q INICIO", int(w*0.30), int(h*0.92), 1.0)
            cv2.imshow(WIN, frame)

            if key in (ord("q"), ord("Q")):
                state = STATE_START
            elif key in (13, 10):
                state = STATE_LISTO
            continue

        # =========================
        # LISTO
        # =========================
        if state == STATE_LISTO:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            if img_listo is not None:
                bg = cv2.resize(img_listo, (w, h), interpolation=cv2.INTER_AREA)
                if bg.shape[2] == 4:
                    frame[:] = 0
                    alpha_blit(frame, bg, 0, 0, w, h)
                else:
                    frame = bg
            else:
                frame[:] = (40, 40, 40)
                draw_text(frame, "LISTO?", int(w*0.42), int(h*0.45), 2.4)

            draw_text(frame, "ENTER JUGAR | Q INICIO", int(w*0.34), int(h*0.92), 1.0)
            cv2.imshow(WIN, frame)

            if key in (ord("q"), ord("Q")):
                state = STATE_START
            elif key in (13, 10):
                reset_match()
                state = STATE_PLAY
            continue

        # =========================
        # PLAY
        # =========================
        if state == STATE_PLAY:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            if img_game_bg is not None:
                bg = cv2.resize(img_game_bg, (w, h), interpolation=cv2.INTER_AREA)
                if bg.shape[2] == 4:
                    frame[:] = 0
                    alpha_blit(frame, bg, 0, 0, w, h)
                else:
                    frame = bg
            else:
                frame[:] = (190, 200, 255)

            # Layout boxes
            cam_box_w = int(w * 0.63)
            cam_box_h = int(h * 0.54)
            cam_x1 = int(w * 0.08)
            cam_y1 = int(h * 0.17)
            cam_x2 = cam_x1 + cam_box_w
            cam_y2 = cam_y1 + cam_box_h

            ai_box_w = int(w * 0.22)
            ai_box_h = cam_box_h
            ai_x1 = cam_x2 + int(w * 0.03)
            ai_y1 = cam_y1
            ai_x2 = ai_x1 + ai_box_w
            ai_y2 = ai_y1 + ai_box_h

            # Bordes
            cv2.rectangle(frame, (cam_x1-3, cam_y1-3), (cam_x2+3, cam_y2+3), (0,0,0), -1)
            cv2.rectangle(frame, (cam_x1-1, cam_y1-1), (cam_x2+1, cam_y2+1), (255,255,255), 2)

            cv2.rectangle(frame, (ai_x1-3, ai_y1-3), (ai_x2+3, ai_y2+3), (0,0,0), -1)
            cv2.rectangle(frame, (ai_x1-1, ai_y1-1), (ai_x2+1, ai_y2+1), (255,255,255), 2)

            # Pegar camara
            cam_small = cv2.resize(cam, (cam_box_w, cam_box_h))
            frame[cam_y1:cam_y2, cam_x1:cam_x2] = cam_small

            now = time.time()
            lm = None
            rps_ai = None

            sid = f"rps_{player_name}"

            # Inference
            if now >= freeze_until and (now - last_ai_send >= ai_every):
                last_ai_send = now
                lm, rps_ai, lat = infer_from_ai(cam, session_id=sid)
                if lat is not None:
                    last_latency_ms = lat

                if rps_ai not in ("rock", "paper", "scissors"):
                    rps_ai = None
                gesture_hist.append(rps_ai if rps_ai else "none")

            # Dibuja landmarks
            if lm is not None:
                tmp = frame[cam_y1:cam_y2, cam_x1:cam_x2].copy()
                draw_landmarks(tmp, lm, color=(0, 0, 255))
                frame[cam_y1:cam_y2, cam_x1:cam_x2] = tmp

            # HUD
            draw_text(frame, f"{player_name}: {wins_p}", int(w*0.06), int(h*0.10), 1.25)
            draw_text(frame, f"RONDA {round_idx}/{BEST_OF}", int(w*0.42), int(h*0.10), 1.10)
            draw_text(frame, f"IA: {wins_ai}", int(w*0.86), int(h*0.10), 1.25)

            if last_latency_ms is not None:
                draw_text(frame, f"{int(last_latency_ms)}ms", int(w*0.90), int(h*0.06), 0.9)

            current_player = stable_gesture(gesture_hist)
            draw_text(frame, f"{player_name}: {NAMEG.get(current_player,'SIN MANO')}",
                      cam_x1 + 10, cam_y1 + 28, 0.9)

            draw_text(frame, "IA ELIGIO", ai_x1 + 8, ai_y1 - 10, 0.9)

            # --------- CONTEO + BEEP (ARREGLADO) ----------
            elapsed = now - phase_start

            if now >= freeze_until:
                if phase == "ready" and elapsed >= PHASE_READY:
                    phase = "one"; phase_start = now
                    beep_play(beep_sound)
                elif phase == "one" and elapsed >= PHASE_1:
                    phase = "two"; phase_start = now
                    beep_play(beep_sound)
                elif phase == "two" and elapsed >= PHASE_2:
                    phase = "three"; phase_start = now
                    beep_play(beep_sound)
                elif phase == "three" and elapsed >= PHASE_3:
                    phase = "go"; phase_start = now
                    beep_play(beep_sound)
                elif phase == "go" and elapsed >= PHASE_GO:
                    frozen_player = stable_gesture(gesture_hist)
                    frozen_ai = random.choice(GESTURES)
                    frozen_result = decide_winner(frozen_player, frozen_ai)

                    if frozen_result == "player":
                        wins_p += 1
                    elif frozen_result == "ai":
                        wins_ai += 1

                    phase = "show"
                    phase_start = now
                    freeze_until = now + PHASE_SHOW

                elif phase == "show" and elapsed >= PHASE_SHOW:
                    if match_done():
                        state = STATE_GAMEOVER
                        score = wins_p * 100
                        extra = {"wins_player": wins_p, "wins_ai": wins_ai, "best_of": BEST_OF}
                        send_score(player_name, score, extra=extra)
                    else:
                        round_idx += 1
                        reset_round()

            # IA image (solo en show)
            if phase == "show" and frozen_ai in ("rock", "paper", "scissors") and ai_imgs.get(frozen_ai) is not None:
                alpha_blit(frame, ai_imgs[frozen_ai], ai_x1, ai_y1, ai_box_w, ai_box_h)

            def center_big(text, y):
                draw_text(frame, text, int(w*0.33), y, 2.0)

            if phase == "ready":
                center_big("ALISTATE", int(h*0.60))
            elif phase == "one":
                center_big("1  PIEDRA", int(h*0.60))
            elif phase == "two":
                center_big("2  PAPEL", int(h*0.60))
            elif phase == "three":
                center_big("3  TIJERA", int(h*0.60))
            elif phase == "go":
                center_big("YA!", int(h*0.60))
            elif phase == "show":
                if frozen_result == "player":
                    msg = f"PUNTO PARA {player_name}"
                elif frozen_result == "ai":
                    msg = "PUNTO PARA LA IA"
                else:
                    msg = "EMPATE"

                cv2.rectangle(frame, (int(w*0.10), int(h*0.76)), (int(w*0.90), int(h*0.83)), (0,0,0), -1)
                cv2.rectangle(frame, (int(w*0.10), int(h*0.76)), (int(w*0.90), int(h*0.83)), (255,255,255), 2)
                    #draw_text(frame, msg, int(w*0.22), int(h*0.81), 1.35)

            cv2.imshow(WIN, frame)

            if key in (ord("q"), ord("Q")):
                state = STATE_START
            elif key in (ord("n"), ord("N")):
                state = STATE_NAME
                buf = player_name
            elif key in (ord("r"), ord("R")):
                reset_match()

            continue

        # =========================
        # GAME OVER
        # =========================
        if state == STATE_GAMEOVER:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            if img_gameover_bg is not None:
                bg = cv2.resize(img_gameover_bg, (w, h), interpolation=cv2.INTER_AREA)
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

            total_score = wins_p * 100

            draw_text(frame, title, int(w*0.25), int(h*0.35), 2.0)
            draw_text(frame, f"{player_name} {wins_p}  -  {wins_ai} IA", int(w*0.30), int(h*0.50), 1.3)
            draw_text(frame, f"TOTAL SCORE: {total_score}", int(w*0.33), int(h*0.62), 1.3)

            draw_text(frame, "R REINICIAR | Q INICIO | N NOMBRE | ESC SALIR", int(w*0.17), int(h*0.92), 1.0)

            cv2.imshow(WIN, frame)

            if key in (ord("q"), ord("Q")):
                state = STATE_START
            elif key in (ord("n"), ord("N")):
                state = STATE_NAME
                buf = player_name
            elif key in (ord("r"), ord("R")):
                reset_match()
                state = STATE_PLAY

            continue

    cap.release()
    cv2.destroyAllWindows()
    music_stop()

if __name__ == "__main__":
    main()
