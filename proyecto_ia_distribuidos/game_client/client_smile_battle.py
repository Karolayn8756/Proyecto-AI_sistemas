import os
import sys
import time
import json
import math
import base64
import traceback
from dataclasses import dataclass

import cv2
import numpy as np
import pygame

# --- HTTP ---
try:
    import requests
except Exception:
    requests = None


# =========================
# Config
# =========================
WIN_W, WIN_H = 1280, 720
FPS = 60

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "smile_battle")

IMG_MENU = "menu.png"
IMG_RULES_TEXT = "reglas escritas.png"
IMG_READY = "Are you ready.png"
IMG_LETSGO = "Lets go.png"
IMG_GAMEOVER = "Game Over.png"
IMG_CAM_BG = "camaras_fondo.png"
FONT_TTF = "font.ttf"

# Gameplay
ROUND_SECONDS = 60

# Calibración (solo avanza si hay 2 caras)
CALIB_SECONDS = 2.0
CALIB_MIN_SAMPLES = 20  # mínimo muestras por jugador

# Smile logic
SMILE_LIMIT_PERCENT = 60          # si pasa esto, comienza peligro
SMILE_HOLD_LOSE_SECONDS = 0.70    # mantenerlo este tiempo = pierde

# Camera
CAM_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# Distributed services
IA_URL = "http://127.0.0.1:8001/infer"
GAME_SERVER_URL = "http://127.0.0.1:8000"
SCORE_ENDPOINT = "/score"   # <- existe en tu server
GAME_ID = "smile_battle"


# =========================
# Helpers
# =========================
def clamp(x, a, b):
    return max(a, min(b, x))

def load_image(name):
    path = os.path.join(ASSET_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"NO ENCONTRÉ el archivo: {path}")
    return pygame.image.load(path).convert_alpha()

def load_font(size):
    path = os.path.join(ASSET_DIR, FONT_TTF)
    if os.path.exists(path):
        return pygame.font.Font(path, size)
    return pygame.font.SysFont("Arial", size)

def crop_face(frame_bgr, bbox_px, pad=0.35):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_px
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2
    cy = y1 + bh / 2

    bw2 = bw * (1 + pad)
    bh2 = bh * (1 + pad)

    nx1 = int(clamp(cx - bw2 / 2, 0, w - 1))
    ny1 = int(clamp(cy - bh2 / 2, 0, h - 1))
    nx2 = int(clamp(cx + bw2 / 2, 1, w))
    ny2 = int(clamp(cy + bh2 / 2, 1, h))

    if ny2 <= ny1 or nx2 <= nx1:
        return None

    return frame_bgr[ny1:ny2, nx1:nx2].copy()

def cv_bgr_to_pygame_surface(frame_bgr, target_size):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
    surf = pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], "RGB")
    return surf.convert()

def make_placeholder_surface(size, title="PLAYER", subtitle="Esperando cara..."):
    w, h = size
    surf = pygame.Surface((w, h)).convert()
    surf.fill((45, 45, 45))
    pygame.draw.rect(surf, (230, 230, 230), (0, 0, w, h), 3)

    f1 = pygame.font.SysFont("Arial", 28, bold=True)
    f2 = pygame.font.SysFont("Arial", 18)

    t1 = f1.render(title, True, (255, 255, 255))
    t2 = f2.render(subtitle, True, (210, 210, 210))

    surf.blit(t1, (w//2 - t1.get_width()//2, h//2 - 35))
    surf.blit(t2, (w//2 - t2.get_width()//2, h//2 + 5))
    return surf

def draw_bar(screen, x, y, w, h, percent, border=3):
    pygame.draw.rect(screen, (255, 255, 255), (x, y, w, h), border)
    fill_w = int((w - border*2) * clamp(percent, 0, 100) / 100)
    pygame.draw.rect(screen, (210, 140, 255), (x + border, y + border, fill_w, h - border*2))

def bgr_to_jpg_b64(frame_bgr, quality=70):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
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


# =========================
# IA parsing
# =========================
@dataclass
class FaceInfo:
    bbox_px: tuple   # (x1,y1,x2,y2) en pixeles
    smile_score: float
    center_x: float  # para ordenar izq->der

def parse_faces_from_ia(resp, frame_w, frame_h):
    """
    resp esperado:
      {
        "faces": [
          {"bbox": {"x1":0..1,"y1":0..1,"x2":0..1,"y2":0..1}, "cx":0..1, "smile_score": float},
          ...
        ]
      }
    """
    faces_out = []
    if not isinstance(resp, dict):
        return faces_out
    faces = resp.get("faces", None)
    if not isinstance(faces, list):
        return faces_out

    for f in faces[:2]:
        bbox = (f or {}).get("bbox", {})
        try:
            x1n = float(bbox.get("x1", 0.0))
            y1n = float(bbox.get("y1", 0.0))
            x2n = float(bbox.get("x2", 0.0))
            y2n = float(bbox.get("y2", 0.0))
            x1 = int(clamp(x1n, 0.0, 1.0) * frame_w)
            y1 = int(clamp(y1n, 0.0, 1.0) * frame_h)
            x2 = int(clamp(x2n, 0.0, 1.0) * frame_w)
            y2 = int(clamp(y2n, 0.0, 1.0) * frame_h)
            # asegura tamaño mínimo
            x2 = max(x2, x1 + 2)
            y2 = max(y2, y1 + 2)
            smile_score = f.get("smile_score", None)
            if smile_score is None:
                continue
            smile_score = float(smile_score)
            cx = float(f.get("cx", (x1n + x2n) * 0.5))
            faces_out.append(FaceInfo(bbox_px=(x1, y1, x2, y2), smile_score=smile_score, center_x=cx))
        except Exception:
            continue

    faces_out.sort(key=lambda ff: ff.center_x)  # izquierda -> derecha
    return faces_out


# =========================
# Names
# =========================
def load_player_names():
    root = os.path.dirname(__file__)
    candidates = [
        os.path.join(root, "players_smile_2p.txt"),
        os.path.join(root, "player_name.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    lines = [x.strip() for x in f.readlines() if x.strip()]
                if len(lines) >= 2:
                    return lines[0], lines[1]
                if len(lines) == 1:
                    return lines[0], "player2"
            except Exception:
                pass
    return "player1", "player2"


# =========================
# Smile normalize (baselines)
# =========================
class SmileNormalizer:
    def __init__(self):
        self.base_left = None
        self.base_right = None

    def set_baselines(self, base_left, base_right):
        self.base_left = base_left
        self.base_right = base_right

    def percent(self, score, side):
        base = self.base_left if side == "left" else self.base_right
        if base is None:
            return 0.0

        # Sensibilidad: sube si marca muy alto; baja si no detecta
        sensitivity = 0.22
        pct = ((score - base) / (base * sensitivity)) * 100.0
        return clamp(pct, 0.0, 150.0)


# =========================
# Dashboard send
# =========================
def send_record_to_server(winner, loser, reason, avg_left, avg_right, duration_sec, name_left, name_right):
    """
    Envía al Game Server (POST /score) el ganador como player.
    score = 100 - promedio sonrisa del ganador (menos sonrisa = más puntos)
    """
    if requests is None:
        return

    url = GAME_SERVER_URL.rstrip("/") + SCORE_ENDPOINT  # ✅ /score

    if winner == "tie":
        return

    # Decide promedio del ganador según si es izquierda/derecha
    if winner == name_left:
        winner_avg = avg_left
    elif winner == name_right:
        winner_avg = avg_right
    else:
        # fallback por si el nombre cambió raro
        winner_avg = min(avg_left, avg_right)

    score = int(max(0, min(100, round(100 - float(winner_avg)))))

    payload = {
        "player": str(winner),
        "game": str(GAME_ID),
        "score": int(score),
        "extra": {
            "loser": str(loser),
            "reason": str(reason),
            "avg_smile_left": round(float(avg_left), 2),
            "avg_smile_right": round(float(avg_right), 2),
            "duration_sec": int(duration_sec),
            "ts": int(time.time()),
        }
    }

    try:
        r = requests.post(url, json=payload, timeout=2.0)
        if r.status_code >= 400:
            print("[SmileBattle] ERROR /score:", r.status_code, r.text)
    except Exception as e:
        print("[SmileBattle] EXCEPTION enviando score:", e)



# =========================
# States
# =========================
STATE_MENU = "menu"
STATE_RULES = "rules"
STATE_READY = "ready"
STATE_LETSGO = "letsgo"
STATE_CALIB = "calib"
STATE_PLAY = "play"
STATE_GAMEOVER = "gameover"


def main():
    pygame.init()
    try:
        pygame.mixer.init()
    except Exception:
        pass

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Smile Battle")
    clock = pygame.time.Clock()

    menu_img = pygame.transform.smoothscale(load_image(IMG_MENU), (WIN_W, WIN_H))
    rules_img = pygame.transform.smoothscale(load_image(IMG_RULES_TEXT), (WIN_W, WIN_H))
    ready_img = pygame.transform.smoothscale(load_image(IMG_READY), (WIN_W, WIN_H))
    letsgo_img = pygame.transform.smoothscale(load_image(IMG_LETSGO), (WIN_W, WIN_H))
    gameover_img = pygame.transform.smoothscale(load_image(IMG_GAMEOVER), (WIN_W, WIN_H))
    cam_bg = pygame.transform.smoothscale(load_image(IMG_CAM_BG), (WIN_W, WIN_H))

    font_big = load_font(64)
    font_mid = load_font(36)
    font_small = load_font(26)

    # ONE camera
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("No pude abrir la cámara. Revisa permisos o CAM_INDEX.")

    name_left, name_right = load_player_names()
    normalizer = SmileNormalizer()

    # UI rects
    BUBBLE_LEFT = pygame.Rect(int(WIN_W*0.08), int(WIN_H*0.12), int(WIN_W*0.36), int(WIN_H*0.42))
    BUBBLE_RIGHT = pygame.Rect(int(WIN_W*0.56), int(WIN_H*0.12), int(WIN_W*0.36), int(WIN_H*0.42))

    BAR_LEFT = pygame.Rect(int(WIN_W*0.12), int(WIN_H*0.70), int(WIN_W*0.32), int(WIN_H*0.05))
    BAR_RIGHT = pygame.Rect(int(WIN_W*0.56), int(WIN_H*0.70), int(WIN_W*0.32), int(WIN_H*0.05))

    timer_pos = (WIN_W//2, int(WIN_H*0.08))

    ph_left = make_placeholder_surface((BUBBLE_LEFT.w, BUBBLE_LEFT.h), "PLAYER 1", "Acércate a la cámara")
    ph_right = make_placeholder_surface((BUBBLE_RIGHT.w, BUBBLE_RIGHT.h), "PLAYER 2", "Acércate a la cámara")

    state = STATE_MENU
    running = True
    paused = False

    # round vars
    round_left_hold = 0.0
    round_right_hold = 0.0
    left_sum = 0.0
    right_sum = 0.0
    sample_count = 0

    start_time = None
    time_left = ROUND_SECONDS

    # calib vars
    calib_active = False
    calib_t0 = None
    base_left_samples = []
    base_right_samples = []

    # result
    winner = None
    loser = None
    reason = None
    sent_record = False

    def reset_round():
        nonlocal round_left_hold, round_right_hold, left_sum, right_sum, sample_count
        nonlocal start_time, time_left, paused, winner, loser, reason, sent_record
        round_left_hold = 0.0
        round_right_hold = 0.0
        left_sum = 0.0
        right_sum = 0.0
        sample_count = 0
        start_time = time.time()
        time_left = ROUND_SECONDS
        paused = False
        winner = None
        loser = None
        reason = None
        sent_record = False

    def start_calibration():
        nonlocal calib_active, calib_t0, base_left_samples, base_right_samples, paused
        calib_active = True
        calib_t0 = None
        base_left_samples = []
        base_right_samples = []
        paused = False

    def decide_winner_time_end():
        nonlocal winner, loser, reason
        avg_l = left_sum / max(1, sample_count)
        avg_r = right_sum / max(1, sample_count)
        if avg_l < avg_r:
            winner, loser = name_left, name_right
        elif avg_r < avg_l:
            winner, loser = name_right, name_left
        else:
            winner, loser = "tie", "tie"
        reason = "time_end"
        return avg_l, avg_r

    def decide_winner_smile_hold(loser_side):
        nonlocal winner, loser, reason
        if loser_side == "left":
            loser = name_left
            winner = name_right
        else:
            loser = name_right
            winner = name_left
        reason = "smile_hold"

    def draw_face_or_placeholder(face: FaceInfo, bubble_rect, placeholder_surf, frame_bgr):
        if face is None:
            screen.blit(placeholder_surf, (bubble_rect.x, bubble_rect.y))
            return
        crop = crop_face(frame_bgr, face.bbox_px, pad=0.40)
        if crop is None:
            screen.blit(placeholder_surf, (bubble_rect.x, bubble_rect.y))
            return
        surf = cv_bgr_to_pygame_surface(crop, (bubble_rect.w, bubble_rect.h))
        screen.blit(surf, (bubble_rect.x, bubble_rect.y))

    # ------------- LOOP -------------
    while running:
        dt = clock.tick(FPS) / 1000.0

        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if state == STATE_MENU:
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        state = STATE_RULES

                elif state == STATE_RULES:
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        state = STATE_READY

                elif state == STATE_READY:
                    if event.key == pygame.K_RETURN:
                        state = STATE_LETSGO
                    if event.key == pygame.K_BACKSPACE:
                        state = STATE_MENU

                elif state == STATE_LETSGO:
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        state = STATE_CALIB
                        start_calibration()

                elif state in (STATE_CALIB, STATE_PLAY):
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    if event.key == pygame.K_r:
                        state = STATE_CALIB
                        start_calibration()
                    if event.key == pygame.K_n:
                        name_left, name_right = load_player_names()

                elif state == STATE_GAMEOVER:
                    if event.key == pygame.K_r:
                        state = STATE_CALIB
                        start_calibration()
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        state = STATE_MENU

        # menu/rules/ready/letsgo
        if state == STATE_MENU:
            screen.blit(menu_img, (0, 0))
            pygame.display.flip()
            continue

        if state == STATE_RULES:
            screen.blit(rules_img, (0, 0))
            hint = font_small.render("ENTER / SPACE = continuar", True, (255, 255, 255))
            screen.blit(hint, (20, WIN_H - 40))
            pygame.display.flip()
            continue

        if state == STATE_READY:
            screen.blit(ready_img, (0, 0))
            hint = font_small.render("ENTER = Yes   |   BACKSPACE = No", True, (255, 255, 255))
            screen.blit(hint, (20, WIN_H - 40))
            pygame.display.flip()
            continue

        if state == STATE_LETSGO:
            screen.blit(letsgo_img, (0, 0))
            hint = font_small.render("ENTER / SPACE = iniciar", True, (255, 255, 255))
            screen.blit(hint, (20, WIN_H - 40))
            pygame.display.flip()
            continue

        # -------- read ONE camera frame --------
        ret, frame = cap.read()
        if not ret or frame is None:
            screen.fill((10, 0, 10))
            err = font_mid.render("No se pudo leer la cámara", True, (255, 80, 80))
            screen.blit(err, (40, 40))
            pygame.display.flip()
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

        # -------- send frame to IA --------
        faces = []
        b64 = bgr_to_jpg_b64(frame, quality=70)
        if b64:
            resp = post_json(IA_URL, {"image_b64": b64}, timeout=1.2)
            faces = parse_faces_from_ia(resp, CAM_WIDTH, CAM_HEIGHT)

        face_left = faces[0] if len(faces) >= 1 else None
        face_right = faces[1] if len(faces) >= 2 else None

        # -------- background + faces (never black) --------
        screen.blit(cam_bg, (0, 0))
        draw_face_or_placeholder(face_left, BUBBLE_LEFT, ph_left, frame)
        draw_face_or_placeholder(face_right, BUBBLE_RIGHT, ph_right, frame)

        have_two = (face_left is not None and face_right is not None)

        # -------- CALIB state --------
        if state == STATE_CALIB:
            # Solo inicia el timer de calibración cuando hay 2 caras
            if not have_two:
                calib_t0 = None
                base_left_samples.clear()
                base_right_samples.clear()

                txt = font_mid.render("CALIBRANDO... (cara neutra)", True, (255, 255, 255))
                screen.blit(txt, (40, 40))
                warn = font_small.render("Se necesitan 2 caras en cámara para calibrar", True, (255, 240, 120))
                screen.blit(warn, (40, 80))
                draw_bar(screen, 40, 110, 420, 30, 0)
                pygame.display.flip()
                continue

            # ya hay 2 caras
            if calib_t0 is None:
                calib_t0 = time.time()

            # acumula muestras
            base_left_samples.append(face_left.smile_score)
            base_right_samples.append(face_right.smile_score)

            elapsed = time.time() - calib_t0
            prog = int(clamp((elapsed / CALIB_SECONDS) * 100, 0, 100))

            txt = font_mid.render("CALIBRANDO... (cara neutra)", True, (255, 255, 255))
            screen.blit(txt, (40, 40))
            draw_bar(screen, 40, 90, 420, 30, prog)

            # termina calibración
            if elapsed >= CALIB_SECONDS and len(base_left_samples) >= CALIB_MIN_SAMPLES and len(base_right_samples) >= CALIB_MIN_SAMPLES:
                base_l = float(np.median(np.array(base_left_samples)))
                base_r = float(np.median(np.array(base_right_samples)))
                normalizer.set_baselines(base_l, base_r)

                reset_round()
                state = STATE_PLAY
                paused = False

            pygame.display.flip()
            continue

        # -------- PLAY state --------
        if state == STATE_PLAY:
            # si falta cara, pausa (pero no negro)
            if not have_two:
                paused = True
            if paused:
                msg = font_mid.render("PAUSA: se necesitan 2 caras en cámara", True, (255, 240, 120))
                screen.blit(msg, (40, 40))
                hint = font_small.render("SPACE=continuar  |  R=reiniciar  |  ESC=salir", True, (255, 255, 255))
                screen.blit(hint, (40, 80))
                pygame.display.flip()
                continue

            # percents
            left_pct = normalizer.percent(face_left.smile_score, "left")
            right_pct = normalizer.percent(face_right.smile_score, "right")

            # timer
            elapsed_round = time.time() - start_time
            time_left = int(max(0, ROUND_SECONDS - elapsed_round))

            # averages
            left_sum += left_pct
            right_sum += right_pct
            sample_count += 1

            # hold logic
            if left_pct >= SMILE_LIMIT_PERCENT:
                round_left_hold += dt
            else:
                round_left_hold = max(0.0, round_left_hold - dt * 1.5)

            if right_pct >= SMILE_LIMIT_PERCENT:
                round_right_hold += dt
            else:
                round_right_hold = max(0.0, round_right_hold - dt * 1.5)

            # labels + bars
            lbl1 = font_small.render(f"{name_left}  Smile: {int(left_pct)}%", True, (255, 255, 255))
            lbl2 = font_small.render(f"{name_right}  Smile: {int(right_pct)}%", True, (255, 255, 255))
            screen.blit(lbl1, (BAR_LEFT.x, BAR_LEFT.y - 30))
            screen.blit(lbl2, (BAR_RIGHT.x, BAR_RIGHT.y - 30))

            draw_bar(screen, BAR_LEFT.x, BAR_LEFT.y, BAR_LEFT.w, BAR_LEFT.h, clamp(left_pct, 0, 100))
            draw_bar(screen, BAR_RIGHT.x, BAR_RIGHT.y, BAR_RIGHT.w, BAR_RIGHT.h, clamp(right_pct, 0, 100))

            timer_text = font_big.render(str(time_left), True, (255, 255, 255))
            timer_rect = timer_text.get_rect(center=timer_pos)
            screen.blit(timer_text, timer_rect)

            hint = font_small.render("R=Restart   N=Names   SPACE=Pause   ESC=Exit", True, (255, 255, 255))
            screen.blit(hint, (int(WIN_W*0.12), int(WIN_H*0.90)))

            # decide gameover
            if round_left_hold >= SMILE_HOLD_LOSE_SECONDS:
                decide_winner_smile_hold("left")
                state = STATE_GAMEOVER
                sent_record = False
            elif round_right_hold >= SMILE_HOLD_LOSE_SECONDS:
                decide_winner_smile_hold("right")
                state = STATE_GAMEOVER
                sent_record = False
            elif time_left <= 0:
                decide_winner_time_end()
                state = STATE_GAMEOVER
                sent_record = False

            pygame.display.flip()
            continue

        # -------- GAMEOVER --------
        if state == STATE_GAMEOVER:
            screen.blit(gameover_img, (0, 0))

            avg_l = left_sum / max(1, sample_count)
            avg_r = right_sum / max(1, sample_count)

            xL = int(WIN_W * 0.06)
            xR = int(WIN_W * 0.72)
            yTop = int(WIN_H * 0.22)

            left_status = "WIN" if winner == name_left else "FAILED"
            right_status = "WIN" if winner == name_right else "FAILED"
            if winner == "tie":
                left_status = "TIE"
                right_status = "TIE"

            stL = font_big.render(left_status, True, (255, 255, 255))
            stR = font_big.render(right_status, True, (255, 255, 255))
            nmL = font_mid.render(name_left, True, (210, 255, 160))
            nmR = font_mid.render(name_right, True, (210, 255, 160))

            screen.blit(nmL, (xL, yTop))
            screen.blit(stL, (xL, yTop + 55))
            screen.blit(nmR, (xR, yTop))
            screen.blit(stR, (xR, yTop + 55))

            a1 = font_small.render(f"Avg Smile: {avg_l:.1f}%", True, (255, 255, 255))
            a2 = font_small.render(f"Avg Smile: {avg_r:.1f}%", True, (255, 255, 255))
            screen.blit(a1, (xL, yTop + 130))
            screen.blit(a2, (xR, yTop + 130))

            rs = "Perdiste por sostener sonrisa" if reason == "smile_hold" else "Tiempo terminado"
            rr = font_small.render(rs, True, (255, 240, 120))
            screen.blit(rr, (int(WIN_W*0.06), int(WIN_H*0.12)))

            hint = font_small.render("R=Reiniciar  |  ENTER/SPACE=Menu  |  ESC=Salir", True, (255, 255, 255))
            screen.blit(hint, (int(WIN_W*0.06), int(WIN_H*0.90)))

            if not sent_record and start_time is not None:
                duration = time.time() - start_time
                if winner != "tie":
                    send_record_to_server(
                        winner=winner,
                        loser=loser,
                        reason=reason,
                        avg_left=avg_l,
                        avg_right=avg_r,
                        duration_sec=duration,
                        name_left=name_left,
                        name_right=name_right,
                    )

                sent_record = True

            pygame.display.flip()
            continue

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR] Algo falló. Detalle:\n")
        print(str(e))
        print("\n--- Traceback ---\n")
        traceback.print_exc()
        sys.exit(1)
