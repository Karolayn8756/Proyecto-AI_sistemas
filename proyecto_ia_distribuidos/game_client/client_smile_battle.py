import os
import sys
import time
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

# ✅ Tus imágenes (según tu carpeta REAL)
IMG_MENU     = "SMILE_INICIO.png"
IMG_RULES    = "SMILE_REGLAS.png"
IMG_READY    = "SMILE_LISTO.png"
IMG_GAMEOVER = "SMILE_GAMEOVER.png"
IMG_CAM_BG   = "SMILE_CAMARAS.png"
IMG_NAME_BG  = "SMILE_NOMBRE.png"

FONT_TTF = "font.ttf"

# ✅ Música
MUSIC_FILE   = "sonrisas.mp3"
MUSIC_VOLUME = 0.35

# Gameplay
ROUND_SECONDS = 60

# Calibración (solo avanza si hay 2 caras)
CALIB_SECONDS = 2.0
CALIB_MIN_SAMPLES = 20

# Smile logic
SMILE_LIMIT_PERCENT = 60
SMILE_HOLD_LOSE_SECONDS = 0.70

# Suavizado (para que la barra no brinque feo)
PCT_EMA_ALPHA = 0.25

# Camera
CAM_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# Distributed services
IA_URL = "http://127.0.0.1:8001/infer"
GAME_SERVER_URL = "http://127.0.0.1:8000"
SCORE_ENDPOINT = "/score"
GAME_ID = "smile_battle"


# =========================
# Helpers
# =========================
def clamp(x, a, b):
    return max(a, min(b, x))

def ema(prev, x, alpha):
    return x if prev is None else (alpha * x + (1 - alpha) * prev)

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

def cv_bgr_to_surface_rgb(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    surf = pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], "RGB")
    return surf.convert()

def blit_fit_center(screen, surf, rect):
    """Dibuja surf dentro del rect SIN deformar (mantiene aspecto), centrado."""
    sw, sh = surf.get_size()
    rw, rh = rect.w, rect.h
    if sw <= 0 or sh <= 0 or rw <= 0 or rh <= 0:
        return
    scale = min(rw / sw, rh / sh)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    scaled = pygame.transform.smoothscale(surf, (nw, nh))
    x = rect.x + (rw - nw) // 2
    y = rect.y + (rh - nh) // 2
    screen.blit(scaled, (x, y))

def make_placeholder_surface(size, title="PLAYER", subtitle="Esperando cara..."):
    w, h = size
    surf = pygame.Surface((w, h)).convert()
    surf.fill((30, 30, 30))
    pygame.draw.rect(surf, (245, 245, 245), (0, 0, w, h), 4)

    f1 = pygame.font.SysFont("Arial", 30, bold=True)
    f2 = pygame.font.SysFont("Arial", 18)

    t1 = f1.render(title, True, (255, 255, 255))
    t2 = f2.render(subtitle, True, (220, 220, 220))

    surf.blit(t1, (w//2 - t1.get_width()//2, h//2 - 35))
    surf.blit(t2, (w//2 - t2.get_width()//2, h//2 + 10))
    return surf

def draw_bar(screen, rect, percent, border=3):
    pygame.draw.rect(screen, (255, 255, 255), rect, border)
    fill_w = int((rect.w - border*2) * clamp(percent, 0, 100) / 100)
    pygame.draw.rect(screen, (210, 140, 255),
                     (rect.x + border, rect.y + border, fill_w, rect.h - border*2))

def draw_input(screen, rect, label, value, font_label, font_value, active=False):
    bg = (230, 230, 230)
    border = (0, 0, 0)
    hi = (255, 250, 180) if active else (245, 245, 245)

    pygame.draw.rect(screen, bg, rect)
    pygame.draw.rect(screen, border, rect, 4)
    pygame.draw.rect(screen, hi, (rect.x+4, rect.y+4, rect.w-8, rect.h//2), 0)

    lab = font_label.render(label, True, (0, 0, 0))
    screen.blit(lab, (rect.x, rect.y - lab.get_height() - 8))

    show = value if value.strip() else "Escribe aquí..."
    col = (20, 20, 20) if value.strip() else (110, 110, 110)
    txt = font_value.render(show[:18], True, col)
    screen.blit(txt, (rect.x + 14, rect.centery - txt.get_height()//2))

    if active:
        cx = rect.x + 14 + txt.get_width() + 4
        cy = rect.centery + txt.get_height()//2
        pygame.draw.line(screen, (0, 0, 0), (cx, cy - txt.get_height()), (cx, cy), 3)


# =========================
# IA parsing
# =========================
@dataclass
class FaceInfo:
    bbox_px: tuple
    smile_score: float
    center_x: float

def parse_faces_from_ia(resp, frame_w, frame_h):
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
            x2 = max(x2, x1 + 2)
            y2 = max(y2, y1 + 2)

            smile_score = f.get("smile_ema", f.get("smile_score", None))
            if smile_score is None:
                continue
            smile_score = float(smile_score)

            cx = float(f.get("cx", (x1n + x2n) * 0.5))
            faces_out.append(FaceInfo(bbox_px=(x1, y1, x2, y2), smile_score=smile_score, center_x=cx))
        except Exception:
            continue

    faces_out.sort(key=lambda ff: ff.center_x)
    return faces_out


# =========================
# Names
# =========================
def names_file_path():
    root = os.path.dirname(__file__)
    return os.path.join(root, "players_smile_2p.txt")

def save_player_names(name_left, name_right):
    try:
        with open(names_file_path(), "w", encoding="utf-8") as f:
            f.write(name_left.strip() + "\n")
            f.write(name_right.strip() + "\n")
    except Exception:
        pass

def load_player_names():
    p = names_file_path()
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
        self.sensitivity = 0.22

    def set_baselines(self, base_left, base_right):
        self.base_left = base_left
        self.base_right = base_right

    def percent(self, score, side):
        base = self.base_left if side == "left" else self.base_right
        if base is None or base <= 1e-6:
            return 0.0
        pct = ((score - base) / (base * self.sensitivity)) * 100.0
        return clamp(pct, 0.0, 150.0)


# =========================
# Dashboard send
# =========================
def send_record_to_server(winner, loser, reason, avg_left, avg_right, duration_sec, name_left, name_right):
    if requests is None:
        return

    url = GAME_SERVER_URL.rstrip("/") + SCORE_ENDPOINT

    if winner == "tie":
        return

    if winner == name_left:
        winner_avg = avg_left
    elif winner == name_right:
        winner_avg = avg_right
    else:
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
# States (SIN LETSGO)
# =========================
STATE_MENU = "menu"
STATE_NAME = "name"
STATE_RULES = "rules"
STATE_READY = "ready"
STATE_CALIB = "calib"
STATE_PLAY = "play"
STATE_GAMEOVER = "gameover"


def main():
    global WIN_W, WIN_H

    pygame.init()
    try:
        pygame.mixer.init()
    except Exception:
        pass

    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
    pygame.display.set_caption("Smile Battle")
    clock = pygame.time.Clock()

    # Fuentes
    font_big = load_font(64)
    font_mid = load_font(36)
    font_small = load_font(26)
    font_tiny = load_font(20)

    # Música (no revienta si no existe)
    music_on = True
    music_path = os.path.join(ASSET_DIR, MUSIC_FILE)
    if os.path.exists(music_path):
        try:
            pygame.mixer.music.load(music_path)
            pygame.mixer.music.set_volume(MUSIC_VOLUME)
            pygame.mixer.music.play(-1)
        except Exception:
            music_on = False

    # Cargar imágenes base
    name_img_src = load_image(IMG_NAME_BG)
    menu_img_src = load_image(IMG_MENU)
    rules_src = load_image(IMG_RULES)
    ready_src = load_image(IMG_READY)
    gameover_src = load_image(IMG_GAMEOVER)
    cam_bg_src = load_image(IMG_CAM_BG)

    # ✅ Coordenadas de los cuadros blancos en tu diseño (1920x1080)
    CAM_BG_NATIVE_W, CAM_BG_NATIVE_H = 1920, 1080
    SLOT_L = (139 / CAM_BG_NATIVE_W, 231 / CAM_BG_NATIVE_H, 911 / CAM_BG_NATIVE_W, 849 / CAM_BG_NATIVE_H)
    SLOT_R = (1008 / CAM_BG_NATIVE_W, 241 / CAM_BG_NATIVE_H, 1780 / CAM_BG_NATIVE_W, 849 / CAM_BG_NATIVE_H)

    def rebuild_layout():
        nonlocal screen
        screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)

        name_img = pygame.transform.smoothscale(name_img_src, (WIN_W, WIN_H))
        menu_img = pygame.transform.smoothscale(menu_img_src, (WIN_W, WIN_H))
        rules_img = pygame.transform.smoothscale(rules_src, (WIN_W, WIN_H))
        ready_img = pygame.transform.smoothscale(ready_src, (WIN_W, WIN_H))
        gameover_img = pygame.transform.smoothscale(gameover_src, (WIN_W, WIN_H))
        cam_bg = pygame.transform.smoothscale(cam_bg_src, (WIN_W, WIN_H))

        def rect_from_norm(nr):
            x1, y1, x2, y2 = nr
            rx1 = int(x1 * WIN_W)
            ry1 = int(y1 * WIN_H)
            rx2 = int(x2 * WIN_W)
            ry2 = int(y2 * WIN_H)
            return pygame.Rect(rx1, ry1, max(1, rx2 - rx1), max(1, ry2 - ry1))

        bubble_left = rect_from_norm(SLOT_L)
        bubble_right = rect_from_norm(SLOT_R)

        # barras debajo
        bar_h = int(WIN_H * 0.05)
        bar_y = int(WIN_H * 0.84)
        bar_w = bubble_left.w
        bar_left = pygame.Rect(bubble_left.x, bar_y, bar_w, bar_h)
        bar_right = pygame.Rect(bubble_right.x, bar_y, bar_w, bar_h)

        timer_pos = (WIN_W // 2, int(WIN_H * 0.10))

        ph_left = make_placeholder_surface((bubble_left.w, bubble_left.h), "PLAYER 1", "Acércate a la cámara")
        ph_right = make_placeholder_surface((bubble_right.w, bubble_right.h), "PLAYER 2", "Acércate a la cámara")

        # inputs nombres
        input_w = int(WIN_W * 0.34)
        input_h = int(WIN_H * 0.10)
        gap = int(WIN_W * 0.03)
        y_inputs = int(WIN_H * 0.70)

        in_left = pygame.Rect(WIN_W//2 - input_w - gap//2, y_inputs, input_w, input_h)
        in_right = pygame.Rect(WIN_W//2 + gap//2, y_inputs, input_w, input_h)

        return {
            "name_img": name_img,
            "menu_img": menu_img,
            "rules_img": rules_img,
            "ready_img": ready_img,
            "gameover_img": gameover_img,
            "cam_bg": cam_bg,
            "BUBBLE_LEFT": bubble_left,
            "BUBBLE_RIGHT": bubble_right,
            "BAR_LEFT": bar_left,
            "BAR_RIGHT": bar_right,
            "timer_pos": timer_pos,
            "ph_left": ph_left,
            "ph_right": ph_right,
            "IN_LEFT": in_left,
            "IN_RIGHT": in_right,
        }

    ui = rebuild_layout()

    # Camera
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("No pude abrir la cámara. Revisa permisos o CAM_INDEX.")

    # Names
    name_left, name_right = load_player_names()
    name_left = name_left if name_left else ""
    name_right = name_right if name_right else ""

    active_input = 0
    max_name_len = 18

    normalizer = SmileNormalizer()

    # ✅ MENU primero
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
    calib_t0 = None
    base_left_samples = []
    base_right_samples = []

    # percent smooth
    left_pct_ema = None
    right_pct_ema = None

    # result
    winner = None
    loser = None
    reason = None
    sent_record = False

    def reset_round():
        nonlocal round_left_hold, round_right_hold, left_sum, right_sum, sample_count
        nonlocal start_time, time_left, paused, winner, loser, reason, sent_record
        nonlocal left_pct_ema, right_pct_ema

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
        left_pct_ema = None
        right_pct_ema = None

    def start_calibration():
        nonlocal calib_t0, base_left_samples, base_right_samples, paused
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
        surf = cv_bgr_to_surface_rgb(crop)
        pygame.draw.rect(screen, (20, 20, 20), bubble_rect)
        blit_fit_center(screen, surf, bubble_rect)

    def total_score_from_avg(avg):
        return int(max(0, min(100, round(100 - float(avg)))))

    # ------------- LOOP -------------
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.VIDEORESIZE:
                WIN_W, WIN_H = event.w, event.h
                ui = rebuild_layout()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # -------- MENU --------
                if state == STATE_MENU:
                    if event.key == pygame.K_RETURN:
                        state = STATE_NAME
                    if event.key == pygame.K_q:
                        running = False

                # -------- NAME --------
                elif state == STATE_NAME:
                    if event.key == pygame.K_TAB:
                        active_input = 1 - active_input

                    elif event.key == pygame.K_BACKSPACE:
                        if active_input == 0 and len(name_left) > 0:
                            name_left = name_left[:-1]
                        elif active_input == 1 and len(name_right) > 0:
                            name_right = name_right[:-1]

                    elif event.key == pygame.K_RETURN:
                        if name_left.strip() and name_right.strip():
                            save_player_names(name_left.strip(), name_right.strip())
                            state = STATE_RULES

                    elif event.key == pygame.K_q:
                        state = STATE_MENU

                    else:
                        ch = event.unicode
                        if ch and ch.isprintable():
                            if active_input == 0 and len(name_left) < max_name_len:
                                name_left += ch
                            elif active_input == 1 and len(name_right) < max_name_len:
                                name_right += ch

                # -------- RULES --------
                elif state == STATE_RULES:
                    if event.key == pygame.K_RETURN:
                        state = STATE_READY
                    if event.key == pygame.K_q:
                        state = STATE_MENU

                # -------- READY --------
                elif state == STATE_READY:
                    if event.key == pygame.K_RETURN:
                        state = STATE_CALIB
                        start_calibration()
                    if event.key in (pygame.K_BACKSPACE, pygame.K_q):
                        state = STATE_MENU

                # -------- CALIB/PLAY --------
                elif state in (STATE_CALIB, STATE_PLAY):
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    if event.key == pygame.K_r:
                        state = STATE_CALIB
                        start_calibration()
                    if event.key == pygame.K_q:
                        state = STATE_MENU
                    if event.key == pygame.K_MINUS:
                        normalizer.sensitivity = clamp(normalizer.sensitivity + 0.01, 0.12, 0.40)
                    if event.key == pygame.K_EQUALS:
                        normalizer.sensitivity = clamp(normalizer.sensitivity - 0.01, 0.12, 0.40)
                    if event.key == pygame.K_m:
                        if os.path.exists(music_path):
                            try:
                                if music_on:
                                    pygame.mixer.music.pause()
                                else:
                                    pygame.mixer.music.unpause()
                                music_on = not music_on
                            except Exception:
                                pass

                # -------- GAMEOVER --------
                elif state == STATE_GAMEOVER:
                    if event.key == pygame.K_r:
                        state = STATE_CALIB
                        start_calibration()
                    if event.key == pygame.K_q:
                        state = STATE_MENU
                    if event.key == pygame.K_RETURN:
                        state = STATE_MENU

        # =========================
        # Render por estado
        # =========================
        if state == STATE_MENU:
            screen.blit(ui["menu_img"], (0, 0))
            hint = font_small.render("ENTER = continuar  |  Q = salir", True, (255, 255, 255))
            screen.blit(hint, (20, WIN_H - 40))
            pygame.display.flip()
            continue

        if state == STATE_NAME:
            screen.blit(ui["name_img"], (0, 0))

            title = font_mid.render("NOMBRES", True, (255, 255, 255))
            screen.blit(title, (WIN_W//2 - title.get_width()//2, int(WIN_H*0.08)))

            draw_input(screen, ui["IN_LEFT"], "Jugador 1", name_left, font_tiny, font_mid, active=(active_input == 0))
            draw_input(screen, ui["IN_RIGHT"], "Jugador 2", name_right, font_tiny, font_mid, active=(active_input == 1))

            ok = bool(name_left.strip() and name_right.strip())
            hint1 = font_small.render("TAB = cambiar campo", True, (255, 255, 255))
            hint2 = font_small.render("ENTER = continuar" if ok else "ENTER = pon ambos nombres",
                                      True, (210, 255, 160) if ok else (255, 240, 120))
            hint3 = font_tiny.render("Q = menú  |  ESC = salir", True, (255, 255, 255))

            screen.blit(hint1, (int(WIN_W*0.08), int(WIN_H*0.90)))
            screen.blit(hint2, (int(WIN_W*0.08), int(WIN_H*0.94)))
            screen.blit(hint3, (int(WIN_W*0.72), int(WIN_H*0.94)))

            pygame.display.flip()
            continue

        if state == STATE_RULES:
            screen.blit(ui["rules_img"], (0, 0))
            hint = font_small.render("ENTER = continuar  |  Q = menú", True, (255, 255, 255))
            screen.blit(hint, (20, WIN_H - 40))
            pygame.display.flip()
            continue

        if state == STATE_READY:
            screen.blit(ui["ready_img"], (0, 0))
            hint = font_small.render("ENTER = iniciar  |  BACKSPACE / Q = menú", True, (255, 255, 255))
            screen.blit(hint, (20, WIN_H - 40))
            pygame.display.flip()
            continue

        # =========================
        # Estados que usan cámara
        # =========================
        ret, frame = cap.read()
        if not ret or frame is None:
            screen.fill((10, 0, 10))
            err = font_mid.render("No se pudo leer la cámara", True, (255, 80, 80))
            screen.blit(err, (40, 40))
            pygame.display.flip()
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

        faces = []
        b64 = bgr_to_jpg_b64(frame, quality=70)
        if b64:
            resp = post_json(IA_URL, {"image_b64": b64, "session_id": "smile_battle"}, timeout=1.2)
            faces = parse_faces_from_ia(resp, CAM_WIDTH, CAM_HEIGHT)

        face_left = faces[0] if len(faces) >= 1 else None
        face_right = faces[1] if len(faces) >= 2 else None
        have_two = (face_left is not None and face_right is not None)

        # Fondo cámaras
        screen.blit(ui["cam_bg"], (0, 0))
        draw_face_or_placeholder(face_left, ui["BUBBLE_LEFT"], ui["ph_left"], frame)
        draw_face_or_placeholder(face_right, ui["BUBBLE_RIGHT"], ui["ph_right"], frame)

        # -------- CALIB --------
        if state == STATE_CALIB:
            if not have_two:
                calib_t0 = None
                base_left_samples.clear()
                base_right_samples.clear()

                txt = font_mid.render("CALIBRANDO... (cara neutra)", True, (255, 255, 255))
                screen.blit(txt, (int(WIN_W*0.08), int(WIN_H*0.07)))

                warn = font_small.render("Se necesitan 2 caras detectadas para calibrar", True, (255, 240, 120))
                screen.blit(warn, (int(WIN_W*0.08), int(WIN_H*0.12)))

                hint = font_tiny.render("R = reiniciar  |  Q = menú  |  ESC = salir", True, (255, 255, 255))
                screen.blit(hint, (int(WIN_W*0.08), int(WIN_H*0.94)))

                pygame.display.flip()
                continue

            if calib_t0 is None:
                calib_t0 = time.time()

            base_left_samples.append(face_left.smile_score)
            base_right_samples.append(face_right.smile_score)

            elapsed = time.time() - calib_t0
            prog = int(clamp((elapsed / CALIB_SECONDS) * 100, 0, 100))

            txt = font_mid.render("CALIBRANDO... (cara neutra)", True, (255, 255, 255))
            screen.blit(txt, (int(WIN_W*0.08), int(WIN_H*0.07)))

            pr = pygame.Rect(int(WIN_W*0.08), int(WIN_H*0.13), int(WIN_W*0.30), int(WIN_H*0.03))
            draw_bar(screen, pr, prog)

            if elapsed >= CALIB_SECONDS and len(base_left_samples) >= CALIB_MIN_SAMPLES and len(base_right_samples) >= CALIB_MIN_SAMPLES:
                base_l = float(np.median(np.array(base_left_samples)))
                base_r = float(np.median(np.array(base_right_samples)))
                normalizer.set_baselines(base_l, base_r)

                reset_round()
                state = STATE_PLAY
                paused = False

            hint = font_tiny.render("R = reiniciar  |  Q = menú  |  (-/+) Sensibilidad", True, (255, 255, 255))
            screen.blit(hint, (int(WIN_W*0.08), int(WIN_H*0.94)))

            pygame.display.flip()
            continue

        # -------- PLAY --------
        if state == STATE_PLAY:
            if not have_two:
                paused = True

            if paused:
                msg = font_mid.render("PAUSA: se necesitan 2 caras en cámara", True, (255, 240, 120))
                screen.blit(msg, (int(WIN_W*0.08), int(WIN_H*0.07)))

                #hint = font_small.render("SPACE = continuar  |  R = reiniciar  |  Q = menú", True, (255, 255, 255))
                #screen.blit(hint, (int(WIN_W*0.08), int(WIN_H*0.12)))

                pygame.display.flip()
                continue

            left_pct_raw = normalizer.percent(face_left.smile_score, "left")
            right_pct_raw = normalizer.percent(face_right.smile_score, "right")

            left_pct_ema = ema(left_pct_ema, left_pct_raw, PCT_EMA_ALPHA)
            right_pct_ema = ema(right_pct_ema, right_pct_raw, PCT_EMA_ALPHA)

            left_pct = left_pct_ema
            right_pct = right_pct_ema

            elapsed_round = time.time() - start_time
            time_left = int(max(0, ROUND_SECONDS - elapsed_round))

            left_sum += left_pct
            right_sum += right_pct
            sample_count += 1

            if left_pct >= SMILE_LIMIT_PERCENT:
                round_left_hold += dt
            else:
                round_left_hold = max(0.0, round_left_hold - dt * 1.5)

            if right_pct >= SMILE_LIMIT_PERCENT:
                round_right_hold += dt
            else:
                round_right_hold = max(0.0, round_right_hold - dt * 1.5)

            lbl1 = font_small.render(f"{name_left}  Smile: {int(left_pct)}%", True, (255, 255, 255))
            lbl2 = font_small.render(f"{name_right}  Smile: {int(right_pct)}%", True, (255, 255, 255))
            screen.blit(lbl1, (ui["BAR_LEFT"].x, ui["BAR_LEFT"].y - 32))
            screen.blit(lbl2, (ui["BAR_RIGHT"].x, ui["BAR_RIGHT"].y - 32))

            draw_bar(screen, ui["BAR_LEFT"], clamp(left_pct, 0, 100))
            draw_bar(screen, ui["BAR_RIGHT"], clamp(right_pct, 0, 100))

            timer_text = font_big.render(str(time_left), True, (255, 255, 255))
            timer_rect = timer_text.get_rect(center=ui["timer_pos"])
            screen.blit(timer_text, timer_rect)

            sens = font_tiny.render(f"Sensibilidad: {normalizer.sensitivity:.2f}  (- / +)", True, (255, 255, 255))
            screen.blit(sens, (int(WIN_W*0.08), int(WIN_H*0.90)))

            #hint = font_tiny.render("R = reiniciar  |  Q = menú  |  SPACE = pausa  |  ESC = salir  |  M = música",
            #                        True, (255, 255, 255))
            screen.blit(hint, (int(WIN_W*0.08), int(WIN_H*0.94)))

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
            screen.blit(ui["gameover_img"], (0, 0))

            avg_l = left_sum / max(1, sample_count)
            avg_r = right_sum / max(1, sample_count)

            score_l = total_score_from_avg(avg_l)
            score_r = total_score_from_avg(avg_r)

            # Estados
            left_status = "WIN" if winner == name_left else "FAILED"
            right_status = "WIN" if winner == name_right else "FAILED"
            if winner == "tie":
                left_status = "TIE"
                right_status = "TIE"

            # ---------- HEADER (centrado y BAJADO) ----------
            if winner == "tie":
                who = font_mid.render("EMPATE 😳", True, (255, 240, 120))
            else:
                who = font_mid.render(f"GANADOR: {winner} 🏆", True, (255, 240, 120))

            rs = "Perdiste por sostener sonrisa" if reason == "smile_hold" else "Tiempo terminado"
            rr = font_small.render(rs, True, (255, 255, 255))

            # 🔽 BAJAMOS TODO EL HEADER
            top = int(WIN_H * 0.24)

            who_rect = who.get_rect(midtop=(WIN_W // 2, top))
            screen.blit(who, who_rect)

            rr_rect = rr.get_rect(midtop=(WIN_W // 2, who_rect.bottom + 16))
            screen.blit(rr, rr_rect)

            # ---------- BLOQUES IZQ / DER ----------
            y0 = rr_rect.bottom + int(WIN_H * 0.14)

            # Centros de columnas
            cx_left = WIN_W * 0.25
            cx_right = WIN_W * 0.75

            # Textos
            nmL = font_mid.render(name_left, True, (210, 255, 160))
            nmR = font_mid.render(name_right, True, (210, 255, 160))

            stL = font_big.render(left_status, True, (255, 255, 255))
            stR = font_big.render(right_status, True, (255, 255, 255))

            a1 = font_small.render(f"Avg Smile: {avg_l:.1f}%", True, (255, 255, 255))
            a2 = font_small.render(f"Avg Smile: {avg_r:.1f}%", True, (255, 255, 255))

            s1 = font_small.render(f"Total Score: {score_l}", True, (255, 255, 255))
            s2 = font_small.render(f"Total Score: {score_r}", True, (255, 255, 255))

            # ---------- DIBUJO CENTRADO ----------
            # Izquierda
            screen.blit(nmL, nmL.get_rect(center=(cx_left, y0)))
            screen.blit(stL, stL.get_rect(center=(cx_left, y0 + 70)))
            screen.blit(a1, a1.get_rect(center=(cx_left, y0 + 130)))
            screen.blit(s1, s1.get_rect(center=(cx_left, y0 + 165)))

            # Derecha
            screen.blit(nmR, nmR.get_rect(center=(cx_right, y0)))
            screen.blit(stR, stR.get_rect(center=(cx_right, y0 + 70)))
            screen.blit(a2, a2.get_rect(center=(cx_right, y0 + 130)))
            screen.blit(s2, s2.get_rect(center=(cx_right, y0 + 165)))

            # ---------- HINT ----------
            txt = "R = reiniciar  |  Q = menú  |  ESC = salir"

            hint_black = font_small.render(txt, True, (0, 0, 0))
            hint_white = font_small.render(txt, True, (255, 255, 255))

            rect = hint_black.get_rect(midbottom=(WIN_W // 2, WIN_H - 20))

            # borde (4 direcciones)
            screen.blit(hint_white, rect.move(-2, 0))
            screen.blit(hint_white, rect.move( 2, 0))
            screen.blit(hint_white, rect.move( 0,-2))
            screen.blit(hint_white, rect.move( 0, 2))

            # texto principal negro
            screen.blit(hint_black, rect)
            # ---------- ENVÍO SCORE ----------
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
