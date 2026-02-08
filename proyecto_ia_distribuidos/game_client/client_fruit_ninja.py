import os
import base64
import time
import cv2
import requests
import math
import random
import threading
import queue
from collections import deque
import pygame
import numpy as np

# ================== URLs ==================
IA_URL = "http://127.0.0.1:8001/infer"
GAME_SERVER_URL = "http://127.0.0.1:8000"

# ================== PATHS ==================
ROOT_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")

ASSETS_FRUIT_DIR = os.path.join(ASSETS_DIR, "fruit")

ASSET_SEARCH_DIRS = [
    ASSETS_FRUIT_DIR,
    ASSETS_DIR,
    os.path.join(ASSETS_DIR, "images"),
    os.path.join(ASSETS_DIR, "sprites"),
]

def find_asset(filename: str) -> str:
    """Busca un archivo en varias carpetas. Devuelve path o ''."""
    for d in ASSET_SEARCH_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    for d in ASSET_SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            if filename in files:
                return os.path.join(root, filename)
    return ""

# Pantallas UI (tus PNG)
P_START   = find_asset("Inicio_fruit.png")
P_NAME    = find_asset("Ingresar_Nom_fruit.png")
P_RULES   = find_asset("reglas_fruit.png")
P_READY   = find_asset("Listo_fruit.png")
P_CAM_UI  = find_asset("CAMARAS.png")
P_GAMEOV  = find_asset("GameOver_fruit.png")

# Música / SFX
MUSIC_PATH = find_asset("bg_music.mp3")
SND_SLICE  = find_asset("slice.mp3")
SND_BOOM   = find_asset("boom.mp3")
SND_COMBO  = find_asset("combo.mp3")

# Sprites
SPRITE_FILES = [
    ("banana",     "banana.png"),
    ("orange",     "orange.png"),
    ("kiwi",       "kiwi.png"),
    ("straw",      "fresa.png"),
    ("watermelon", "watermelon.png"),
]
BOMB_FILE = "bomb.png"

# ================== PLAYER NAME ==================
PLAYER_NAME_FILE = os.path.join(ROOT_DIR, "player_name.txt")

def load_player_name():
    try:
        if os.path.exists(PLAYER_NAME_FILE):
            name = open(PLAYER_NAME_FILE, "r", encoding="utf-8").read().strip()
            return name if name else ""
    except Exception:
        pass
    return ""

def save_player_name(name: str):
    try:
        open(PLAYER_NAME_FILE, "w", encoding="utf-8").write(name.strip())
    except Exception:
        pass

PLAYER_NAME = load_player_name()

# ================== CAMERA ==================
CAM_W, CAM_H = 640, 480

# DSHOW reduce warnings MSMF en Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

WIN = "Fruit Ninja - PUCE M"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

# ================== IA THREAD ==================
# Más fluido (sin trabarse)
IA_FPS = 24
IA_TIMEOUT = 1.2

ai_q = queue.Queue(maxsize=1)
ai_lock = threading.Lock()
ai_last = None

def bgr_to_b64jpg(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return base64.b64encode(buf).decode("utf-8")

def ai_worker():
    global ai_last
    session = requests.Session()
    while True:
        item = ai_q.get()
        if item is None:
            break
        try:
            r = session.post(
                IA_URL,
                json={"image_b64": item, "session_id": "fruit"},
                timeout=IA_TIMEOUT
            )
            data = r.json()
        except Exception:
            data = None
        with ai_lock:
            ai_last = data

threading.Thread(target=ai_worker, daemon=True).start()

def ai_push_frame(frame_bgr):
    # Más pequeño = menos lag
    small = cv2.resize(frame_bgr, (432, 324), interpolation=cv2.INTER_AREA)
    img_b64 = bgr_to_b64jpg(small)
    if img_b64 is None:
        return
    try:
        if ai_q.full():
            try:
                ai_q.get_nowait()
            except Exception:
                pass
        ai_q.put_nowait(img_b64)
    except Exception:
        pass

def ai_get_last():
    with ai_lock:
        return ai_last

# ================== AUDIO (pygame) ==================
AUDIO_ENABLED = True
MUSIC_ENABLED = True

snd_slice = None
snd_boom  = None
snd_combo = None
_last_slice = 0.0
SLICE_MIN_GAP = 0.05

def audio_init():
    global snd_slice, snd_boom, snd_combo
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    pygame.mixer.set_num_channels(16)

    snd_slice = pygame.mixer.Sound(SND_SLICE) if SND_SLICE and os.path.exists(SND_SLICE) else None
    snd_boom  = pygame.mixer.Sound(SND_BOOM)  if SND_BOOM  and os.path.exists(SND_BOOM)  else None
    snd_combo = pygame.mixer.Sound(SND_COMBO) if SND_COMBO and os.path.exists(SND_COMBO) else None

def music_start():
    if not MUSIC_ENABLED:
        return
    if not MUSIC_PATH or (not os.path.exists(MUSIC_PATH)):
        return
    try:
        pygame.mixer.music.load(MUSIC_PATH)
        pygame.mixer.music.set_volume(0.35)
        pygame.mixer.music.play(-1)
    except Exception:
        pass

def music_stop():
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass

def music_toggle():
    global MUSIC_ENABLED
    MUSIC_ENABLED = not MUSIC_ENABLED
    if MUSIC_ENABLED:
        music_start()
    else:
        music_stop()

def sfx(sound, force=False):
    global _last_slice
    if (not AUDIO_ENABLED) or (sound is None):
        return
    if (not force) and (sound == snd_slice):
        if time.time() - _last_slice < SLICE_MIN_GAP:
            return
        _last_slice = time.time()
    try:
        sound.play()
    except Exception:
        pass

audio_init()
music_start()

# ================== SPRITES UTILS ==================
def load_sprite_rgba(path):
    if not path or (not os.path.exists(path)):
        return None
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def normalize_sprite(sprite, max_size=140):
    if sprite is None:
        return None
    h, w = sprite.shape[:2]
    m = max(h, w)
    if m <= max_size:
        return sprite
    scale = max_size / float(m)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return cv2.resize(sprite, (nw, nh), interpolation=cv2.INTER_AREA)

def overlay_rgba(dst_bgr, sprite_rgba, x, y, scale=1.0):
    """Dibuja sprite en dst_bgr. Si sprite se sale, OpenCV lo recorta (perfecto para ROI)."""
    if sprite_rgba is None:
        return 30

    sp = sprite_rgba
    if scale != 1.0:
        h, w = sp.shape[:2]
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        sp = cv2.resize(sp, (nw, nh), interpolation=cv2.INTER_AREA)

    sh, sw = sp.shape[:2]
    x1 = int(x - sw / 2)
    y1 = int(y - sh / 2)
    x2 = x1 + sw
    y2 = y1 + sh

    H, W = dst_bgr.shape[:2]
    if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
        return int(max(sw, sh) / 2)

    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = sw - max(0, x2 - W)
    sy2 = sh - max(0, y2 - H)
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(W, x2)
    y2c = min(H, y2)

    sp_crop = sp[sy1:sy2, sx1:sx2]
    if sp_crop.shape[2] == 4:
        rgb = sp_crop[:, :, :3]
        a = sp_crop[:, :, 3:4] / 255.0
        dst_roi = dst_bgr[y1c:y2c, x1c:x2c]
        dst_bgr[y1c:y2c, x1c:x2c] = (dst_roi * (1 - a) + rgb * a).astype("uint8")
    else:
        dst_bgr[y1c:y2c, x1c:x2c] = sp_crop[:, :, :3]

    return int(max(sw, sh) / 2)

# ================== SCREEN IMAGES ==================
def load_screen(path):
    if not path or (not os.path.exists(path)):
        return None
    return cv2.imread(path, cv2.IMREAD_COLOR)

IMG_START  = load_screen(P_START)
IMG_NAME   = load_screen(P_NAME)
IMG_RULES  = load_screen(P_RULES)
IMG_READY  = load_screen(P_READY)
IMG_CAM_UI = load_screen(P_CAM_UI)
IMG_GAMEOV = load_screen(P_GAMEOV)

# ================== ROI DETECTION (rectángulo blanco en CAMARAS.png) ==================
def detect_white_roi(cam_ui_img):
    """
    Detecta el rectángulo blanco más grande (área de cámara) dentro de CAMARAS.png.
    Devuelve (x, y, w, h) en coords de la imagen original.
    """
    if cam_ui_img is None:
        return None
    gray = cv2.cvtColor(cam_ui_img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    th = cv2.medianBlur(th, 5)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best = None
    best_area = 0
    H, W = gray.shape[:2]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < best_area:
            continue
        if w < W * 0.30 or h < H * 0.20:
            continue
        ar = w / float(h)
        if ar < 1.10:
            continue
        best_area = area
        best = (x, y, w, h)

    return best

CAM_ROI_ORIG = detect_white_roi(IMG_CAM_UI)

# ================== WINDOW RESIZE (sin errores de canales) ==================
def resize_to_window(img, ww, wh):
    if img is None:
        canvas = np.zeros((wh, ww, 3), dtype=np.uint8)
        return canvas, 1.0, (0, 0)

    h, w = img.shape[:2]
    scale = min(ww / w, wh / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    ox = (ww - nw) // 2
    oy = (wh - nh) // 2

    canvas = np.zeros((wh, ww, 3), dtype=np.uint8)
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, scale, (ox, oy)

def scaled_roi(roi_orig, scale, offset):
    if roi_orig is None:
        return None
    x, y, w, h = roi_orig
    ox, oy = offset
    return (int(x * scale + ox), int(y * scale + oy), int(w * scale), int(h * scale))

def get_window_size(win_name):
    try:
        _, _, ww, wh = cv2.getWindowImageRect(win_name)
        if ww <= 0 or wh <= 0:
            return 1280, 720
        return ww, wh
    except Exception:
        return 1280, 720

def ui_scale_from_w(w):
    return max(0.75, min(1.20, w / 720.0))

# ================== TEXT / UI ==================
def draw_text(frame, text, x, y, scale=1.0, color=(255, 255, 255), thick=2):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)

def alpha_rect(frame, x, y, w, h, bgr=(0, 0, 0), alpha=0.55):
    x2, y2 = x + w, y + h
    x = max(0, x); y = max(0, y)
    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
    if x2 <= x or y2 <= y:
        return
    roi = frame[y:y2, x:x2].copy()
    overlay = np.full_like(roi, bgr, dtype=np.uint8)
    blended = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
    frame[y:y2, x:x2] = blended

def draw_badge(frame, label, value, x, y, w, h, ui=1.0, accent=(255, 0, 255)):
    alpha_rect(frame, x, y, w, h, bgr=(0, 0, 0), alpha=0.50)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), max(1, int(2 * ui)))
    alpha_rect(frame, x, y, w, int(24 * ui), bgr=accent, alpha=0.35)
    draw_text(frame, f"{label}", x + int(10 * ui), y + int(18 * ui), scale=0.70 * ui, color=(255, 255, 255), thick=2)
    draw_text(frame, f"{value}", x + int(10 * ui), y + int(56 * ui), scale=0.95 * ui, color=(0, 255, 200), thick=2)

# ================== GAME PHYSICS ==================
# Equilibrio: ni lento ni rápido
TIME_SCALE = 0.90
GRAVITY = 520
AIR_FRICTION = 0.998
MAX_FALL_SPEED = 720

# Más frutas, intercaladas
SPAWN_MIN = 0.28
SPAWN_MAX = 0.52

BOMB_CHANCE = 0.10

# A veces salen 2 o 3
BURST_CHANCE = 0.65
BURST_WEIGHTS = [45, 40, 15]  # 1,2,3

# Corte más fácil (menos “paso encima y no corta”)
TOUCH_CUT = True
CUT_PADDING = 44
CUT_COOLDOWN = 0.06

SLASH_CUT = True
CUT_SPEED = 540
SLASH_EXTRA_PAD = 26

FRUIT_SCALE = 0.90
BOMB_SCALE = 0.85
SPRITE_MAX_SIZE_FRUIT = 140
SPRITE_MAX_SIZE_BOMB = 130

GAME_SECONDS = 60

def clamp(v, a, b):
    return max(a, min(b, v))

def seg_circle_intersect(p1, p2, c, r):
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = c
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return ((cx - x1) ** 2 + (cy - y1) ** 2) <= r * r
    t = ((cx - x1) * dx + (cy - y1) * dy) / float(dx * dx + dy * dy)
    t = clamp(t, 0.0, 1.0)
    px = x1 + t * dx
    py = y1 + t * dy
    dist2 = (cx - px) ** 2 + (cy - py) ** 2
    return dist2 <= r * r

def send_score(game: str, score: int):
    try:
        payload = {"player": PLAYER_NAME, "game": game, "score": int(score), "extra": {}}
        requests.post(f"{GAME_SERVER_URL}/score", json=payload, timeout=1.0)
    except Exception:
        pass

class Fruit:
    def __init__(self, x, y, vx, vy, kind="fruit", sprite=None, scale=1.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.kind = kind
        self.alive = True
        self.sprite = sprite
        self.scale = scale
        self.r = 28
        self.last_cut_ts = 0.0

    def update(self, dt):
        self.vy += GRAVITY * dt
        self.vy = min(self.vy, MAX_FALL_SPEED)
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= AIR_FRICTION
        self.vy *= AIR_FRICTION

def spawn_item(rw, rh, fruit_sprites, bomb_sprite):
    """
    IMPORTANTÍSIMO:
    Spawn en coordenadas LOCALES del ROI (0..rw, 0..rh).
    Así las frutas SOLO existen dentro de la cámara.
    """
    x = random.randint(60, max(61, rw - 60))
    y = random.randint(-260, -120)

    vx = random.uniform(-170, 170)
    vy = random.uniform(-980, -760)
    s = random.uniform(0.90, 1.18)

    if random.random() < BOMB_CHANCE:
        return Fruit(x, y, vx * 0.70, vy, kind="bomb", sprite=bomb_sprite, scale=s * BOMB_SCALE)
    else:
        _, sp = random.choice(fruit_sprites)
        return Fruit(x, y, vx, vy, kind="fruit", sprite=sp, scale=s * FRUIT_SCALE)

# ================== SMOOTH TIP ==================
class EMA2D:
    def __init__(self, alpha=0.62):
        self.alpha = alpha
        self.prev = None

    def reset(self):
        self.prev = None

    def update(self, x, y):
        if self.prev is None:
            self.prev = (float(x), float(y))
            return self.prev
        px, py = self.prev
        nx = (1 - self.alpha) * px + self.alpha * float(x)
        ny = (1 - self.alpha) * py + self.alpha * float(y)
        self.prev = (nx, ny)
        return self.prev

def clamp_int(v, a, b):
    return int(max(a, min(b, v)))

# Cuando NO hay mano: NO queremos que se quede el punto.
# Hold corto para evitar parpadeo, pero si se pierde la mano, desaparece rápido.
TIP_HOLD_MS = 90
tip_ema = EMA2D(alpha=0.62)
last_tip_local = None
last_tip_ts = 0.0

# ================== GAME STATE MACHINE ==================
STATE_START = "START"
STATE_NAME  = "NAME"
STATE_RULES = "RULES"
STATE_READY = "READY"
STATE_PLAY  = "PLAY"

state = STATE_START
name_buffer = PLAYER_NAME if PLAYER_NAME else ""
must_enter_name = True

# Carga sprites
fruit_sprites = []
for nm, file in SPRITE_FILES:
    p = find_asset(file)
    sp = normalize_sprite(load_sprite_rgba(p), SPRITE_MAX_SIZE_FRUIT)
    fruit_sprites.append((nm, sp))

bomb_sprite = normalize_sprite(load_sprite_rgba(find_asset(BOMB_FILE)), SPRITE_MAX_SIZE_BOMB)

# Vars del juego
items = []
last_spawn = 0.0
spawn_every = random.uniform(SPAWN_MIN, SPAWN_MAX)

score = 0
combo = 0
last_cut_time = 0.0
game_over = False
score_sent = False
start_time = time.time()

# Trail LOCAL al ROI
trail = deque(maxlen=18)
prev_tip = None
prev_tip_t = None
tip_speed = 0.0

last_ai_send = 0.0
ai_every = 1.0 / IA_FPS
last_latency_ms = None

is_fullscreen = False
prev_frame_t = time.time()

def reset_game():
    global items, last_spawn, spawn_every, score, combo, last_cut_time, game_over, score_sent
    global start_time, trail, prev_tip, prev_tip_t, tip_speed
    global last_tip_local, last_tip_ts

    items = []
    last_spawn = time.time()
    spawn_every = random.uniform(SPAWN_MIN, SPAWN_MAX)
    score = 0
    combo = 0
    last_cut_time = 0.0
    game_over = False
    score_sent = False
    start_time = time.time()

    trail.clear()
    prev_tip = None
    prev_tip_t = None
    tip_speed = 0.0

    tip_ema.reset()
    last_tip_local = None
    last_tip_ts = 0.0

def do_cut(it, now, points):
    global score, combo, last_cut_time
    it.alive = False
    score += points
    if now - last_cut_time <= 0.85:
        combo += 1
    else:
        combo = 1
    last_cut_time = now
    sfx(snd_slice)
    if combo >= 3:
        sfx(snd_combo, force=True)

def do_bomb():
    global game_over
    game_over = True
    sfx(snd_boom, force=True)

# ================== MAIN LOOP ==================
while True:
    ret, cam = cap.read()
    if not ret:
        break

    cam = cv2.flip(cam, 1)

    now = time.time()
    dt = max(now - prev_frame_t, 1e-6)
    dt = min(dt, 1 / 18.0)
    dt *= TIME_SCALE
    prev_frame_t = now

    ww, wh = get_window_size(WIN)
    ui = ui_scale_from_w(ww)

    # ================== START ==================
    if state == STATE_START:
        canvas, _, _ = resize_to_window(IMG_START, ww, wh)

        draw_text(canvas, "ENTER: continuar", int(28 * ui), wh - int(42 * ui), scale=0.95 * ui)
        draw_text(canvas, "ESC: salir", int(28 * ui), wh - int(14 * ui), scale=0.85 * ui)

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        if key in (13, 10):
            state = STATE_NAME if must_enter_name else STATE_RULES
        continue

    # ================== NAME ==================
    if state == STATE_NAME:
        canvas, _, _ = resize_to_window(IMG_NAME, ww, wh)

        draw_text(canvas, "ESCRIBE TU NOMBRE", ww // 2 - int(260 * ui), int(160 * ui), scale=1.25 * ui)
        draw_text(canvas, (name_buffer if name_buffer else "") + "_", ww // 2 - int(260 * ui), int(270 * ui),
                  scale=1.35 * ui, color=(0, 255, 200), thick=3)
        draw_text(canvas, "ENTER: guardar y continuar", ww // 2 - int(300 * ui), int(340 * ui), scale=0.9 * ui)
        draw_text(canvas, "Backspace: borrar", ww // 2 - int(220 * ui), int(380 * ui), scale=0.85 * ui)
        draw_text(canvas, "ESC: salir", int(28 * ui), wh - int(14 * ui), scale=0.85 * ui)

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        if key in (8, 127):
            name_buffer = name_buffer[:-1]
        elif key in (13, 10):
            name_buffer = name_buffer.strip()
            if len(name_buffer) >= 1:
                PLAYER_NAME = name_buffer
                save_player_name(PLAYER_NAME)
                state = STATE_RULES
        elif 32 <= key <= 126:
            if len(name_buffer) < 16:
                name_buffer += chr(key)
        continue

    # ================== RULES ==================
    if state == STATE_RULES:
        canvas, _, _ = resize_to_window(IMG_RULES, ww, wh)
        draw_text(canvas, "ENTER: continuar", int(28 * ui), wh - int(42 * ui), scale=0.95 * ui)
        draw_text(canvas, "Q: inicio", int(28 * ui), wh - int(14 * ui), scale=0.85 * ui)

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        if key in (ord("q"), ord("Q")):
            state = STATE_START
        if key in (13, 10):
            state = STATE_READY
        continue

    # ================== READY ==================
    if state == STATE_READY:
        canvas, _, _ = resize_to_window(IMG_READY, ww, wh)
        draw_text(canvas, f"PLAYER: {PLAYER_NAME}", int(28 * ui), int(60 * ui), scale=0.95 * ui, color=(0, 255, 200))
        draw_text(canvas, "ENTER: empezar", int(28 * ui), wh - int(42 * ui), scale=0.95 * ui)
        draw_text(canvas, "Q: inicio", int(28 * ui), wh - int(14 * ui), scale=0.85 * ui)

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        if key in (ord("q"), ord("Q")):
            state = STATE_START
        if key in (13, 10):
            reset_game()
            state = STATE_PLAY
        continue

    # ================== PLAY ==================
    cam_ui_canvas, scale_ui, offset_ui = resize_to_window(IMG_CAM_UI, ww, wh)

    roi = scaled_roi(CAM_ROI_ORIG, scale_ui, offset_ui)
    if roi is None:
        roi = (int(ww * 0.14), int(wh * 0.18), int(ww * 0.72), int(wh * 0.60))
    rx, ry, rw, rh = roi

    # Colocar cámara en ROI
    cam_fit = cv2.resize(cam, (rw, rh), interpolation=cv2.INTER_AREA)
    cam_ui_canvas[ry:ry + rh, rx:rx + rw] = cam_fit

    # ¡CLAVE! Todo lo del juego se dibuja SOLO aquí
    roi_view = cam_ui_canvas[ry:ry + rh, rx:rx + rw]

    # IA push
    if (not game_over) and (now - last_ai_send >= ai_every):
        last_ai_send = now
        ai_push_frame(cam)

    data = ai_get_last()

    # timer
    elapsed = int(now - start_time)
    remaining = max(0, GAME_SECONDS - elapsed)
    if remaining == 0 and (not game_over):
        game_over = True

    # enviar score una sola vez
    if game_over and (not score_sent):
        send_score("fruit_ninja", score)
        score_sent = True

    # Spawn (LOCAL al ROI)
    if (not game_over) and (now - last_spawn >= spawn_every):
        last_spawn = now
        spawn_every = random.uniform(SPAWN_MIN, SPAWN_MAX)

        burst = 1
        if random.random() < BURST_CHANCE:
            burst = random.choices([1, 2, 3], weights=BURST_WEIGHTS, k=1)[0]

        for _ in range(burst):
            items.append(spawn_item(rw, rh, fruit_sprites, bomb_sprite))

    # Update items (LOCAL)
    for it in items:
        if it.alive:
            it.update(dt)
            if it.y - it.r > (rh + 90):
                it.alive = False
                combo = 0

    # ------------------ DEDO IA (LOCAL al ROI) ------------------
    tip_local = None
    hand_ok = False

    if (not game_over) and data:
        hp = data.get("hand_point")

        # 1) x_ema/y_ema (preferido)
        if hp and (hp.get("x_ema") is not None) and (hp.get("y_ema") is not None):
            hand_ok = True
            tx = int(float(hp["x_ema"]) * rw)
            ty = int(float(hp["y_ema"]) * rh)
            tx = clamp_int(tx, 0, rw - 1)
            ty = clamp_int(ty, 0, rh - 1)

            sx, sy = tip_ema.update(tx, ty)
            tip_local = (int(sx), int(sy))

            last_tip_local = tip_local
            last_tip_ts = now

        else:
            # 2) fallback landmarks[8]
            lm = data.get("hand_landmarks")
            if lm and len(lm) > 8 and ("x" in lm[8]) and ("y" in lm[8]):
                hand_ok = True
                rawx = int(float(lm[8]["x"]) * rw)
                rawy = int(float(lm[8]["y"]) * rh)
                rawx = clamp_int(rawx, 0, rw - 1)
                rawy = clamp_int(rawy, 0, rh - 1)

                sx, sy = tip_ema.update(rawx, rawy)
                tip_local = (int(sx), int(sy))

                last_tip_local = tip_local
                last_tip_ts = now

    # HOLD corto anti-parpadeo (pero luego desaparece)
    if (not hand_ok) and (last_tip_local is not None):
        if ((now - last_tip_ts) * 1000.0) < TIP_HOLD_MS:
            hand_ok = True
            tip_local = last_tip_local
        else:
            # Ya no hay mano: se borra todo (tu pedido)
            last_tip_local = None
            trail.clear()
            prev_tip = None
            prev_tip_t = None
            tip_speed = 0.0
            tip_ema.reset()

    # trail + speed (LOCAL)
    if hand_ok and tip_local is not None:
        trail.append(tip_local)
        if prev_tip is not None and prev_tip_t is not None:
            dtt = max(now - prev_tip_t, 1e-6)
            dx = tip_local[0] - prev_tip[0]
            dy = tip_local[1] - prev_tip[1]
            tip_speed = math.sqrt(dx * dx + dy * dy) / dtt
        prev_tip = tip_local
        prev_tip_t = now
    else:
        # si no hay mano: nada dibujado
        tip_local = None

    # Trail SOLO en ROI (color magenta/cyan)
    if tip_local is not None:
        for i in range(1, len(trail)):
            cv2.line(roi_view, trail[i - 1], trail[i], (255, 0, 255), 5, cv2.LINE_AA)
        cv2.circle(roi_view, tip_local, 10, (0, 255, 200), -1, cv2.LINE_AA)

    # Draw items SOLO en ROI (nunca se salen a la interfaz)
    for it in items:
        if not it.alive:
            continue
        cx, cy = int(it.x), int(it.y)
        it.r = overlay_rgba(roi_view, it.sprite, cx, cy, scale=it.scale)

    # Colisiones touch (LOCAL)
    if (not game_over) and tip_local is not None and TOUCH_CUT:
        tx, ty = tip_local

        speed_boost = clamp(tip_speed / 900.0, 0.0, 1.0)
        extra = int(18 * speed_boost)

        for it in items:
            if not it.alive:
                continue
            if now - it.last_cut_ts < CUT_COOLDOWN:
                continue

            rr = (it.r + CUT_PADDING + extra)
            dist2 = (it.x - tx) ** 2 + (it.y - ty) ** 2

            if dist2 <= rr * rr:
                it.last_cut_ts = now
                if it.kind == "bomb":
                    do_bomb()
                    break
                else:
                    do_cut(it, now, 10)

    # Slash cut (LOCAL)
    if (not game_over) and tip_local is not None and SLASH_CUT and len(trail) >= 2:
        p1 = trail[-2]
        p2 = trail[-1]

        if len(trail) >= 3:
            jump = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
            if jump > (160**2):
                p1 = trail[-3]

        if tip_speed >= (CUT_SPEED * 0.75):
            pad = SLASH_EXTRA_PAD + int(clamp(tip_speed / 1000.0, 0.0, 1.0) * 16)

            for it in items:
                if not it.alive:
                    continue
                if now - it.last_cut_ts < CUT_COOLDOWN:
                    continue

                if seg_circle_intersect(p1, p2, (int(it.x), int(it.y)), it.r + pad):
                    it.last_cut_ts = now
                    if it.kind == "bomb":
                        do_bomb()
                        break
                    else:
                        do_cut(it, now, 10)

    # Limpia lista
    if len(items) > 80:
        items = [it for it in items if it.alive]

    # HUD (fuera del ROI, pero encima de la interfaz está bien)
    draw_badge(cam_ui_canvas, "SCORE", score, int(20 * ui), int(18 * ui), int(175 * ui), int(78 * ui), ui=ui, accent=(255, 0, 255))
    draw_badge(cam_ui_canvas, "TIME", remaining, int(210 * ui), int(18 * ui), int(175 * ui), int(78 * ui), ui=ui, accent=(0, 255, 200))
    draw_badge(cam_ui_canvas, "COMBO", f"x{combo}", int(400 * ui), int(18 * ui), int(190 * ui), int(78 * ui), ui=ui, accent=(255, 80, 0))

    if combo >= 3 and (not game_over):
        draw_text(cam_ui_canvas, "COMBO!", int(rx + rw - 180 * ui), int(ry + 70 * ui),
                  scale=1.15 * ui, color=(0, 255, 200), thick=3)

    # IA latency
    if data:
        lat = data.get("latency_ms")
        if lat is not None:
            last_latency_ms = lat
    if last_latency_ms is not None:
        draw_text(cam_ui_canvas, f"IA {last_latency_ms} ms", int(20 * ui), wh - int(20 * ui),
                  scale=0.78 * ui, color=(255, 255, 255), thick=2)

    draw_text(cam_ui_canvas, f"PLAYER: {PLAYER_NAME}", int(20 * ui), wh - int(55 * ui),
              scale=0.78 * ui, color=(255, 255, 255), thick=2)

    # Si no hay mano: avisito (y ya NO queda el punto)
    if (tip_local is None) and (not game_over):
        draw_text(cam_ui_canvas, "NO HAND DETECTED", int(rx + 20 * ui), int(ry + rh - 20 * ui),
                  scale=0.95 * ui, color=(0, 0, 255), thick=3)

    # GAME OVER overlay
    if game_over:
        over_canvas, _, _ = resize_to_window(IMG_GAMEOV, ww, wh)
        if over_canvas is None:
            over_canvas = cam_ui_canvas.copy()
        draw_text(over_canvas, f"TOTAL SCORE: {score}", ww // 2 - int(240 * ui), wh // 2 + int(80 * ui),
                  scale=1.15 * ui, color=(255, 255, 255), thick=3)
        draw_text(over_canvas, "R: reiniciar", ww // 2 - int(160 * ui), wh // 2 + int(130 * ui),
                  scale=0.95 * ui, color=(255, 255, 255), thick=2)
        draw_text(over_canvas, "Q: inicio", ww // 2 - int(140 * ui), wh // 2 + int(170 * ui),
                  scale=0.95 * ui, color=(255, 255, 255), thick=2)
        draw_text(over_canvas, "ESC: salir", ww // 2 - int(140 * ui), wh // 2 + int(210 * ui),
                  scale=0.95 * ui, color=(255, 255, 255), thick=2)

        cv2.imshow(WIN, over_canvas)
    else:
        cv2.imshow(WIN, cam_ui_canvas)

    key = cv2.waitKey(1) & 0xFF

    # CONTROLES
    if key == 27:
        break
    elif key in (ord("r"), ord("R")):
        reset_game()
        game_over = False
        score_sent = False
    elif key in (ord("q"), ord("Q")):
        reset_game()
        state = STATE_START
    elif key in (ord("s"), ord("S")):
        AUDIO_ENABLED = not AUDIO_ENABLED
    elif key in (ord("m"), ord("M")):
        music_toggle()
    elif key in (ord("f"), ord("F")):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

# Cleanup
try:
    ai_q.put(None)
except Exception:
    pass

cap.release()
cv2.destroyAllWindows()
