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

# ========= AUDIO (pygame) =========
import pygame

IA_URL = "http://127.0.0.1:8001/infer"
GAME_SERVER_URL = "http://127.0.0.1:8000"

# ========= PLAYER NAME (archivo + input en pantalla) =========
PLAYER_NAME_FILE = os.path.join(os.path.dirname(__file__), "player_name.txt")

def load_player_name():
    try:
        if os.path.exists(PLAYER_NAME_FILE):
            name = open(PLAYER_NAME_FILE, "r", encoding="utf-8").read().strip()
            if name:
                return name
    except Exception:
        pass
    return ""

def save_player_name(name: str):
    try:
        open(PLAYER_NAME_FILE, "w", encoding="utf-8").write(name.strip())
    except Exception:
        pass

PLAYER_NAME = load_player_name()
name_edit_mode = (PLAYER_NAME == "")
name_buffer = PLAYER_NAME if PLAYER_NAME else ""

# ========= PATHS =========
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
BG_PATH = os.path.join(ASSETS_DIR, "bg.jpg")
MUSIC_PATH = os.path.join(ASSETS_DIR, "bg_music.mp3")  # o .ogg o .wav

SND_SLICE = os.path.join(ASSETS_DIR, "slice.wav")
SND_BOOM  = os.path.join(ASSETS_DIR, "boom.wav")
SND_COMBO = os.path.join(ASSETS_DIR, "combo.wav")

SPRITES = [
    ("banana",     os.path.join(ASSETS_DIR, "banana.png")),
    ("orange",     os.path.join(ASSETS_DIR, "orange.png")),
    ("kiwi",       os.path.join(ASSETS_DIR, "kiwi.png")),
    ("straw",      os.path.join(ASSETS_DIR, "fresa.png")),
    ("watermelon", os.path.join(ASSETS_DIR, "watermelon.png")),
]
BOMB_SPRITE = os.path.join(ASSETS_DIR, "bomb.png")

# ========= AJUSTES =========
CAM_W, CAM_H = 640, 480

IA_FPS = 18
IA_TIMEOUT = 1.2

EMA_ALPHA = 0.62  # dedo más responsivo

# ---- FÍSICA ----
TIME_SCALE = 0.82
GRAVITY = 430
AIR_FRICTION = 0.996
MAX_FALL_SPEED = 420

# ---- SPAWN ----
SPAWN_MIN = 0.75
SPAWN_MAX = 1.05
BOMB_CHANCE = 0.12

BURST_CHANCE = 0.45
BURST_WEIGHTS = [60, 30, 10]  # 1,2,3

TOUCH_CUT = True
CUT_PADDING = 22
CUT_COOLDOWN = 0.18

SLASH_CUT = True
CUT_SPEED = 720

FRUIT_SCALE = 1.08
BOMB_SCALE  = 1.02
SPRITE_MAX_SIZE_FRUIT = 165
SPRITE_MAX_SIZE_BOMB  = 155

GAME_SECONDS = 60

# ========= UTIL =========
def clamp(v, a, b):
    return max(a, min(b, v))

def bgr_to_b64jpg(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return base64.b64encode(buf).decode("utf-8")

def seg_circle_intersect(p1, p2, c, r):
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = c
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return ((cx-x1)**2 + (cy-y1)**2) <= r*r

    t = ((cx - x1)*dx + (cy - y1)*dy) / float(dx*dx + dy*dy)
    t = clamp(t, 0.0, 1.0)
    px = x1 + t*dx
    py = y1 + t*dy
    dist2 = (cx - px)**2 + (cy - py)**2
    return dist2 <= r*r

def send_score(game: str, score: int):
    """Envía el score al Game Server (no rompe si falla)."""
    try:
        payload = {"player": PLAYER_NAME, "game": game, "score": int(score), "extra": {}}
        requests.post(f"{GAME_SERVER_URL}/score", json=payload, timeout=1.0)
    except Exception:
        pass

# ========= EMA =========
class EMA2D:
    def __init__(self, alpha=0.62):
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

ema = EMA2D(alpha=EMA_ALPHA)

# ========= SPRITES =========
def load_sprite(path):
    if not os.path.exists(path):
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
    if sprite_rgba is None:
        return 34

    sp = sprite_rgba
    if scale != 1.0:
        h, w = sp.shape[:2]
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        sp = cv2.resize(sp, (nw, nh), interpolation=cv2.INTER_AREA)

    sh, sw = sp.shape[:2]
    x1 = int(x - sw/2)
    y1 = int(y - sh/2)
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

# ========= FULLSCREEN sin distorsión =========
def fit_to_window(frame, win_name):
    try:
        _, _, ww, wh = cv2.getWindowImageRect(win_name)
        if ww <= 0 or wh <= 0:
            return frame
        h, w = frame.shape[:2]
        scale = min(ww / w, wh / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = (0 * resized[:1, :1]).repeat(wh, 0).repeat(ww, 1)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        ox = (ww - nw) // 2
        oy = (wh - nh) // 2
        canvas[oy:oy+nh, ox:ox+nw] = resized
        return canvas
    except Exception:
        return frame

# ========= ENTIDADES =========
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
        self.r = 34
        self.last_cut_ts = 0.0

    def update(self, dt):
        self.vy += GRAVITY * dt
        self.vy = min(self.vy, MAX_FALL_SPEED)
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= AIR_FRICTION
        self.vy *= AIR_FRICTION

def spawn_item(w, h, fruit_sprites, bomb_sprite):
    x = random.randint(80, w - 80)
    y = random.randint(-220, -90)
    vx = random.uniform(-110, 110)
    vy = random.uniform(-760, -560)  # más tiempo en pantalla
    s = random.uniform(0.95, 1.15)

    if random.random() < BOMB_CHANCE:
        return Fruit(x, y, vx*0.75, vy, kind="bomb", sprite=bomb_sprite, scale=s * BOMB_SCALE)
    else:
        _, sp = random.choice(fruit_sprites)
        return Fruit(x, y, vx, vy, kind="fruit", sprite=sp, scale=s * FRUIT_SCALE)

# ========= IA EN HILO =========
ai_q = queue.Queue(maxsize=1)
ai_lock = threading.Lock()
ai_last = None

def ai_worker():
    global ai_last
    session = requests.Session()
    while True:
        item = ai_q.get()
        if item is None:
            break
        try:
            r = session.post(IA_URL, json={"image_b64": item}, timeout=IA_TIMEOUT)
            data = r.json()
        except Exception:
            data = None
        with ai_lock:
            ai_last = data

threading.Thread(target=ai_worker, daemon=True).start()

def ai_push_frame(frame_clean):
    small = cv2.resize(frame_clean, (480, 360), interpolation=cv2.INTER_AREA)
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

# ========= AUDIO pygame =========
AUDIO_ENABLED = True
MUSIC_ENABLED = True

_last_slice = 0.0
SLICE_MIN_GAP = 0.05

snd_slice = None
snd_boom  = None
snd_combo = None

def audio_init():
    global snd_slice, snd_boom, snd_combo
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    pygame.mixer.set_num_channels(16)

    snd_slice = pygame.mixer.Sound(SND_SLICE) if os.path.exists(SND_SLICE) else None
    snd_boom  = pygame.mixer.Sound(SND_BOOM)  if os.path.exists(SND_BOOM)  else None
    snd_combo = pygame.mixer.Sound(SND_COMBO) if os.path.exists(SND_COMBO) else None

def music_start():
    if not MUSIC_ENABLED:
        return
    if not os.path.exists(MUSIC_PATH):
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

# ========= CÁMARA =========
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

WIN = "Fruit Ninja - Hand Slicing"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

# ========= CARGA ASSETS =========
bg_img = cv2.imread(BG_PATH, cv2.IMREAD_COLOR) if os.path.exists(BG_PATH) else None

fruit_sprites = []
for nm, path in SPRITES:
    sp = normalize_sprite(load_sprite(path), SPRITE_MAX_SIZE_FRUIT)
    fruit_sprites.append((nm, sp))

bomb_sprite = normalize_sprite(load_sprite(BOMB_SPRITE), SPRITE_MAX_SIZE_BOMB)

# ========= GAME STATE =========
items = []
last_spawn = 0.0
spawn_every = random.uniform(SPAWN_MIN, SPAWN_MAX)

score = 0
combo = 0
last_cut_time = 0.0
game_over = False
score_sent = False

start_time = time.time()

trail = deque(maxlen=16)
prev_tip = None
prev_tip_t = None
tip_speed = 0.0

last_ai_send = 0.0
ai_every = 1.0 / IA_FPS
last_latency_ms = None

is_fullscreen = False

def restart():
    global items, last_spawn, spawn_every, score, combo, last_cut_time, game_over, score_sent
    global start_time, trail, prev_tip, prev_tip_t, tip_speed
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
    ema.prev = None

def ui_scale_from_w(w):
    return max(0.75, min(1.15, w / 720.0))

def draw_text(frame, text, x, y, scale=1.0, color=(255,255,255), thick=2):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)

def do_cut(it, now, points):
    global score, combo, last_cut_time
    it.alive = False
    score += points
    if now - last_cut_time <= 0.9:
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

prev_frame_t = time.time()

while True:
    ret, cam = cap.read()
    if not ret:
        break

    cam = cv2.flip(cam, 1)

    now = time.time()
    dt = max(now - prev_frame_t, 1e-6)
    dt = min(dt, 1/20)
    dt *= TIME_SCALE
    prev_frame_t = now

    h, w = cam.shape[:2]
    ui = ui_scale_from_w(w)

    # IA
    if (not game_over) and (now - last_ai_send >= ai_every):
        last_ai_send = now
        ai_push_frame(cam)

    data = ai_get_last()

    # frame base (cam + fondo)
    frame = cam.copy()
    if bg_img is not None:
        bg = cv2.resize(bg_img, (w, h), interpolation=cv2.INTER_AREA)
        frame = cv2.addWeighted(cam, 0.35, bg, 0.65, 0)

    # tiempo
    elapsed = int(now - start_time)
    remaining = max(0, GAME_SECONDS - elapsed)
    if remaining == 0 and not game_over:
        game_over = True

    # enviar score 1 sola vez
    if game_over and (not score_sent) and (not name_edit_mode):
        send_score("fruit_ninja", score)
        score_sent = True

    # SPAWN
    if (not game_over) and (now - last_spawn >= spawn_every):
        last_spawn = now
        spawn_every = random.uniform(SPAWN_MIN, SPAWN_MAX)

        burst = 1
        if random.random() < BURST_CHANCE:
            burst = random.choices([1, 2, 3], weights=BURST_WEIGHTS, k=1)[0]

        old_bomb = BOMB_CHANCE
        if burst >= 2:
            globals()["BOMB_CHANCE"] = max(0.06, old_bomb * 0.65)

        for _ in range(burst):
            items.append(spawn_item(w, h, fruit_sprites, bomb_sprite))

        globals()["BOMB_CHANCE"] = old_bomb

    # update items
    for it in items:
        if it.alive:
            it.update(dt)
            if it.y - it.r > h + 80:
                it.alive = False
                combo = 0

    # dedo
    tip = None
    hand_ok = False
    if (not game_over) and (not name_edit_mode) and data and data.get("hand_landmarks"):
        lm = data["hand_landmarks"]
        hand_ok = True
        ix = int(lm[8]["x"] * w)
        iy = int(lm[8]["y"] * h)
        sx, sy = ema.update(ix, iy)
        tip = (int(sx), int(sy))
        trail.append(tip)

        if prev_tip is not None and prev_tip_t is not None:
            dtt = max(now - prev_tip_t, 1e-6)
            dx = tip[0] - prev_tip[0]
            dy = tip[1] - prev_tip[1]
            tip_speed = math.sqrt(dx*dx + dy*dy) / dtt

        prev_tip = tip
        prev_tip_t = now
    else:
        trail.clear()
        prev_tip = None
        prev_tip_t = None
        tip_speed = 0.0

    # trail
    if tip is not None:
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i-1], trail[i], (0, 255, 255), 5, cv2.LINE_AA)
        cv2.circle(frame, tip, 10, (0, 255, 0), -1, cv2.LINE_AA)

    # draw items
    for it in items:
        if not it.alive:
            continue
        cx, cy = int(it.x), int(it.y)
        it.r = overlay_rgba(frame, it.sprite, cx, cy, scale=it.scale)

    # colisiones
    if (not game_over) and (not name_edit_mode) and tip is not None and TOUCH_CUT:
        tx, ty = tip
        for it in items:
            if not it.alive:
                continue
            if now - it.last_cut_ts < CUT_COOLDOWN:
                continue
            rr = (it.r + CUT_PADDING)
            dist2 = (it.x - tx)**2 + (it.y - ty)**2
            if dist2 <= rr*rr:
                it.last_cut_ts = now
                if it.kind == "bomb":
                    do_bomb()
                    break
                else:
                    do_cut(it, now, 10)

    # slash cut
    if (not game_over) and (not name_edit_mode) and tip is not None and SLASH_CUT and len(trail) >= 2 and tip_speed >= CUT_SPEED:
        p1 = trail[-2]
        p2 = trail[-1]
        for it in items:
            if not it.alive:
                continue
            if now - it.last_cut_ts < CUT_COOLDOWN:
                continue
            if seg_circle_intersect(p1, p2, (int(it.x), int(it.y)), it.r + 8):
                it.last_cut_ts = now
                if it.kind == "bomb":
                    do_bomb()
                    break
                else:
                    do_cut(it, now, 10)

    # limpia lista
    if len(items) > 45:
        items = [it for it in items if it.alive]

    # UI normal
    if not name_edit_mode:
        top_h = int(70 * ui)
        cv2.rectangle(frame, (0, 0), (w, top_h), (0, 0, 0), -1)

        draw_text(frame, f"SCORE {score}", 14, int(48 * ui), scale=1.05 * ui, color=(255,255,255), thick=2)
        draw_text(frame, f"COMBO x{combo}", w//2 - int(120 * ui), int(48 * ui),
                  scale=1.05 * ui, color=(0,255,255), thick=2)
        draw_text(frame, f"TIME {remaining}", w - int(190 * ui), int(48 * ui),
                  scale=1.05 * ui, color=(255,255,255), thick=2)

        sound_label = "SOUND: ON (S)" if AUDIO_ENABLED else "SOUND: MUTED (S)"
        sound_color = (0,255,0) if AUDIO_ENABLED else (0,0,255)
        draw_text(frame, sound_label, 14, h - int(14 * ui), scale=0.70 * ui, color=sound_color, thick=2)

        music_label = "MUSIC: ON (M)" if MUSIC_ENABLED else "MUSIC: OFF (M)"
        music_color = (0,255,0) if MUSIC_ENABLED else (0,0,255)
        draw_text(frame, music_label, 14, h - int(36 * ui), scale=0.70 * ui, color=music_color, thick=2)

        if not hand_ok and not game_over:
            draw_text(frame, "NO HAND DETECTED", w//2 - int(150 * ui), h - int(14 * ui),
                      scale=0.75 * ui, color=(0,0,255), thick=2)
            
         # Nombre del jugador (debajo de la barra)
        draw_text(frame, f"PLAYER: {PLAYER_NAME}", 14, int(92 * ui),
              scale=0.75 * ui, color=(255,255,255), thick=2)

        if data:
            lat = data.get("latency_ms")
            if lat is not None:
                last_latency_ms = lat
        if last_latency_ms is not None:
            draw_text(frame, f"IA {last_latency_ms} ms", w - int(150 * ui), h - int(14 * ui),
                      scale=0.75 * ui, color=(255,255,255), thick=2)

        draw_text(frame, "ESC salir | R reiniciar | S sonido | M musica | F fullscreen | N nom",
                  14, h - int(60 * ui), scale=0.70 * ui, color=(255,255,255), thick=2)

        if game_over:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)
            draw_text(frame, "GAME OVER", w//2 - int(180 * ui), h//2 - int(30 * ui),
                      scale=1.6 * ui, color=(0,0,255), thick=4)
            draw_text(frame, f"TOTAL SCORE: {score}", w//2 - int(190 * ui), h//2 + int(25 * ui),
                      scale=1.0 * ui, color=(255,255,255), thick=2)
            draw_text(frame, "Press R to restart", w//2 - int(160 * ui), h//2 + int(65 * ui),
                      scale=0.9 * ui, color=(255,255,255), thick=2)

    # UI nombre (encima de TODO)
    if name_edit_mode:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)
        draw_text(frame, "ESCRIBE TU NOMBRE", w//2 - int(220*ui), h//2 - int(60*ui),
                  scale=1.2*ui, color=(255,255,255), thick=3)
        draw_text(frame, (name_buffer if name_buffer else "") + "_", w//2 - int(220*ui), h//2,
                  scale=1.1*ui, color=(0,255,255), thick=3)
        draw_text(frame, "ENTER para guardar", w//2 - int(220*ui), h//2 + int(50*ui),
                  scale=0.9*ui, color=(255,255,255), thick=2)

    # mostrar (después de dibujar todo)
    out = fit_to_window(frame, WIN)
    cv2.imshow(WIN, out)

    # teclado
    key = cv2.waitKey(1) & 0xFF

    # ====== captura de nombre (dentro del while) ======
    if name_edit_mode:
        if key in (13, 10):  # Enter
            name_buffer = name_buffer.strip()
            PLAYER_NAME = name_buffer if name_buffer else "Jugador"
            save_player_name(PLAYER_NAME)
            name_edit_mode = False
        elif key in (8, 127):  # Backspace
            name_buffer = name_buffer[:-1]
        elif 32 <= key <= 126:
            if len(name_buffer) < 16:
                name_buffer += chr(key)
        continue  # mientras escribe nombre, no ejecutar controles del juego

    # controles normales
    if key == 27:
        break
    elif key in (ord("r"), ord("R")):
        restart()
    elif key in (ord("s"), ord("S")):
        AUDIO_ENABLED = not AUDIO_ENABLED
    elif key in (ord("m"), ord("M")):
        music_toggle()
    elif key in (ord("n"), ord("N")):
        name_edit_mode = True
        name_buffer = ""
    elif key in (ord("f"), ord("F")):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

try:
    ai_q.put(None)
except Exception:
    pass

cap.release()
cv2.destroyAllWindows()
pygame.quit()
