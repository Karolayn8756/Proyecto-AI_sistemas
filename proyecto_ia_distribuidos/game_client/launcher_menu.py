import os
import sys
import subprocess
import pygame

# =========================
# CONFIG
# =========================
WIN_W, WIN_H = 1280, 720
FPS = 60

BASE_DIR = os.path.dirname(__file__)

# ✅ Fondo (según tu carpeta)
BG_IMAGE = os.path.join(BASE_DIR, "MENU_LAUNCH.png")

# Archivos de tus juegos (en la MISMA carpeta game_client)
GAMES = [
    ("Smile Battle", "client_smile_battle.py"),
    ("Rock Paper Scissors", "client_rps.py"),
    ("Fruit Ninja", "client_fruit_ninja.py"),
]

# =========================
# Helpers UI
# =========================
def draw_button(screen, rect, text, font, hovered=False):
    bg = (245, 245, 245) if not hovered else (255, 250, 180)
    border = (0, 0, 0)

    pygame.draw.rect(screen, bg, rect, border_radius=16)
    pygame.draw.rect(screen, border, rect, width=4, border_radius=16)

    t = font.render(text, True, (0, 0, 0))
    screen.blit(t, t.get_rect(center=rect.center))

def run_game(script_name):
    script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(script_path):
        print("[ERROR] No existe:", script_path)
        return

    try:
        subprocess.Popen([sys.executable, script_path], cwd=BASE_DIR)
    except Exception as e:
        print("[ERROR] No pude abrir el juego:", e)

def blit_cover(screen, img, w, h):
    """
    Escala el fondo para cubrir toda la pantalla SIN deformar (tipo 'cover').
    """
    iw, ih = img.get_size()
    if iw <= 0 or ih <= 0:
        return

    scale = max(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    scaled = pygame.transform.smoothscale(img, (nw, nh))

    x = (w - nw) // 2
    y = (h - nh) // 2
    screen.blit(scaled, (x, y))

def main():
    pygame.init()

    # ✅ Ventana redimensionable (puedes maximizar)
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
    pygame.display.set_caption("Game Launcher")
    clock = pygame.time.Clock()

    # Fuentes
    font_title = pygame.font.SysFont("Arial", 60, bold=True)
    font_btn = pygame.font.SysFont("Arial", 32, bold=True)
    font_hint = pygame.font.SysFont("Arial", 22)

    # ✅ Cargar fondo
    bg_img = None
    bg_error = None
    try:
        if not os.path.exists(BG_IMAGE):
            bg_error = f"NO ENCONTRÉ: {BG_IMAGE}"
        else:
            bg_img = pygame.image.load(BG_IMAGE).convert()
    except Exception as e:
        bg_error = f"ERROR cargando fondo: {e}"

    # Layout inicial
    w, h = screen.get_size()

    def build_buttons(w, h):
        btn_w = int(w * 0.55)
        btn_h = 80
        gap = 22
        start_y = int(h * 0.32)

        buttons = []
        for i, (label, script) in enumerate(GAMES):
            x = w // 2 - btn_w // 2
            y = start_y + i * (btn_h + gap)
            buttons.append((pygame.Rect(x, y, btn_w, btn_h), label, script))
        return buttons

    buttons = build_buttons(w, h)

    running = True
    while running:
        dt = clock.tick(FPS)
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.VIDEORESIZE:
                # ✅ Recalcular botones al redimensionar
                w, h = event.w, event.h
                screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                buttons = build_buttons(w, h)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for rect, label, script in buttons:
                    if rect.collidepoint(mx, my):
                        run_game(script)

        # ============ DRAW ============
        if bg_img is not None:
            blit_cover(screen, bg_img, w, h)
        else:
            # fallback morado si no carga
            screen.fill((40, 10, 70))

        # Si el fondo no carga, muestro el error en pantalla (para que lo arregles rápido)
        if bg_error:
            err = font_hint.render(bg_error, True, (255, 100, 100))
            screen.blit(err, (20, 20))
            err2 = font_hint.render("Revisa que MENU_LAUNCH.png esté en game_client/assets/", True, (255, 200, 200))
            screen.blit(err2, (20, 46))

        title = font_title.render("MAIN MENU", True, (255, 255, 255))
        screen.blit(title, title.get_rect(center=(w // 2, int(h * 0.16))))

        for rect, label, script in buttons:
            hovered = rect.collidepoint(mx, my)
            draw_button(screen, rect, label, font_btn, hovered=hovered)

        # ✅ Hint abajo (si lo quieres negro, cambia (255,255,255) por (0,0,0))
        hint = font_hint.render("Click para abrir un juego  |  ESC para salir", True, (255, 255, 255))
        screen.blit(hint, hint.get_rect(midbottom=(w // 2, h - 20)))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
