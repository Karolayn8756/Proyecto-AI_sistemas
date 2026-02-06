import time
import json
from pathlib import Path
from typing import Dict, Any, List
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI(title="Game Server")

# =========================
#  RANKING (mejor score por jugador y juego)
# =========================
# Guardamos el mejor score por (game, player)
BEST: Dict[str, Dict[str, int]] = {}  # BEST[game][player] = score

# Historial opcional (si quieres debug)
RANKING_LOG: List[Dict[str, Any]] = []

@app.get("/health")
def health():
    return {"ok": True, "service": "game_server"}

@app.post("/score")
def add_score(item: Dict[str, Any]):
    """
    Espera:
    {
      "player": "Emily",
      "game": "rps" | "smile" | "fruit_ninja",
      "score": 120,
      "extra": {...}   (opcional)
    }

    ✅ Guarda SOLO el mejor score por jugador en ese juego.
    ✅ "IA" también cuenta como player si lo mandas.
    """
    player = str(item.get("player", "unknown")).strip() or "unknown"
    game = str(item.get("game", "unknown")).strip() or "unknown"
    score = int(item.get("score", 0) or 0)

    if game not in BEST:
        BEST[game] = {}

    prev = BEST[game].get(player)
    if prev is None or score > prev:
        BEST[game][player] = score

    # log opcional
    RANKING_LOG.append({
        "ts": int(time.time()),
        "player": player,
        "game": game,
        "score": score,
        "extra": item.get("extra", {})
    })

    return {"saved": True, "best_score": BEST[game][player]}

@app.get("/ranking")
def get_ranking():
    """
    Devuelve el ranking "normalizado" para el dashboard:
    lista de items con (player, game, score)
    """
    out: List[Dict[str, Any]] = []
    for game, players in BEST.items():
        for player, score in players.items():
            out.append({"player": player, "game": game, "score": score})

    # orden global (por score desc)
    out.sort(key=lambda x: x["score"], reverse=True)
    return {"ranking": out[:100]}  # el dashboard filtra por juego

@app.websocket("/ws/ranking")
async def ws_ranking(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"info": "WebSocket conectado (ranking real-time opcional)."})
    await ws.close()

# =========================
#  VIDEOS (YouTube) persistentes
# =========================
VIDEOS_PATH = Path(__file__).resolve().parent / "videos.json"

def _load_videos() -> List[Dict[str, Any]]:
    if not VIDEOS_PATH.exists():
        return []
    try:
        return json.loads(VIDEOS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

def _save_videos(videos: List[Dict[str, Any]]):
    VIDEOS_PATH.write_text(json.dumps(videos, ensure_ascii=False, indent=2), encoding="utf-8")

def _youtube_id(url: str) -> str:
    """
    Acepta:
    - https://www.youtube.com/watch?v=VIDEOID
    - https://youtu.be/VIDEOID
    - https://www.youtube.com/shorts/VIDEOID
    """
    url = (url or "").strip()
    if not url:
        return ""

    u = urlparse(url)

    # youtu.be/VIDEOID
    if "youtu.be" in u.netloc:
        return u.path.strip("/").split("/")[0].strip()

    # youtube.com/watch?v=VIDEOID
    if "youtube.com" in u.netloc:
        if u.path.startswith("/watch"):
            q = parse_qs(u.query)
            return (q.get("v", [""])[0]).strip()
        if u.path.startswith("/shorts/"):
            return u.path.split("/shorts/")[1].split("/")[0].strip()

    return ""

@app.get("/videos")
def list_videos():
    return {"videos": _load_videos()}

@app.post("/videos")
def add_video(item: Dict[str, Any]):
    """
    Espera:
    {"title":"Video chistoso", "url":"https://youtu.be/xxxx"}
    """
    title = str(item.get("title", "Video")).strip() or "Video"
    url = str(item.get("url", "")).strip()
    vid = _youtube_id(url)

    if not vid:
        return {"saved": False, "error": "Link inválido. Usa watch?v=..., youtu.be/... o /shorts/..."}

    videos = _load_videos()

    # no duplicados
    for v in videos:
        if v.get("id") == vid:
            return {"saved": True, "id": vid, "note": "Ya existía"}

    videos.append({"id": vid, "title": title, "url": url, "ts": int(time.time())})
    _save_videos(videos)
    return {"saved": True, "id": vid}

@app.delete("/videos/{video_id}")
def delete_video(video_id: str):
    videos = _load_videos()
    new_list = [v for v in videos if v.get("id") != video_id]
    _save_videos(new_list)
    return {"deleted": True, "before": len(videos), "after": len(new_list)}

# =========================
#  DASHBOARD + MENU
# =========================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    base_dir = Path(__file__).resolve().parent
    html_path = base_dir / "dashboard.html"
    if not html_path.exists():
        return HTMLResponse(
            f"<h2>No encontré dashboard.html</h2><pre>{html_path}</pre>",
            status_code=200
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/", response_class=HTMLResponse)
def menu():
    return HTMLResponse("""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>IA SD - Arcade</title>
<style>
  body{margin:0;font-family:Arial;background:#0b1020;color:#fff;}
  .wrap{max-width:980px;margin:0 auto;padding:28px 16px;}
  .title{font-size:34px;font-weight:900;color:#ffe36e;text-shadow:0 3px 0 #000;}
  .sub{opacity:.85;margin-top:6px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-top:18px;}
  .card{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:16px;padding:16px;box-shadow:0 10px 20px rgba(0,0,0,.25);}
  .btn{display:inline-block;background:#ff77c8;color:#0b1020;padding:10px 14px;border-radius:12px;font-weight:800;text-decoration:none;margin-right:10px;}
  .btn2{background:#7cf7d4;}
  .hint{font-size:13px;opacity:.85;margin-top:10px;}
</style>
</head>
<body>
  <div class="wrap">
    <div class="title">IA SD - ARCADE</div>
    <div class="sub">Corre los juegos desde Python y mira ranking + videos en el dashboard.</div>

    <div style="margin-top:14px">
      <a class="btn2 btn" href="/dashboard">📊 Dashboard / Ranking</a>
      <a class="btn" href="/health">✅ Health</a>
    </div>

    <div class="grid">
      <div class="card">
        <h3>🍉 Fruit Ninja</h3>
        <div class="hint">Se ejecuta: <b>python game_client/client_fruit_ninja.py</b></div>
      </div>
      <div class="card">
        <h3>😶 Smile Battle</h3>
        <div class="hint">Se ejecuta: <b>python game_client/client_smile_battle.py</b></div>
      </div>
      <div class="card">
        <h3>✌️ RPS</h3>
        <div class="hint">Se ejecuta: <b>python game_client/client_rps.py</b></div>
      </div>
    </div>
  </div>
</body>
</html>
""")
