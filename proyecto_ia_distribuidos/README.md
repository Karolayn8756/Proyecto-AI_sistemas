# Proyecto IA + Sistemas Distribuidos (Starter)

## 1) IA Service (Hands + Face)
Abrir terminal en `ai_service/`:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001
```

Luego entra a:
- http://127.0.0.1:8001/docs

## 2) Cliente cámara
Abrir terminal en `game_client/`:

```bash
pip install opencv-python requests
python client_cam.py
```

## 3) Game Server (placeholder)
Abrir terminal en `game_server/`:

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8002
```

Health:
- http://127.0.0.1:8002/health
