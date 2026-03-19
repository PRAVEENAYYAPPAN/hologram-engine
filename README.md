# 🔮 Hologram Engine

A complete, open-source, AI-powered hologram generation system. Input any image or camera feed — the system detects objects, understands them semantically, retrieves or generates a matching 3D model, and renders it as a holographic visualization using Three.js.

## Architecture

```
Image/Camera → YOLOv8 Detection → OpenCLIP Semantics → 3D Retrieval/Generation → Three.js Hologram
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| Semantic Understanding | OpenCLIP (ViT-B-32) |
| 3D Model Retrieval | Objaverse + CLIP embeddings |
| 3D Model Generation | TripoSR (fallback: procedural GLB) |
| Backend API | FastAPI + Uvicorn |
| Frontend Rendering | Three.js + Custom Holographic Shaders |
| Deployment | Docker + Nginx |

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (optional, for frontend dev server)
- 8GB+ RAM recommended
- GPU optional (CUDA support auto-detected)

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

### 2. Frontend Setup

The frontend is pure HTML/CSS/JS — just serve it with any static server:

```bash
cd frontend

# Option A: Python's built-in server
python -m http.server 3000

# Option B: Node.js live-server
npx live-server --port=3000

# Option C: Open index.html directly in a browser
```

Visit `http://localhost:3000` to use the application.

### 3. First Run

1. Start the backend server (Step 1)
2. Open the frontend (Step 2)
3. Wait for "Backend Online" status indicator
4. Upload an image or use the camera
5. Click "GENERATE HOLOGRAM"
6. Watch the AI pipeline process your image and generate a holographic model!

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /health` | GET | Health check |
| `POST /detect` | POST | Detect objects + semantic classification |
| `POST /retrieve-model` | POST | Retrieve matching 3D model from catalog |
| `POST /generate-model` | POST | Generate 3D model from image |
| `POST /pipeline` | POST | Full pipeline (detect → retrieve → generate) |
| `GET /models/{filename}` | GET | Serve generated model files |

### Example: Full Pipeline

```bash
curl -X POST http://localhost:8000/pipeline \
  -F "file=@spiderman.jpg" | python -m json.tool
```

Response:
```json
{
    "success": true,
    "detection": {
        "yolo_label": "person",
        "confidence": 0.89,
        "semantic_class": "Spider-Man",
        "semantic_confidence": 0.8234,
        "top_5_semantic": [...]
    },
    "model": {
        "url": "/models/spider_man_abc123.glb",
        "method": "procedural",
        "match_name": "Spider-Man",
        "match_similarity": 0.7891
    },
    "processing_time_seconds": 2.45
}
```

## Project Structure

```
hologram-engine/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application & endpoints
│   │   ├── config.py        # Configuration & constants
│   │   ├── detection.py     # YOLOv8 object detection
│   │   ├── semantic.py      # OpenCLIP semantic understanding
│   │   ├── retrieval.py     # 3D model retrieval (Objaverse)
│   │   └── generation.py    # 3D model generation (TripoSR/procedural)
│   ├── models/              # Downloaded model weights (auto-cached)
│   ├── output/              # Generated 3D model files
│   ├── catalog/             # Pre-computed model catalog
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
├── frontend/
│   ├── index.html           # Main HTML
│   ├── css/style.css        # Design system & styles
│   └── js/
│       ├── app.js           # Main application logic
│       ├── api.js           # Backend API client
│       ├── camera.js        # Camera capture module
│       └── hologram.js      # Three.js hologram renderer
├── deploy/
│   ├── nginx.conf           # Nginx reverse proxy config
│   └── deploy.md            # Deployment instructions
└── README.md
```

## Deployment

### Docker (Backend)

```bash
cd backend
docker build -t hologram-engine .
docker run -p 8000:8000 -v ./models:/app/models -v ./output:/app/output hologram-engine
```

### Docker Compose

```bash
cd backend
docker-compose up -d
```

### Frontend (Static Hosting)

Deploy the `frontend/` directory to:
- **Vercel**: `npx vercel frontend/`
- **Netlify**: Drag-and-drop `frontend/` or `netlify deploy --dir=frontend`
- **GitHub Pages**: Push `frontend/` to `gh-pages` branch

### Full Deployment Guide

See [deploy/deploy.md](deploy/deploy.md) for complete production deployment instructions.

## Performance

| Component | CPU Time | GPU Time |
|-----------|---------|---------|
| YOLOv8 Detection | ~200ms | ~30ms |
| OpenCLIP Embedding | ~300ms | ~50ms |
| Semantic Classification | ~100ms | ~20ms |
| Model Generation | ~500ms | ~200ms |
| **Total Pipeline** | **~1.1s** | **~300ms** |

## GPU vs CPU

- The system **auto-detects CUDA** and uses GPU if available
- All models work on **CPU** (slower but fully functional)
- For production, a GPU instance (NVIDIA T4 or better) is recommended

## License

MIT License. All components are open-source:
- YOLOv8: AGPL-3.0
- OpenCLIP: MIT
- Objaverse: ODC-BY-1.0
- Three.js: MIT
- FastAPI: MIT
