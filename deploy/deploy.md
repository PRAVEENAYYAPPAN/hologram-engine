# Hologram Engine — Deployment Guide

## Table of Contents

1. [Local Development](#local-development)
2. [Backend Deployment](#backend-deployment)
3. [Frontend Deployment](#frontend-deployment)
4. [Model Handling in Production](#model-handling-in-production)
5. [GPU vs CPU Considerations](#gpu-vs-cpu-considerations)

---

## Local Development

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Linux/Mac
.\venv\Scripts\activate    # Windows

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

---

## Backend Deployment

### Option 1: Docker on VPS (Recommended)

1. **Provision a server** (at least 4GB RAM, 8GB recommended):
   - AWS EC2: t3.large or g4dn.xlarge (GPU)
   - GCP: n1-standard-2 or n1-standard-4 with T4 GPU
   - DigitalOcean: 4GB Droplet
   - Hetzner: CX31 or CX41

2. **Install Docker**:
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   ```

3. **Deploy**:
   ```bash
   git clone <your-repo> hologram-engine
   cd hologram-engine/backend
   docker-compose up -d
   ```

4. **Set up reverse proxy** (Nginx):
   ```bash
   sudo cp ../deploy/nginx.conf /etc/nginx/sites-available/hologram
   sudo ln -s /etc/nginx/sites-available/hologram /etc/nginx/sites-enabled/
   sudo nginx -t && sudo systemctl reload nginx
   ```

5. **SSL with Certbot**:
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

### Option 2: Render

1. Create a new **Web Service** on [render.com](https://render.com)
2. Connect your repository
3. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Docker**: Use the provided Dockerfile
4. Set environment variables:
   - `CORS_ORIGINS=https://your-frontend.vercel.app`

### Option 3: Railway

1. Create a new project on [railway.app](https://railway.app)
2. Deploy from GitHub repo (point to `backend/` directory)
3. Railway auto-detects the Dockerfile
4. Set environment variables as needed

### Option 4: Google Cloud Run

```bash
# Build and push container
gcloud builds submit --tag gcr.io/YOUR_PROJECT/hologram-engine

# Deploy
gcloud run deploy hologram-engine \
  --image gcr.io/YOUR_PROJECT/hologram-engine \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --allow-unauthenticated \
  --port 8000
```

---

## Frontend Deployment

The frontend is pure static files — deploy anywhere:

### Vercel

```bash
cd frontend
npx vercel
```

Or connect your GitHub repo and set:
- **Root Directory**: `frontend`
- **Framework**: None (static)
- **Build Command**: (leave empty)
- **Output Directory**: `.`

### Netlify

```bash
cd frontend
npx netlify-cli deploy --dir=. --prod
```

Or drag-and-drop the `frontend/` folder on netlify.com.

### GitHub Pages

1. Push the `frontend/` contents to a `gh-pages` branch
2. Enable Pages in repo settings

### Important: Update API URL

Before deploying the frontend, update the API base URL in `js/api.js`:

```javascript
// Change this to your deployed backend URL
const API_BASE = window.HOLOGRAM_API_BASE || 'https://your-backend.onrender.com';
```

Or set it via a script tag in `index.html`:
```html
<script>window.HOLOGRAM_API_BASE = 'https://your-backend.onrender.com';</script>
```

---

## Model Handling in Production

### Weight Caching

AI model weights are downloaded on first run:
- **YOLOv8**: ~6MB (yolov8n.pt)
- **OpenCLIP ViT-B-32**: ~340MB 
- **TripoSR** (optional): ~800MB

To avoid re-downloading on each container restart:

```yaml
# docker-compose.yml
volumes:
  - model-cache:/root/.cache
  - ./output:/app/output

volumes:
  model-cache:
```

### Generated Model Storage

Generated GLB models are stored in `output/`. For production:

1. **Local disk** (simplest): Mount a persistent volume
2. **S3/GCS bucket**: Upload models and return signed URLs
3. **CDN**: Serve models via CloudFront/Cloudflare for global performance

### Pre-warming

Pre-download models on container start:
```python
# Add to app startup
@app.on_event("startup")
async def startup():
    from .detection import get_yolo_model
    from .semantic import get_clip_model
    get_yolo_model()
    get_clip_model()
```

---

## GPU vs CPU Considerations

### CPU Deployment (Default)

- All models work on CPU
- Processing time: ~1-3 seconds per image
- Min 4GB RAM recommended
- Suitable for: demos, low-traffic apps, development

### GPU Deployment

For production with heavy traffic:

1. **Use NVIDIA Docker runtime**:
   ```bash
   # Install nvidia-container-toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt update && sudo apt install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Update Dockerfile**:
   ```dockerfile
   FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
   # ... rest of Dockerfile
   ```

3. **Update docker-compose.yml**:
   ```yaml
   services:
     backend:
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
   ```

### Cloud GPU Options

| Provider | Instance | GPU | Cost (approx) |
|----------|----------|-----|------|
| AWS | g4dn.xlarge | T4 | $0.53/hr |
| GCP | n1-standard-4 + T4 | T4 | $0.45/hr |
| Lambda Labs | 1x A10 | A10 | $0.73/hr |
| Vast.ai | Various | Various | $0.10-0.50/hr |
| RunPod | Community | Various | $0.20-0.50/hr |

---

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok", "service": "hologram-engine", "timestamp": ...}
```

### Docker Logs

```bash
docker-compose logs -f backend
```

### Recommended Monitoring Stack

- **Prometheus** + **Grafana** for metrics
- **Sentry** for error tracking
- **Uptime Robot** / **BetterUptime** for availability monitoring
