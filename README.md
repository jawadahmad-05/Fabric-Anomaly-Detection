# Automated Visual Anomaly Detection in Textile Fabrics

**Pixel-level SOTA on 4 texture benchmarks — zero training, zero fine-tuning, zero reference images.**

Achieves SOTA pixel AUROC using frozen DINOv2 ViT-L/14 features with 3 complementary scoring
branches (kNN + Wasserstein + Centroid) fused via homogeneity modulation. Includes farthest-point
coreset (50× reduction, +0.17%), zero-shot CLIP defect classifier (10 fabric types), and a
real-time IoT demo (phone camera → laptop inference over Wi-Fi).

## Directory Structure

```
ZSAD/
├── model/               # Core model code
│   ├── zsad_architecture.py   # ZSAD, scoring, CLIP classifier
│   └── zsad_api.py            # Simple predict API
├── scripts/             # Run scripts
│   ├── iot_server.py          # FastAPI server (IoT demo)
│   ├── evaluate_zsad.py       # Full benchmark eval
│   ├── demo_zsad.py           # Anomaly map visualization
│   └── paper_figures.py       # Publication figures
├── paper_figures/       # Generated publication figures (10 PNGs)
├── frontend/            # React app (Vite + Tailwind)
│   ├── src/                   # React source
│   ├── Backend/reports.csv    # Inspection history
│   ├── package.json
│   └── vite.config.js
├── dataset/             # Dataset configs (JSONL hub files)
├── static/              # Legacy mobile web UI
├── utils.py             # Shared utilities
└── requirements.txt
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run IoT server (FastAPI backend)

```bash
python scripts/iot_server.py
```
Serves API at `http://0.0.0.0:8000`. Phone on same Wi-Fi can connect via laptop IP.

### 3a. Run React frontend (development)

```bash
cd frontend
npm install
npm run dev
```
Opens at `http://0.0.0.0:5173`. Proxies API calls to port 8000 automatically.

### 3b. Build React frontend for production (single server)

```bash
cd frontend
npm run build
python ../scripts/iot_server.py
```
Now `http://laptop-ip:8000` serves both frontend and API.

### 4. Run benchmark evaluation

```bash
python scripts/evaluate_zsad.py --datasets mvtec
```

### 5. Quick inference demo

```bash
python scripts/demo_zsad.py
```


## Requirements

- Python 3.10+, PyTorch 2.0+, CUDA 12.1+
- DINOv2 auto-downloaded via `torch.hub`
- NVIDIA GPU with 4GB+ VRAM recommended
