"""
ZSAD IoT Server — Phone browser → Laptop inference
===================================================
Usage:
    python iot_server.py
    Then open the printed URL on any device on the same Wi-Fi.
"""
import sys, os, socket, io, base64, csv, uuid, json
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from model.zsad_architecture import ZSAD, CLIPDefectClassifier

app = FastAPI(title="ZSAD Fabric Inspector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
HERE = Path(__file__).parent
STATIC_DIR = HERE.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
UPLOADS_DIR = STATIC_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
REPORTS_CSV = HERE.parent / "frontend" / "Backend" / "reports.csv"
REACT_DIST = HERE.parent / "frontend" / "dist"

print("[ZSAD IoT] Loading DINOv2 model...")
model = ZSAD(device='cuda', pca_dim=64, weights=(0.5, 0.3, 0.2), k_knn=80, block_indices=(11,))
print("[ZSAD IoT] Model ready!")

# ---------------------------------------------------------------------------
# Defect classifier — uses CLIP zero-shot (much better than hand-crafted rules)
# ---------------------------------------------------------------------------
print("[ZSAD IoT] Loading CLIP classifier...")
_classifier = CLIPDefectClassifier()
print("[ZSAD IoT] CLIP classifier ready!")

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def get_laptop_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

@app.get("/", response_class=HTMLResponse)
async def index():
    react_index = REACT_DIST / "index.html"
    if react_index.exists():
        return HTMLResponse(react_index.read_text(encoding='utf-8'))
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding='utf-8'))
    return HTMLResponse("<h1>ZSAD Server Running</h1><p>React build not found. Run 'cd frontend && npm run build' to generate it.</p>")

# Serve React's built assets (JS, CSS, etc.)
from fastapi.staticfiles import StaticFiles
if REACT_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(REACT_DIST / "assets")), name="react_assets")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    score_map, img_score, h_score = model.score(img)

    sm = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
    sm_uint8 = (sm * 255).astype(np.uint8)
    sm_pil = Image.fromarray(sm_uint8, mode='L')
    buf = io.BytesIO()
    sm_pil.save(buf, format='PNG')
    sm_b64 = base64.b64encode(buf.getvalue()).decode()

    defect_score = img_score * (1.3 - h_score * 0.5)

    # Always run classifier — CLIP zero-shot classification
    try:
        defect_name, defect_conf = _classifier.classify_simple(img)
    except Exception as e:
        print(f"[ZSAD IoT] Classifier error: {e}")
        import traceback
        traceback.print_exc()
        defect_name, defect_conf = "Float", min(0.5, defect_score)

    is_anomaly = defect_name != "Good Fabric"
    if is_anomaly and defect_score < 0.50:
        # Very low score but classifier says defect — still trust classifier
        pass
    if not is_anomaly and defect_score > 0.75:
        # Very high score but classifier says good — still trust classifier
        pass

    severity = "high" if defect_score > 0.78 else "medium" if defect_score > 0.55 else "low"
    if not is_anomaly:
        severity = "none"

    print(f"[ZSAD IoT] score={defect_score:.3f}  anomaly={is_anomaly}  name={defect_name}  conf={defect_conf}")

    return JSONResponse({
        "img_score": round(float(img_score), 4),
        "h_score": round(float(h_score), 4),
        "defect_score": round(float(defect_score), 4),
        "severity": severity,
        "is_anomaly": is_anomaly,
        "defect_name": defect_name,
        "defect_confidence": defect_conf,
        "score_map_shape": list(score_map.shape),
        "score_map_png_base64": sm_b64,
    })

@app.get("/all_reports")
async def all_reports():
    if not REPORTS_CSV.exists():
        return JSONResponse([])
    rows = []
    with open(REPORTS_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return JSONResponse(rows)

@app.post("/save_report")
async def save_report(file: UploadFile = File(...), defect_name: str = Form(""), defect_confidence: float = Form(0.0), defect_score: float = Form(0.0)):
    report_id = uuid.uuid4().hex[:8]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    img_path = UPLOADS_DIR / f"res_{report_id}.jpg"
    contents = await file.read()
    img_path.write_bytes(contents)
    img_url = f"http://127.0.0.1:8000/static/uploads/res_{report_id}.jpg"
    defect_type = defect_name if defect_name else "Unknown"
    new_row = {
        "ID": report_id,
        "Timestamp": now,
        "Defect_Type": defect_type,
        "Confidence": str(defect_confidence),
        "Image_Url": img_url,
    }
    header = ["ID", "Timestamp", "Defect_Type", "Confidence", "Image_Url"]
    file_exists = REPORTS_CSV.exists()
    with open(REPORTS_CSV, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(new_row)
    return JSONResponse({"status": "saved", "id": report_id})

@app.get("/health")
async def health():
    return {"status": "ok"}

# Serve uploads directory for heatmap images
from fastapi.staticfiles import StaticFiles
STATIC_DIR_ABS = str(STATIC_DIR.absolute())
app.mount("/static", StaticFiles(directory=STATIC_DIR_ABS), name="static")

# SPA catch-all for React Router paths
@app.get("/app", response_class=HTMLResponse)
@app.api_route("/app/{rest:path}", methods=["GET"], response_class=HTMLResponse)
async def spa_app(rest: str = ""):
    react_index = REACT_DIST / "index.html"
    if react_index.exists():
        return HTMLResponse(react_index.read_text(encoding='utf-8'))
    return HTMLResponse("<h1>React build not found</h1>")

def main():
    port = 8000
    ip = get_laptop_ip()
    print(f"\n{'='*60}")
    print(f"  ZSAD Fabric Inspector — IoT Server")
    print(f"{'='*60}")
    if REACT_DIST.exists():
        print(f"  React frontend    :  http://{ip}:{port}")
    else:
        print(f"  React frontend    :  not built (cd frontend && npm run build)")
        print(f"  Vite dev server   :  http://{ip}:5173")
    print(f"  API server        :  http://{ip}:{port}/predict")
    print(f"  Reports API       :  http://{ip}:{port}/all_reports")
    print(f"  Old web UI        :  http://{ip}:{port}/static/index.html")
    print(f"  Make sure phone is on the SAME Wi-Fi network ({ip.replace('.', '.')}.x)")
    print(f"  If not reachable, check Windows Firewall -> allow port {port}")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

if __name__ == "__main__":
    main()
