"""FallNet Thermal Fall Detection - Web Application Server

Provides an interactive web interface to test any folder (preprocessed or raw)
with real-time sliding-window prediction, timeline visualization, and keyframe inspection.

Run:
  python web_app/app.py
  Open in browser: http://127.0.0.1:5000
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import base64
import json
from io import BytesIO
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch
from flask import Flask, render_template, request, jsonify

import config as C
from data_model import get_model, MEAN, STD

app = Flask(__name__, template_folder="templates", static_folder="static")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = PROJECT_ROOT / "work" / "checkpoints"
CACHE_DIR = PROJECT_ROOT / "work" / "cache"

MODELS = {}
THRESHOLDS = {}


def load_all_models():
    """Preload trained models into memory."""
    global MODELS, THRESHOLDS
    print(f"Loading FallNet models onto {DEVICE} from {CKPT_DIR}...")

    # ResNet-GRU
    try:
        m1 = get_model("resnet_gru", pretrained=False).to(DEVICE)
        pt1 = CKPT_DIR / "resnet_gru_best.pt"
        if not pt1.exists():
            pt1 = CKPT_DIR / "best.pt"
        m1.load_state_dict(torch.load(pt1, map_location=DEVICE))
        m1.eval()
        MODELS["resnet_gru"] = m1
        js1 = CKPT_DIR / "resnet_gru_best.json"
        if not js1.exists():
            js1 = CKPT_DIR / "best.json"
        t1 = json.load(open(js1)).get("threshold", 0.002) if js1.exists() else 0.002
        THRESHOLDS["resnet_gru"] = t1
        print("  [OK] Loaded ResNet18 + Bi-GRU")
    except Exception as e:
        print(f"  [WARN] Failed to load ResNet-GRU: {e}")

    # Transformer
    try:
        m2 = get_model("transformer", pretrained=False).to(DEVICE)
        pt2 = CKPT_DIR / "transformer_best.pt"
        m2.load_state_dict(torch.load(pt2, map_location=DEVICE))
        m2.eval()
        MODELS["transformer"] = m2
        js2 = CKPT_DIR / "transformer_best.json"
        t2 = json.load(open(js2)).get("threshold", 0.005) if js2.exists() else 0.005
        THRESHOLDS["transformer"] = t2
        print("  [OK] Loaded MobileNetV3 + Transformer")
    except Exception as e:
        print(f"  [WARN] Failed to load Transformer: {e}")

    # R3D
    try:
        m3 = get_model("r3d", pretrained=False).to(DEVICE)
        pt3 = CKPT_DIR / "r3d_best.pt"
        if pt3.exists():
            m3.load_state_dict(torch.load(pt3, map_location=DEVICE))
            m3.eval()
            MODELS["r3d"] = m3
            js3 = CKPT_DIR / "r3d_best.json"
            t3 = json.load(open(js3)).get("threshold", 0.01) if js3.exists() else 0.01
            THRESHOLDS["r3d"] = t3
            print("  [OK] Loaded R3D-18 Spatiotemporal CNN")
    except Exception as e:
        print(f"  [WARN] Failed to load R3D: {e}")

    # Ensemble Threshold (from benchmark)
    THRESHOLDS["ensemble"] = 0.006


def preprocess_raw_folder(folder_path: Path):
    """Preprocess raw image folder on the fly if not cached."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted([p for p in folder_path.iterdir() if p.suffix.lower() in exts])
    if len(img_paths) < C.WIN:
        raise ValueError(f"Folder contains only {len(img_paths)} frames (minimum required: {C.WIN})")

    resized = []
    for p in img_paths:
        with Image.open(p) as img:
            im = img.convert("RGB").resize((C.IMG_SIZE, C.IMG_SIZE), Image.BILINEAR)
            resized.append(np.asarray(im, dtype=np.uint8))
    arr = np.stack(resized)
    T = len(arr)

    # Quantile normalization on CPU
    rgb_list = []
    for i in range(T):
        x = arr[i].astype(np.float32)
        lo, hi = np.percentile(x, C.P_LOW), np.percentile(x, C.P_HIGH)
        if hi - lo < 1e-3:
            rgb_list.append(np.zeros_like(x, dtype=np.uint8))
        else:
            x = np.clip((x - lo) / (hi - lo), 0, 1)
            rgb_list.append((x * 255).astype(np.uint8))
    rgb = np.stack(rgb_list)

    # Differential motion channel
    g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    m = np.array([frame.mean() / 255.0 for frame in g])
    gain = np.abs(np.diff(m, prepend=m[0])) > C.GAIN_JUMP_THRESH
    good = g.copy()
    for t in range(1, T):
        if gain[t]:
            good[t] = good[t - 1]
    bg = np.median(good, axis=0)
    motion = np.zeros_like(good, dtype=np.uint8)
    for t in range(T):
        a = np.abs(good[t] - bg)
        b = np.abs(good[t] - good[max(0, t - C.LAG)])
        motion[t] = np.clip(np.maximum(a, b) * 2.0, 0, 255).astype(np.uint8)

    return rgb, motion


def load_or_process_folder(folder_query: str):
    """Find preprocessed cache or process from raw thermal folder."""
    folder_name = Path(folder_query).name
    # 1. Try finding in cache
    cache_matches = list(CACHE_DIR.glob(f"*_{folder_name}.npz"))
    if not cache_matches:
        for p in CACHE_DIR.glob("*.npz"):
            if folder_name.lower() in p.stem.lower():
                cache_matches.append(p)
                break

    if cache_matches:
        cache_file = cache_matches[0]
        with np.load(cache_file) as z:
            return z["rgb"], z["motion"], cache_file.name, True

    # 2. Check direct filesystem path
    direct_path = Path(folder_query)
    if direct_path.exists() and direct_path.is_dir():
        rgb, motion = preprocess_raw_folder(direct_path)
        return rgb, motion, direct_path.name, False

    # 3. Check inside Thermal root (fall/ or nonfall/)
    for sub in ("fall", "nonfall"):
        p = C.THERMAL_ROOT / sub / folder_name
        if p.exists() and p.is_dir():
            rgb, motion = preprocess_raw_folder(p)
            return rgb, motion, p.name, False

    raise FileNotFoundError(f"Folder '{folder_query}' not found in cache or on disk.")


def frame_to_base64(arr_uint8):
    """Convert numpy array (H, W, 3) or (H, W) to base64 JPEG."""
    if arr_uint8.ndim == 2:
        im = Image.fromarray(arr_uint8, mode="L").convert("RGB")
    else:
        im = Image.fromarray(arr_uint8)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return list of quick-test sample folders."""
    samples = [
        {"name": "Fall14", "label": "Fall (Test Participant 14)", "type": "fall"},
        {"name": "Fall25", "label": "Fall (Test Participant 25)", "type": "fall"},
        {"name": "Fall26", "label": "Fall (Test Participant 26)", "type": "fall"},
        {"name": "Fall0", "label": "Fall (General Sample 0)", "type": "fall"},
        {"name": "Fall5", "label": "Fall (General Sample 5)", "type": "fall"},
        {"name": "NonFall10", "label": "Non-Fall (ADL Walk/Stand)", "type": "nonfall"},
        {"name": "NonFall25", "label": "Non-Fall (ADL Sitting)", "type": "nonfall"},
        {"name": "NonFall50", "label": "Non-Fall (Normal Activity)", "type": "nonfall"},
    ]
    return jsonify({"samples": samples, "models_available": list(MODELS.keys()) + ["ensemble"]})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json or {}
    folder_query = data.get("folder", "").strip()
    arch = data.get("model", "ensemble").lower().strip()

    if not folder_query:
        return jsonify({"error": "Please provide a folder name or path."}), 400

    start_time = time.time()

    try:
        rgb, motion, loaded_name, is_cached = load_or_process_folder(folder_query)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    T = len(rgb)
    stride = 4
    starts = list(range(0, T - C.WIN + 1, stride))
    if not starts:
        starts = [0]

    # Build windows batch
    batch_x = []
    for s in starts:
        e = s + C.WIN
        if e > T:
            e = T
            s = max(0, e - C.WIN)
        x = np.concatenate([rgb[s:e], motion[s:e, ..., None]], axis=-1)
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
        x = (x - MEAN) / STD
        batch_x.append(x)
    batch_tensor = torch.stack(batch_x).to(DEVICE)

    # Predict
    with torch.no_grad():
        if arch == "ensemble" and "resnet_gru" in MODELS and "transformer" in MODELS:
            p1 = torch.sigmoid(MODELS["resnet_gru"](batch_tensor)).cpu().numpy()
            p2 = torch.sigmoid(MODELS["transformer"](batch_tensor)).cpu().numpy()
            probs = (p1 + p2) / 2.0
            threshold = THRESHOLDS["ensemble"]
            model_disp_name = "2-Way Hybrid Ensemble (ResNet + Transformer) [Best]"
        elif arch in MODELS:
            probs = torch.sigmoid(MODELS[arch](batch_tensor)).cpu().numpy()
            threshold = THRESHOLDS.get(arch, 0.5)
            model_disp_name = arch.upper()
        else:
            # Fallback to whatever model is available
            avail_key = next(iter(MODELS.keys()))
            probs = torch.sigmoid(MODELS[avail_key](batch_tensor)).cpu().numpy()
            threshold = THRESHOLDS.get(avail_key, 0.5)
            model_disp_name = avail_key

    # Timeline & Hysteresis Smoothing
    timeline = []
    for s, p in zip(starts, probs):
        timeline.append({"start": int(s), "end": int(s + C.WIN), "prob": float(p)})

    # Hysteresis & Temporal Persistence:
    # A true fall persists across consecutive windows; single-window isolated spikes are rejected
    smoothed_alarms = []
    for i in range(len(probs)):
        is_fall_raw = probs[i] >= threshold
        prev_win = probs[i - 1] >= threshold if i > 0 else False
        next_win = probs[i + 1] >= threshold if i < len(probs) - 1 else False
        smoothed_alarms.append(is_fall_raw and (prev_win or next_win))

    is_fall_detected = any(smoothed_alarms)
    peak_prob = float(np.max(probs))
    peak_idx = int(np.argmax(probs))
    peak_start = starts[peak_idx]
    peak_frame_num = min(T - 1, peak_start + C.WIN // 2)

    # Encode peak frame thumbnail
    peak_rgb_b64 = frame_to_base64(rgb[peak_frame_num])
    peak_motion_b64 = frame_to_base64(motion[peak_frame_num])

    elapsed_ms = int((time.time() - start_time) * 1000)

    return jsonify({
        "folder": loaded_name,
        "is_cached": is_cached,
        "total_frames": T,
        "total_windows": len(starts),
        "model_used": model_disp_name,
        "decision": "FALL DETECTED" if is_fall_detected else "NORMAL",
        "is_fall": bool(is_fall_detected),
        "peak_prob": peak_prob,
        "threshold": float(threshold),
        "peak_frame": peak_frame_num,
        "latency_ms": elapsed_ms,
        "timeline": timeline,
        "peak_rgb_b64": peak_rgb_b64,
        "peak_motion_b64": peak_motion_b64
    })


if __name__ == "__main__":
    load_all_models()
    port = 5000
    print(f"\n============================================================")
    print(f"  FallNet Web Application running on: http://127.0.0.1:{port}")
    print(f"============================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
