"""Genera un video de demo con el polígono de cada asiento y su estado
(libre/ocupado/reservado) superpuestos, usando el clasificador entrenado
(models/weights/seat_classifier.pt) en vez de YOLO.

Uso:
    python src/render_demo_classifier.py
    python src/render_demo_classifier.py --start 3293 --duration 72 --fps 8 --output data/processed/mi_demo.mp4
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from config import load_config
from seat_dataset import STATUSES, seat_bboxes
from train_classifier import IMAGENET_MEAN, IMAGENET_STD, build_model

STATUS_COLOR = {
    "libre": (40, 180, 99),
    "ocupado": (230, 126, 34),
    "reservado": (155, 60, 140),
}


def _format_ts(seconds):
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}m{s:02d}s"


def _load_font(size=20):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_demo(video_path, start=0.0, duration=60.0, fps=8.0,
                 config_path="config/pipeline_config.yaml",
                 model_path="models/weights/seat_classifier.pt",
                 output_path=None, tmp_dir="output/_demo_clf_tmp"):
    cfg = load_config(config_path)
    width = cfg["sampling"]["frame_width"]
    bboxes = seat_bboxes(cfg["seats_config"])
    seats_geom = json.load(open(cfg["seats_config"], encoding="utf-8"))

    if output_path is None:
        output_path = f"data/processed/occupancy_demo_classifier_{_format_ts(start)}.mp4"

    raw_dir = os.path.join(tmp_dir, "raw")
    ann_dir = os.path.join(tmp_dir, "annotated")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    print(f"Extrayendo {duration:.0f}s desde t={start:.0f}s a {fps:.0f} fps...", flush=True)
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start), "-i", video_path, "-t", str(duration),
        "-vf", f"fps={fps},scale={width}:-2", "-qscale:v", "2",
        os.path.join(raw_dir, "f_%06d.jpg"),
    ], check=True, capture_output=True)

    frame_files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".jpg"))
    print(f"{len(frame_files)} frames extraídos.", flush=True)

    print("Cargando clasificador...", flush=True)
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    infer_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    font = _load_font(20)

    print("Clasificando y anotando frames...", flush=True)
    for i, fname in enumerate(frame_files):
        image = Image.open(os.path.join(raw_dir, fname)).convert("RGB")

        crops = []
        for seat in seats_geom:
            crop = image.crop(bboxes[seat["seat_id"]])
            crops.append(infer_tf(crop))
        batch = torch.stack(crops)

        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

        draw = ImageDraw.Draw(image)
        for seat, pred_idx, conf in zip(seats_geom, preds.tolist(), confs.tolist()):
            status = STATUSES[pred_idx]
            color = STATUS_COLOR[status]
            pts = [tuple(p) for p in seat["polygon"]]
            draw.line(pts + [pts[0]], fill=color, width=4)
            label = f"{seat['seat_id']} · {status} ({conf:.2f})"
            cx = sum(p[0] for p in pts) / len(pts)
            cy = min(p[1] for p in pts)
            tw = draw.textlength(label, font=font)
            draw.rectangle([cx - tw / 2 - 6, cy - 4, cx + tw / 2 + 6, cy + 26], fill=(0, 0, 0))
            draw.text((cx - tw / 2, cy), label, fill=color, font=font)

        image.save(os.path.join(ann_dir, fname))

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(frame_files)} anotados", flush=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Codificando video de salida...", flush=True)
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(ann_dir, "f_%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_path,
    ], check=True, capture_output=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Video guardado en {output_path}", flush=True)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/raw/WIN_20260828_18_42_07_Pro.mp4")
    parser.add_argument("--start", type=float, default=3293.0)
    parser.add_argument("--duration", type=float, default=72.0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--model", default="models/weights/seat_classifier.pt")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    render_demo(args.video, start=args.start, duration=args.duration, fps=args.fps,
                model_path=args.model, output_path=args.output)
