"""Genera un video de demo con las cajas de detección y el estado de cada
asiento (libre/ocupado/reservado) superpuestos, sobre un tramo del video
original. Usa la misma lógica de clasificación que occupancy_detect.py.

Uso:
    python src/render_demo.py
    python src/render_demo.py --start 3293 --duration 72 --fps 8 --output data/processed/mi_demo.mp4
"""

import argparse
import os
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
from config import load_config
from occupancy_detect import PERSON_CLASS, OBJECT_CLASSES, _load_seats, _best_match, _classify_seat

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
                 output_path=None, tmp_dir="output/_demo_tmp"):
    cfg = load_config(config_path)
    seats_cfg_path = cfg["seats_config"]
    conf_threshold = cfg["detection"]["confidence_threshold"]
    object_conf_threshold = cfg["detection"].get("object_confidence_threshold", 0.25)
    weights_path = cfg["detection"].get("weights", "models/weights/yolov8n.pt")
    conf_margin = cfg["refinement"]["confidence_margin"]
    min_overlap = cfg.get("matching", {}).get("min_seat_overlap", 0.25)
    width = cfg["sampling"]["frame_width"]

    if output_path is None:
        output_path = f"data/processed/occupancy_demo_{_format_ts(start)}.mp4"

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

    model = YOLO(weights_path)
    name_to_id = {v: k for k, v in model.names.items()}
    person_id = name_to_id[PERSON_CLASS]
    object_ids = [name_to_id[c] for c in OBJECT_CLASSES if c in name_to_id]
    classes_of_interest = [person_id] + object_ids
    low_conf = max(0.01, min(conf_threshold - conf_margin, object_conf_threshold) - 0.05)

    frame_paths = [os.path.join(raw_dir, f) for f in frame_files]

    print("Detección principal...", flush=True)
    all_results = model.predict(source=frame_paths, classes=classes_of_interest, conf=low_conf, verbose=False)

    seats = None
    font = _load_font(20)

    print("Clasificando y anotando frames...", flush=True)
    for i, (fname, results) in enumerate(zip(frame_files, all_results)):
        h, w = results.orig_shape
        if seats is None:
            seats = _load_seats(seats_cfg_path, w, h)

        person_dets, object_dets = [], []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            (person_dets if cls_id == person_id else object_dets).append((conf, xyxy))

        # Copia "limpia" para recortes de refinamiento: nunca se dibuja sobre ella,
        # para no contaminar el recorte con nuestras propias cajas/etiquetas.
        image_clean = Image.fromarray(results.orig_img[:, :, ::-1]).convert("RGB")
        seat_predictions = []
        for seat in seats:
            status, confidence, refined = _classify_seat(
                seat, person_dets, object_dets, model, image_clean, person_id,
                min_overlap, conf_threshold, conf_margin, object_conf_threshold, low_conf,
            )
            seat_predictions.append((seat, status, confidence))

        image = image_clean.copy()
        draw = ImageDraw.Draw(image)

        for conf, box in person_dets:
            draw.rectangle(box, outline=(255, 255, 255), width=1)
        for conf, box in object_dets:
            draw.rectangle(box, outline=(255, 220, 0), width=1)

        for seat, status, confidence in seat_predictions:
            color = STATUS_COLOR[status]
            pts = [tuple(p) for p in seat["polygon"]]
            draw.line(pts + [pts[0]], fill=color, width=4)
            label = f"{seat['seat_id']} · {status} ({confidence:.2f})"
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
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    render_demo(args.video, start=args.start, duration=args.duration, fps=args.fps, output_path=args.output)
