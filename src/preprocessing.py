import csv
import os
import subprocess

import cv2

from config import load_config


def get_video_duration(video_path):
    """Duración del video en segundos vía ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_frames(video_path, output_dir, fps=None, width=None, max_duration=None,
                    trim_start=None, trim_end=None, config_path="config/pipeline_config.yaml"):
    """Extrae frames muestreados del video con ffmpeg.

    fps, width, trim_start y trim_end son opcionales: si no se pasan, se leen de
    config/pipeline_config.yaml.

    trim_start / trim_end (segundos) excluyen los extremos del video donde se
    manipula la cámara (colocación al inicio, retiro al final) para que no
    generen frames espurios.
    max_duration (segundos) limita cuánto se procesa después de trim_start; si no
    se pasa, se calcula a partir de la duración total del video y trim_end.

    Retorna la ruta al manifest.csv con columnas frame_file,timestamp_seconds,timestamp.
    Los timestamps ya incluyen el offset de trim_start (tiempo real dentro del video original).
    """
    cfg = None
    if fps is None or width is None or trim_start is None or trim_end is None:
        cfg = load_config(config_path)
        fps = fps if fps is not None else cfg["sampling"]["fps"]
        width = width if width is not None else cfg["sampling"]["frame_width"]
        trim_start = trim_start if trim_start is not None else cfg["sampling"].get("trim_start_seconds", 0)
        trim_end = trim_end if trim_end is not None else cfg["sampling"].get("trim_end_seconds", 0)

    if max_duration is None and (trim_start or trim_end):
        total_duration = get_video_duration(video_path)
        max_duration = total_duration - trim_start - trim_end

    os.makedirs(output_dir, exist_ok=True)

    cmd = ["ffmpeg", "-y"]
    if trim_start:
        cmd += ["-ss", str(trim_start)]
    cmd += ["-i", video_path]
    if max_duration is not None:
        cmd += ["-t", str(max_duration)]
    cmd += [
        "-vf", f"fps={fps},scale={width}:-2",
        "-qscale:v", "2",
        os.path.join(output_dir, "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    frame_files = sorted(f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".jpg"))

    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_file", "timestamp_seconds", "timestamp"])
        for i, frame_file in enumerate(frame_files):
            ts_seconds = trim_start + i / fps
            writer.writerow([frame_file, f"{ts_seconds:.3f}", _format_timestamp(ts_seconds)])

    return manifest_path


def normalize_illumination(frame):
    """Ecualiza el canal V (HSV) para compensar sub/sobreexposición."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)


def _format_timestamp(seconds):
    total_seconds = int(seconds)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
