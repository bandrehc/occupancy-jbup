import argparse
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
                    trim_start=None, trim_end=None, target_frames=None,
                    config_path="config/pipeline_config.yaml"):
    """Extrae frames muestreados del video con ffmpeg.

    Parámetros opcionales (se leen de config si no se pasan):
      fps           — frames por segundo a muestrear; si es 0 o None se calcula
                      automáticamente a partir de target_frames.
      target_frames — número deseado de frames por video; sobrescribe fps cuando
                      fps no está fijado. Útil para videos de distintas duraciones.
      width         — ancho de salida en píxeles (alto se calcula proporcionalmente).
      trim_start/end— segundos a excluir al inicio y al final.

    Retorna la ruta al manifest.csv con columnas frame_file,timestamp_seconds,timestamp.
    """
    cfg = load_config(config_path)
    s = cfg["sampling"]

    if width is None:
        width = s["frame_width"]
    if trim_start is None:
        trim_start = s.get("trim_start_seconds", 0)
    if trim_end is None:
        trim_end = s.get("trim_end_seconds", 0)

    total_duration = get_video_duration(video_path)
    usable = total_duration - trim_start - trim_end

    if fps is None or fps == 0:
        n = target_frames if target_frames is not None else s.get("target_frames")
        cfg_fps = s.get("fps", 0)
        if n and n > 0:
            fps = n / usable
        elif cfg_fps and cfg_fps > 0:
            fps = cfg_fps
        else:
            raise ValueError("Especifica fps o target_frames en el config o como argumento.")

    if max_duration is None:
        max_duration = usable

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

    print(f"  {len(frame_files)} frames → {output_dir}  (fps={fps:.4f}, usable={usable:.1f}s)")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrae frames muestreados de un video.")
    parser.add_argument("--video", required=True, help="Ruta al video (data/processed/videoN.mp4)")
    parser.add_argument("--output", required=True, help="Directorio de salida para los frames")
    parser.add_argument("--target-frames", type=int, default=None,
                        help="Número de frames a extraer (calcula fps automáticamente). "
                             "Sobrescribe el fps del config cuando se especifica.")
    parser.add_argument("--fps", type=float, default=None,
                        help="FPS de muestreo explícito (opcional).")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    manifest = extract_frames(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        target_frames=args.target_frames,
        config_path=args.config,
    )
    print(f"Manifest: {manifest}")
