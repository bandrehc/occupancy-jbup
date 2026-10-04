"""Dataset de recortes ROI por asiento para entrenar/evaluar un clasificador
de estado (libre/ocupado/reservado).

Soporta datasets multi-video: cada entry tiene un campo opcional 'video_id'
que determina qué directorio de frames y qué calibración de asientos usar.
Los entries sin 'video_id' se tratan como pertenecientes a "video1" (legado).

Convención de archivos de etiquetas:
  labels.json           → video_id = "video1"  (legado)
  labels_video2.json    → video_id = "video2"
  labels_videoN.json    → video_id = "videoN"
"""

import glob
import json
import os

from PIL import Image
from torch.utils.data import Dataset

STATUSES = ["libre", "ocupado", "reservado"]
STATUS_TO_IDX = {s: i for i, s in enumerate(STATUSES)}


def seat_bboxes(seats_path, pad=10, image_size=(1280, 720)):
    seats = json.load(open(seats_path, encoding="utf-8"))
    width, height = image_size
    bboxes = {}
    for seat in seats:
        xs = [p[0] for p in seat["polygon"]]
        ys = [p[1] for p in seat["polygon"]]
        x0 = max(0, min(xs) - pad)
        y0 = max(0, min(ys) - pad)
        x1 = min(width, max(xs) + pad)
        y1 = min(height, max(ys) + pad)
        bboxes[seat["seat_id"]] = (x0, y0, x1, y1)
    return bboxes


def _video_id_from_filename(filename):
    """Infiere video_id a partir del nombre del archivo de etiquetas.

    labels.json         → "video1"
    labels_video2.json  → "video2"
    otros               → stem del archivo
    """
    stem = os.path.splitext(filename)[0]
    if stem == "labels":
        return "video1"
    if stem.startswith("labels_"):
        return stem[len("labels_"):]
    return stem


def load_entries(labels_path):
    """Carga etiquetas e inyecta 'video_id' desde el nombre del archivo.

    Acepta un .json individual o un directorio (lee todos los .json del dir).
    Al cargar desde directorio, el video_id se deriva del nombre de cada
    archivo — permite agregar una sesión nueva soltando labels_videoN.json
    en data/ground_truth/ sin tocar código.
    """
    if os.path.isdir(labels_path):
        entries = []
        for path in sorted(glob.glob(os.path.join(labels_path, "*.json"))):
            video_id = _video_id_from_filename(os.path.basename(path))
            for e in json.load(open(path, encoding="utf-8")):
                entries.append(e if "video_id" in e else dict(e, video_id=video_id))
        return entries
    video_id = _video_id_from_filename(os.path.basename(labels_path))
    return [
        e if "video_id" in e else dict(e, video_id=video_id)
        for e in json.load(open(labels_path, encoding="utf-8"))
    ]


def chronological_split(entries, val_fraction=0.2):
    """Val = último tramo cronológico (bloque contiguo) de CADA video.

    Separa por (video_id, frame_file) para que todos los recortes de un
    mismo frame queden en el mismo split (evita data leakage entre crops
    correlacionados del mismo instante)."""
    by_video = {}
    for e in entries:
        vid = e.get("video_id", "video1")
        by_video.setdefault(vid, set()).add(e["frame_file"])

    val_frames = set()
    for vid, frames in by_video.items():
        sorted_frames = sorted(frames)
        n_val = max(1, int(len(sorted_frames) * val_fraction))
        for f in sorted_frames[-n_val:]:
            val_frames.add((vid, f))

    train = [e for e in entries if (e.get("video_id", "video1"), e["frame_file"]) not in val_frames]
    val   = [e for e in entries if (e.get("video_id", "video1"), e["frame_file"]) in val_frames]
    return train, val


def interleaved_split(entries, val_fraction=0.2):
    """Val = 1 de cada N frames, distribuidos a lo largo de CADA video."""
    by_video = {}
    for e in entries:
        vid = e.get("video_id", "video1")
        by_video.setdefault(vid, set()).add(e["frame_file"])

    val_frames = set()
    step = max(2, round(1 / val_fraction))
    for vid, frames in by_video.items():
        for f in list(sorted(frames))[::step]:
            val_frames.add((vid, f))

    train = [e for e in entries if (e.get("video_id", "video1"), e["frame_file"]) not in val_frames]
    val   = [e for e in entries if (e.get("video_id", "video1"), e["frame_file"]) in val_frames]
    return train, val


def full_split(entries, val_fraction=0.0):
    """Sin validación: todo el dataset a entrenamiento (modelo de producción)."""
    return list(entries), []


def camera_holdout_split(entries, holdout_video_id="video9cut2"):
    """Val = todos los frames del video holdout (cámara no vista en train).

    Es el split más exigente para medir generalización cross-cámara: el modelo
    nunca vio ese ángulo durante el entrenamiento.
    """
    train = [e for e in entries if e.get("video_id", "video1") != holdout_video_id]
    val   = [e for e in entries if e.get("video_id", "video1") == holdout_video_id]
    return train, val


class SeatCropDataset(Dataset):
    """Dataset de recortes ROI de asientos.

    Soporta dos modos:
    - Single-video (legado): pasar frames_dir (str) y bboxes (dict seat_id→bbox).
    - Multi-video: pasar además frames_dirs ({video_id: dir}) y seats_map
      ({video_id: bboxes_dict}). Si el video_id de un entry no está en los
      dicts, se usa el fallback frames_dir / bboxes.
    """

    def __init__(self, entries, frames_dir, bboxes, transform=None,
                 frames_dirs=None, seats_map=None):
        self.entries = entries
        self.frames_dir = frames_dir
        self.bboxes = bboxes
        self.transform = transform
        self.frames_dirs = frames_dirs or {}
        self.seats_map = seats_map or {}

    def _resolve(self, entry):
        vid = entry.get("video_id", "video1")
        fdir = self.frames_dirs.get(vid, self.frames_dir)
        bb   = self.seats_map.get(vid, self.bboxes)
        return fdir, bb

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        fdir, bb = self._resolve(entry)
        path = os.path.join(fdir, entry["frame_file"])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Frame no encontrado: {path}  "
                f"(video_id={entry.get('video_id','video1')}, "
                f"frame={entry['frame_file']})"
            )
        image = Image.open(path).convert("RGB")
        crop = image.crop(bb[entry["seat_id"]])
        if self.transform:
            crop = self.transform(crop)
        return crop, STATUS_TO_IDX[entry["status"]]
