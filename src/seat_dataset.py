"""Dataset de recortes ROI por asiento para entrenar/evaluar un clasificador
de estado (libre/ocupado/reservado), a partir de data/ground_truth/labels.json.
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


def load_entries(labels_path):
    """Carga etiquetas desde un archivo .json, o desde TODOS los .json de un
    directorio (una sesión de grabación por archivo, ver README: "cómo agregar
    más sesiones"). Permite sumar datos nuevos con solo dejar un archivo más
    en data/ground_truth/, sin tocar código."""
    if os.path.isdir(labels_path):
        entries = []
        for path in sorted(glob.glob(os.path.join(labels_path, "*.json"))):
            entries.extend(json.load(open(path, encoding="utf-8")))
        return entries
    return json.load(open(labels_path, encoding="utf-8"))


def chronological_split(entries, val_fraction=0.2):
    """Separa por frame_file (orden temporal), no al azar: si se mezclara al
    azar, los 4 recortes de un mismo frame (altamente correlacionados entre
    sí) podrían quedar repartidos entre train y val, inflando la métrica.

    Deja como validación el último tramo cronológico del video (bloque
    contiguo) — la prueba más estricta: ¿generaliza hacia adelante en el
    tiempo, sin haber visto nada de ese tramo?"""
    frame_files = sorted(set(e["frame_file"] for e in entries))
    n_val = max(1, int(len(frame_files) * val_fraction))
    val_frames = set(frame_files[-n_val:])
    train = [e for e in entries if e["frame_file"] not in val_frames]
    val = [e for e in entries if e["frame_file"] in val_frames]
    return train, val


def interleaved_split(entries, val_fraction=0.2):
    """Igual que chronological_split (separa por frame_file, no al azar), pero
    tomando 1 de cada N frames repartidos a lo largo de todo el video en vez
    de un bloque final contiguo. Sirve para chequear que un accuracy alto no
    sea solo porque el tramo final es menos diverso (misma iluminación,
    mismas personas) que el resto de la sesión."""
    frame_files = sorted(set(e["frame_file"] for e in entries))
    step = max(2, round(1 / val_fraction))
    val_frames = set(frame_files[i] for i in range(0, len(frame_files), step))
    train = [e for e in entries if e["frame_file"] not in val_frames]
    val = [e for e in entries if e["frame_file"] in val_frames]
    return train, val


def full_split(entries, val_fraction=0.0):
    """Sin validación: todo el dataset a entrenamiento. Para el modelo de
    producción final, una vez que el enfoque ya fue validado con
    chronological_split/interleaved_split — usar toda la data disponible."""
    return list(entries), []


class SeatCropDataset(Dataset):
    def __init__(self, entries, frames_dir, bboxes, transform=None):
        self.entries = entries
        self.frames_dir = frames_dir
        self.bboxes = bboxes
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = Image.open(os.path.join(self.frames_dir, entry["frame_file"])).convert("RGB")
        crop = image.crop(self.bboxes[entry["seat_id"]])
        if self.transform:
            crop = self.transform(crop)
        return crop, STATUS_TO_IDX[entry["status"]]
