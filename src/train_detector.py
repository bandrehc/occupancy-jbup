"""Canal 1 del serial dual-channel: clasificador binario (persona / sin-persona)
sobre recortes ROI de asiento.

En el pipeline serial:
  - si Canal 1 detecta persona  → estado = "ocupado"  (no pasa a Canal 2)
  - si Canal 1 no detecta persona → pasa a Canal 2 (clasifica reservado/libre)

Las etiquetas se derivan automáticamente del campo 'status' existente:
  "ocupado"             → 1 (persona presente)
  "libre" / "reservado" → 0 (sin persona)

No se necesitan anotaciones adicionales: se reutiliza el mismo labels_videoN.json
que usa Canal 2. El mismo split camera_holdout permite medir generalización
cross-cámara usando video9 como cámara no vista durante entrenamiento.

Arquitectura: ResNet18 fine-tuning parcial (layer4 + fc), igual que Canal 2.
La salida es binaria (2 clases) en vez de 3. Se puede sustituir por Faster RCNN
cuando se dispongan de anotaciones de bounding box por asiento.

Uso:
    python src/train_detector.py
    python src/train_detector.py --split camera_holdout --holdout-video video9
    python src/train_detector.py --epochs 20 --lr 5e-4
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from PIL import Image

from config import load_config
from evaluate import evaluate_pairs
from seat_dataset import (
    SeatCropDataset, camera_holdout_split, chronological_split,
    full_split, interleaved_split, load_entries, seat_bboxes,
)
from train_classifier import _load_multi_video_config

PERSON_STATUSES = ["sin_persona", "persona"]
PERSON_TO_IDX = {"libre": 0, "reservado": 0, "ocupado": 1}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

SPLITTERS = {
    "tail":           chronological_split,
    "interleaved":    interleaved_split,
    "full":           full_split,
    "camera_holdout": camera_holdout_split,
}


class PersonCropDataset(Dataset):
    """Igual que SeatCropDataset pero con etiqueta binaria (persona/no-persona)."""

    def __init__(self, entries, frames_dir, bboxes, transform=None,
                 frames_dirs=None, seats_map=None):
        self._inner = SeatCropDataset(
            entries, frames_dir, bboxes, transform,
            frames_dirs=frames_dirs, seats_map=seats_map,
        )
        self.entries = entries

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        crop, _ = self._inner[idx]
        label = PERSON_TO_IDX[self.entries[idx]["status"]]
        return crop, label


def build_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def train(labels_path="data/ground_truth", frames_dir="data/frames",
          config_path="config/pipeline_config.yaml", epochs=20, batch_size=16, lr=1e-3,
          split="tail", holdout_video=None,
          model_out="models/weights/person_detector.pt",
          report_out="output/reports/evaluation_detector.md"):
    """Entrena Canal 1: clasificador binario persona/no-persona sobre crops de asiento."""
    cfg = load_config(config_path)
    seats_map, frames_dirs = _load_multi_video_config(cfg, frames_dir)

    if holdout_video is None:
        holdout_video = cfg.get("camera_holdout_video", "video9cut2")

    entries = load_entries(labels_path)

    if split == "camera_holdout":
        train_entries, val_entries = camera_holdout_split(entries, holdout_video_id=holdout_video)
    else:
        train_entries, val_entries = SPLITTERS[split](entries, val_fraction=0.2)

    n_person    = sum(1 for e in train_entries if e["status"] == "ocupado")
    n_no_person = len(train_entries) - n_person
    n_videos_train = len(set(e.get("video_id", "video1") for e in train_entries))
    n_videos_val   = len(set(e.get("video_id", "video1") for e in val_entries))
    print(
        f"train={len(train_entries)} ({n_videos_train} videos) "
        f"[persona={n_person}, sin_persona={n_no_person}]  "
        f"val={len(val_entries)} ({n_videos_val} videos)  "
        f"split='{split}'",
        flush=True,
    )

    legacy_bboxes = next(iter(seats_map.values())) if seats_map else {}

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.RandomRotation(5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = PersonCropDataset(
        train_entries, frames_dir, legacy_bboxes, train_tf,
        frames_dirs=frames_dirs, seats_map=seats_map,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    has_val = len(val_entries) > 0
    val_ds = PersonCropDataset(
        val_entries, frames_dir, legacy_bboxes, val_tf,
        frames_dirs=frames_dirs, seats_map=seats_map,
    ) if has_val else None
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if has_val else None

    # Peso inverso de frecuencia para manejar desbalance persona/no-persona
    counts  = [max(1, n_no_person), max(1, n_person)]
    weights = torch.tensor([len(train_entries) / (2 * c) for c in counts], dtype=torch.float32)
    print("pesos [sin_persona, persona]:", [round(w, 2) for w in weights.tolist()], flush=True)

    model     = build_model()
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_acc = -1.0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        train_loss = total_loss / len(train_ds)

        if has_val:
            model.eval()
            correct = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    correct += (model(images).argmax(dim=1) == labels).sum().item()
            val_acc = correct / len(val_ds)
            print(f"epoch {epoch:02d}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}", flush=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            print(f"epoch {epoch:02d}/{epochs}  train_loss={train_loss:.4f}", flush=True)

    if has_val:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    torch.save(model.state_dict(), model_out)

    if not has_val:
        print(f"Modelo Canal 1 guardado en {model_out}", flush=True)
        return model_out, None

    print(f"Mejor val_acc={best_val_acc:.3f} — modelo Canal 1 guardado en {model_out}", flush=True)

    model.eval()
    pairs = []
    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images).argmax(dim=1)
            for y, p in zip(labels.tolist(), preds.tolist()):
                pairs.append((PERSON_STATUSES[y], PERSON_STATUSES[p]))

    result = evaluate_pairs(pairs, report_out, title="Evaluación Canal 1 — detector de persona (ResNet18 binario)")
    print(f"Accuracy (val): {result['accuracy']:.3f}", flush=True)
    print(f"Reporte guardado en {result['report_path']}", flush=True)
    return model_out, report_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--split",        choices=["tail", "interleaved", "full", "camera_holdout"], default="tail")
    parser.add_argument("--holdout-video", default="video9cut2")
    parser.add_argument("--labels",       default="data/ground_truth")
    parser.add_argument("--model-out",    default="models/weights/person_detector.pt")
    parser.add_argument("--report-out",   default="output/reports/evaluation_detector.md")
    args = parser.parse_args()
    train(
        labels_path=args.labels, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, split=args.split, holdout_video=args.holdout_video,
        model_out=args.model_out, report_out=args.report_out,
    )
