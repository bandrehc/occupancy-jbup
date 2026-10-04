"""Entrena un clasificador liviano (ResNet18 preentrenado, fine-tuning parcial)
de estado de asiento (libre/ocupado/reservado) sobre los recortes ROI de
data/ground_truth/labels.json — alternativa a la heurística YOLO+solape de
occupancy_detect.py cuando lo único que importa es la precisión de la
clasificación de estado sobre esta cámara fija (no detección de objetos ni
generalización a otra escena).

Solo se re-entrena la capa 4 y la cabeza (fc) de ResNet18: con ~1000 ejemplos
de entrenamiento, ajustar la red completa arriesga sobreajuste severo.

Uso:
    python src/train_classifier.py
    python src/train_classifier.py --epochs 30 --lr 5e-4
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from config import load_config
from evaluate import evaluate_pairs
from seat_dataset import (
    STATUSES, SeatCropDataset, camera_holdout_split, chronological_split,
    full_split, interleaved_split, load_entries, seat_bboxes,
)

SPLITTERS = {
    "tail":           chronological_split,
    "interleaved":    interleaved_split,
    "full":           full_split,
    "camera_holdout": camera_holdout_split,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(STATUSES))
    return model


def _load_multi_video_config(cfg, legacy_frames_dir):
    """Construye seats_map y frames_dirs desde cfg['videos'].

    Si cfg no tiene sección 'videos', usa el valor legado seats_config +
    legacy_frames_dir mapeado a video_id="video1".
    """
    videos_cfg = cfg.get("videos", {})
    if videos_cfg:
        seats_map   = {vid: seat_bboxes(info["seats_config"]) for vid, info in videos_cfg.items()}
        frames_dirs = {vid: info.get("frames_dir", f"data/frames/{vid}") for vid, info in videos_cfg.items()}
    else:
        seats_map   = {"video1": seat_bboxes(cfg["seats_config"])}
        frames_dirs = {"video1": legacy_frames_dir}
    return seats_map, frames_dirs


def train(labels_path="data/ground_truth", frames_dir="data/frames",
          config_path="config/pipeline_config.yaml", epochs=25, batch_size=16, lr=1e-3,
          split="tail", holdout_video=None,
          model_out="models/weights/seat_classifier.pt",
          report_out="output/reports/evaluation_classifier.md"):
    """Canal 2 del serial dual-channel: clasifica asientos sin persona como
    'reservado' (objetos presentes) o 'libre' (vacío).

    labels_path puede ser un .json puntual o el directorio data/ground_truth/
    (carga todos los .json del dir, cada uno con su video_id derivado del nombre).

    split='camera_holdout' reserva holdout_video como cámara de validación
    cross-ángulo (prueba de generalización real).
    """
    cfg = load_config(config_path)
    seats_map, frames_dirs = _load_multi_video_config(cfg, frames_dir)

    if holdout_video is None:
        holdout_video = cfg.get("camera_holdout_video", "video9cut2")

    entries = load_entries(labels_path)

    if split == "camera_holdout":
        train_entries, val_entries = camera_holdout_split(entries, holdout_video_id=holdout_video)
    else:
        train_entries, val_entries = SPLITTERS[split](entries, val_fraction=0.2)

    n_videos_train = len(set(e.get("video_id", "video1") for e in train_entries))
    n_videos_val   = len(set(e.get("video_id", "video1") for e in val_entries))
    print(
        f"train={len(train_entries)} ({n_videos_train} videos)  "
        f"val={len(val_entries)} ({n_videos_val} videos)  "
        f"split='{split}'",
        flush=True,
    )

    # Fallback legacy: si entries solo tienen "video1", frames_dir y primer bboxes
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

    train_ds = SeatCropDataset(
        train_entries, frames_dir, legacy_bboxes, train_tf,
        frames_dirs=frames_dirs, seats_map=seats_map,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    has_val = len(val_entries) > 0
    val_ds = SeatCropDataset(
        val_entries, frames_dir, legacy_bboxes, val_tf,
        frames_dirs=frames_dirs, seats_map=seats_map,
    ) if has_val else None
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if has_val else None

    counts = [max(1, sum(1 for e in train_entries if e["status"] == s)) for s in STATUSES]
    weights = torch.tensor([len(train_entries) / (len(STATUSES) * c) for c in counts], dtype=torch.float32)
    print("pesos por clase (inverso de frecuencia):", dict(zip(STATUSES, [round(w, 2) for w in weights.tolist()])), flush=True)

    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        train_loss = total_loss / len(train_ds)

        if has_val:
            model.eval()
            correct = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    preds = model(images).argmax(dim=1)
                    correct += (preds == labels).sum().item()
            val_acc = correct / len(val_ds)
            print(f"epoch {epoch:02d}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}", flush=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            print(f"epoch {epoch:02d}/{epochs}  train_loss={train_loss:.4f}", flush=True)

    if has_val:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    torch.save(model.state_dict(), model_out)

    if not has_val:
        print(f"Modelo de producción (sin held-out, {len(train_entries)} ejemplos) guardado en {model_out}", flush=True)
        print("Accuracy esperado: ver evaluación cruzada previa (splits 'tail'/'interleaved') en output/reports/.", flush=True)
        return model_out, None

    print(f"Mejor val_acc={best_val_acc:.3f} — modelo guardado en {model_out}", flush=True)

    model.eval()
    pairs = []
    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images).argmax(dim=1)
            for y, p in zip(labels.tolist(), preds.tolist()):
                pairs.append((STATUSES[y], STATUSES[p]))

    result = evaluate_pairs(pairs, report_out, title="Evaluación del clasificador (ResNet18)")
    print(f"Accuracy (val): {result['accuracy']:.3f}", flush=True)
    print(f"Reporte guardado en {result['report_path']}", flush=True)

    return model_out, report_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--split", choices=["tail", "interleaved", "full", "camera_holdout"], default="tail")
    parser.add_argument("--holdout-video", default="video9cut2",
                        help="video_id reservado para validación cross-cámara (solo con --split camera_holdout).")
    parser.add_argument("--labels", default="data/ground_truth",
                         help="Archivo .json puntual o directorio con todas las sesiones (default).")
    parser.add_argument("--model-out", default="models/weights/seat_classifier.pt")
    parser.add_argument("--report-out", default="output/reports/evaluation_classifier.md")
    args = parser.parse_args()
    train(labels_path=args.labels, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          split=args.split, holdout_video=args.holdout_video,
          model_out=args.model_out, report_out=args.report_out)
