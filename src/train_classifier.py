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
    STATUSES, SeatCropDataset, chronological_split, full_split, interleaved_split, load_entries, seat_bboxes,
)

SPLITTERS = {"tail": chronological_split, "interleaved": interleaved_split, "full": full_split}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(STATUSES))
    return model


def train(labels_path="data/ground_truth", frames_dir="data/frames",
          config_path="config/pipeline_config.yaml", epochs=25, batch_size=16, lr=1e-3,
          split="tail",
          model_out="models/weights/seat_classifier.pt",
          report_out="output/reports/evaluation_classifier.md"):
    """labels_path puede ser un .json puntual o (por defecto) el directorio
    data/ground_truth/, que junta TODOS los .json que haya ahí — así sumar una
    sesión de grabación nueva es solo agregar un archivo, sin tocar código."""
    cfg = load_config(config_path)
    bboxes = seat_bboxes(cfg["seats_config"])

    entries = load_entries(labels_path)
    train_entries, val_entries = SPLITTERS[split](entries, val_fraction=0.2)
    print(f"train={len(train_entries)} val={len(val_entries)} (split='{split}', por frame, no al azar)", flush=True)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = SeatCropDataset(train_entries, frames_dir, bboxes, train_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    has_val = len(val_entries) > 0
    val_ds = SeatCropDataset(val_entries, frames_dir, bboxes, val_tf) if has_val else None
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
    parser.add_argument("--split", choices=["tail", "interleaved", "full"], default="tail")
    parser.add_argument("--labels", default="data/ground_truth",
                         help="Archivo .json puntual o directorio con todas las sesiones (default).")
    parser.add_argument("--model-out", default="models/weights/seat_classifier.pt")
    parser.add_argument("--report-out", default="output/reports/evaluation_classifier.md")
    args = parser.parse_args()
    train(labels_path=args.labels, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, split=args.split,
          model_out=args.model_out, report_out=args.report_out)
