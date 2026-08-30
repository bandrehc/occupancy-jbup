"""Evalúa las predicciones de occupancy_detect.py contra el ground truth manual.

Uso:
    python src/evaluate.py
    python src/evaluate.py --predictions output/reports/predictions.json --ground-truth data/ground_truth/labels.json
"""

import argparse
import json
import os
from collections import Counter, defaultdict

STATUSES = ["libre", "ocupado", "reservado"]


def _load_pairs(pred_path, gt_path):
    preds = {
        (p["frame_file"], p["seat_id"]): p["status"]
        for p in json.load(open(pred_path, encoding="utf-8"))
    }
    gts = json.load(open(gt_path, encoding="utf-8"))

    pairs = []
    unmatched = 0
    for g in gts:
        key = (g["frame_file"], g["seat_id"])
        if key in preds:
            pairs.append((g["status"], preds[key]))
        else:
            unmatched += 1
    return pairs, unmatched


def evaluate_pairs(pairs, output_path, unmatched=0, title="Evaluación de ocupación"):
    """Calcula accuracy/precision/recall/f1/matriz de confusión para una lista
    de pares (status_real, status_predicho) y escribe un reporte en Markdown.
    Reusado tanto por la evaluación de YOLO como por la del clasificador, para
    que los reportes sean comparables 1:1."""
    n = len(pairs)
    correct = sum(1 for y, p in pairs if y == p)
    accuracy = correct / n if n else 0.0

    confusion = defaultdict(Counter)
    for y, p in pairs:
        confusion[y][p] += 1

    per_class = {}
    for cls in STATUSES:
        tp = confusion[cls][cls]
        fn = sum(v for k, v in confusion[cls].items() if k != cls)
        fp = sum(confusion[other][cls] for other in STATUSES if other != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    lines = [
        f"# {title}",
        "",
        f"Pares evaluados: {n} (sin predicción: {unmatched})",
        f"Accuracy global: {accuracy:.3f}",
        "",
        "## Por clase",
        "",
        "| clase | precision | recall | f1 | support |",
        "|---|---|---|---|---|",
    ]
    for cls in STATUSES:
        m = per_class[cls]
        lines.append(f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")

    lines += [
        "",
        "## Matriz de confusión (filas=real, columnas=predicho)",
        "",
        "| real \\ pred | " + " | ".join(STATUSES) + " |",
        "|" + "---|" * (len(STATUSES) + 1),
    ]
    for real in STATUSES:
        row = [str(confusion[real][pred]) for pred in STATUSES]
        lines.append(f"| {real} | " + " | ".join(row) + " |")

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "report_path": output_path,
    }


def evaluate(pred_path="output/reports/predictions.json",
             gt_path="data/ground_truth/labels.json",
             output_path="output/reports/evaluation.md"):
    pairs, unmatched = _load_pairs(pred_path, gt_path)
    return evaluate_pairs(pairs, output_path, unmatched=unmatched)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="output/reports/predictions.json")
    parser.add_argument("--ground-truth", default="data/ground_truth/labels.json")
    parser.add_argument("--output", default="output/reports/evaluation.md")
    args = parser.parse_args()

    result = evaluate(args.predictions, args.ground_truth, args.output)
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(f"Reporte guardado en {result['report_path']}")
