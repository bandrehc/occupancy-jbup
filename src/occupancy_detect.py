"""Detección automática de ocupación por asiento usando YOLO.

Para cada frame de data/frames/, corre detección de personas (y de objetos
personales: mochila, laptop, cartera, maleta) y cruza cada detección contra
los polígonos de data/calibration/seats.json para decidir el estado de cada
asiento: ocupado / libre / reservado.

Usa el esquema de refinamiento de pipeline_config.yaml: una detección de
persona con confianza dentro del margen (confidence_threshold - confidence_margin
hasta confidence_threshold) se considera dudosa y se re-verifica corriendo YOLO
otra vez solo sobre el recorte de esa región del asiento.

Uso:
    python src/occupancy_detect.py
    python src/occupancy_detect.py --output output/reports/predictions.json
"""

import argparse
import csv
import json
import os

from PIL import Image
from ultralytics import YOLO

from config import load_config

PERSON_CLASS = "person"
OBJECT_CLASSES = ["backpack", "laptop", "handbag", "suitcase"]


def _seat_mask(polygon, width, height):
    from PIL import ImageDraw
    mask_img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask_img).polygon([tuple(p) for p in polygon], fill=1)
    import numpy as np
    return np.array(mask_img, dtype=bool)


def _load_seats(seats_path, width, height):
    seats = json.load(open(seats_path, encoding="utf-8"))
    for seat in seats:
        seat["mask"] = _seat_mask(seat["polygon"], width, height)
        xs = [p[0] for p in seat["polygon"]]
        ys = [p[1] for p in seat["polygon"]]
        seat["bbox"] = (min(xs), min(ys), max(xs), max(ys))
    return seats


def _match_fraction(mask, box):
    """Qué tan 'adentro' del asiento está una detección.

    Se toma el máximo entre la fracción del ÁREA DEL ASIENTO cubierta por la
    caja (funciona bien para personas, que suelen ser más grandes que el
    asiento) y la fracción de la CAJA DE LA DETECCIÓN contenida en el asiento
    (necesario para objetos chicos como una mochila, que nunca van a cubrir
    una porción relevante del polígono completo aunque estén 100% dentro).
    """
    h, w = mask.shape
    x0 = max(0, min(w, int(round(box[0]))))
    y0 = max(0, min(h, int(round(box[1]))))
    x1 = max(0, min(w, int(round(box[2]))))
    y1 = max(0, min(h, int(round(box[3]))))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = mask[y0:y1, x0:x1].sum()
    seat_area = mask.sum()
    box_area = (x1 - x0) * (y1 - y0)
    frac_of_seat = intersection / seat_area if seat_area else 0.0
    frac_of_box = intersection / box_area if box_area else 0.0
    return max(frac_of_seat, frac_of_box)


def _best_match(dets, seat, min_overlap):
    """Detección de mayor confianza que solapa el asiento por encima de min_overlap.

    Retorna (confianza, caja) o (0.0, None) si ninguna califica.
    """
    best_conf, best_box = 0.0, None
    for conf, box in dets:
        if conf > best_conf and _match_fraction(seat["mask"], box) >= min_overlap:
            best_conf, best_box = conf, box
    return best_conf, best_box


def _classify_seat(seat, person_dets, object_dets, model, image, person_id,
                    min_overlap, conf_threshold, conf_margin, object_conf_threshold, low_conf):
    person_conf, person_box = _best_match(person_dets, seat, min_overlap)
    refined = False

    if conf_threshold - conf_margin <= person_conf < conf_threshold:
        # Recorte dudoso: se une la caja de la detección con la del asiento para
        # no cercenar cabeza/hombros (el polígono del asiento no llega tan arriba).
        sx0, sy0, sx1, sy1 = seat["bbox"]
        bx0, by0, bx1, by1 = person_box
        x0, y0 = min(sx0, bx0), min(sy0, by0)
        x1, y1 = max(sx1, bx1), max(sy1, by1)
        pad = 20
        crop_box = (
            max(0, x0 - pad), max(0, y0 - pad),
            min(image.width, x1 + pad), min(image.height, y1 + pad),
        )
        crop = image.crop(crop_box)
        refined_results = model.predict(source=crop, classes=[person_id], conf=low_conf, verbose=False)[0]
        refined_conf = max((float(b.conf.item()) for b in refined_results.boxes), default=0.0)
        if refined_conf > person_conf:
            person_conf = refined_conf
            refined = True

    if person_conf >= conf_threshold:
        return "ocupado", person_conf, refined

    object_conf, _ = _best_match(object_dets, seat, min_overlap)
    if object_conf >= object_conf_threshold:
        return "reservado", object_conf, False

    return "libre", max(person_conf, object_conf), False


def detect_occupancy(frames_dir="data/frames", manifest_path=None,
                      config_path="config/pipeline_config.yaml",
                      output_path="output/reports/predictions.json"):
    cfg = load_config(config_path)
    seats_path = cfg["seats_config"]
    conf_threshold = cfg["detection"]["confidence_threshold"]
    object_conf_threshold = cfg["detection"].get("object_confidence_threshold", 0.25)
    weights_path = cfg["detection"].get("weights", "models/weights/yolov8n.pt")
    conf_margin = cfg["refinement"]["confidence_margin"]
    min_overlap = cfg.get("matching", {}).get("min_seat_overlap", 0.25)

    manifest_path = manifest_path or os.path.join(frames_dir, "manifest.csv")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    model = YOLO(weights_path)
    name_to_id = {v: k for k, v in model.names.items()}
    person_id = name_to_id[PERSON_CLASS]
    object_ids = [name_to_id[c] for c in OBJECT_CLASSES if c in name_to_id]
    classes_of_interest = [person_id] + object_ids
    # low_conf es el piso que se le pide al modelo al recuperar detecciones crudas:
    # debe cubrir tanto la banda dudosa de personas como el umbral (más bajo) de objetos.
    low_conf = max(0.01, min(conf_threshold - conf_margin, object_conf_threshold) - 0.05)

    frame_paths = [os.path.join(frames_dir, row["frame_file"]) for row in manifest]

    # Fase 1: detección principal en un solo batch (sin stream=True: se completa
    # por entero antes de seguir, para no dejar un generador a medio consumir).
    print(f"Detección principal sobre {len(frame_paths)} frames...", flush=True)
    all_results = model.predict(
        source=frame_paths, classes=classes_of_interest, conf=low_conf, verbose=False,
    )

    seats = None
    per_frame_dets = []
    for results in all_results:
        h, w = results.orig_shape
        if seats is None:
            seats = _load_seats(seats_path, w, h)
        person_dets, object_dets = [], []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            (person_dets if cls_id == person_id else object_dets).append((conf, xyxy))
        per_frame_dets.append((person_dets, object_dets))
    del all_results  # libera las imágenes originales de memoria

    # Fase 2: clasificación por asiento; solo reabre la imagen y llama al modelo
    # de nuevo (recorte) cuando un asiento cae en la banda dudosa de refinamiento.
    print("Clasificando asientos (con refinamiento donde aplica)...", flush=True)
    predictions = []
    for i, (row, (person_dets, object_dets)) in enumerate(zip(manifest, per_frame_dets)):
        needs_image = any(
            conf_threshold - conf_margin <= _best_match(person_dets, seat, min_overlap)[0] < conf_threshold
            for seat in seats
        )
        image = None
        if needs_image:
            frame_path = os.path.join(frames_dir, row["frame_file"])
            image = Image.open(frame_path).convert("RGB")

        for seat in seats:
            status, confidence, refined = _classify_seat(
                seat, person_dets, object_dets, model, image, person_id,
                min_overlap, conf_threshold, conf_margin, object_conf_threshold, low_conf,
            )
            predictions.append({
                "frame_file": row["frame_file"],
                "timestamp_seconds": float(row["timestamp_seconds"]),
                "seat_id": seat["seat_id"],
                "status": status,
                "confidence": round(confidence, 4),
                "refined": refined,
            })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(manifest)} frames clasificados", flush=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="data/frames")
    parser.add_argument("--output", default="output/reports/predictions.json")
    args = parser.parse_args()

    out = detect_occupancy(frames_dir=args.frames_dir, output_path=args.output)
    print(f"Predicciones guardadas en {out}")
