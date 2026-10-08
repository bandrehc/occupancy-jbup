# Detección de Ocupación de Asientos — Biblioteca JBUP

Sistema de detección automática del estado de asientos en una biblioteca
universitaria a partir de video de cámara fija. Clasifica cada asiento como
**libre**, **ocupado** o **reservado** (objetos presentes sin persona sentada)
mediante una arquitectura serial de dos canales basada en ResNet18.

Adaptado del enfoque propuesto en *A Serial Dual-Channel Library Occupancy
Detection System Based on Faster RCNN*.

## Arquitectura

Pipeline serial de dos canales donde el resultado del primero condiciona la
entrada al segundo:

```
Frame completo
  └─► recortar ROI de cada asiento (polígonos calibrados por cámara)
        └─► Canal 1 — ¿hay persona sentada?
              ├─ SÍ → estado = "ocupado"
              └─ NO → Canal 2 — ¿hay objetos en el asiento?
                        ├─ SÍ → estado = "reservado"
                        └─ NO → estado = "libre"
```

| Canal | Modelo | Salida | Script |
|-------|--------|--------|--------|
| Canal 1 | ResNet18 fine-tuned (binario) | persona / sin persona | `src/train_detector.py` |
| Canal 2 | ResNet18 fine-tuned (3 clases) | libre / ocupado / reservado | `src/train_classifier.py` |

Ambos canales comparten backbone (ResNet18 preentrenado en ImageNet) con
`layer4` + `fc` reentrenables y el resto congelado.

## Dataset

9 videos de cámara fija con diferentes posiciones y ángulos, 300 frames
extraídos por video, etiquetados manualmente asiento por asiento:

| Video | Entradas | Libre | Reservado | Ocupado |
|-------|----------|-------|-----------|---------|
| video1 | 1,196 | 521 | 198 | 477 |
| video2 | 1,500 | 705 | 19 | 776 |
| video3 | 1,200 | 0 | 149 | 1,051 |
| video4 | 1,200 | 459 | 386 | 355 |
| video5 | 1,200 | 0 | 600 | 600 |
| video6 | 1,435 | 549 | 74 | 812 |
| video7 | 328 | 187 | 2 | 139 |
| video9cut1 | 1,200 | 180 | 124 | 896 |
| video9cut2 | 1,200 | 0 | 126 | 1,074 |
| **Total** | **10,459** | **2,601** | **1,678** | **6,180** |

Criterio para **reservado**: asiento vacío con objetos personales visibles
(mochila, laptop, etc.) que indican que el puesto está tomado aunque nadie
esté sentado.

## Evaluación

Split **leave-one-camera-out**: se entrena con 8 videos y se evalúa sobre
`video4` (1,200 entradas con distribución balanceada de las 3 clases). Esto
mide generalización real a cámaras no vistas durante entrenamiento.

**Accuracy del clasificador (Canal 2): 99.9%** sobre el video holdout.

| Clase | Precisión | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| ocupado | 1.000 | 0.999 | 1.000 | 1,074 |
| reservado | 0.992 | 1.000 | 0.996 | 126 |

## Pipeline completo

```
1. Preprocesamiento      ffmpeg (recorte temporal, sin audio)
2. Extracción de frames  src/preprocessing.py (300 frames/video, auto-FPS)
3. Calibración ROI       data/calibration/seats_videoN.json (polígonos por asiento)
4. Anotación manual      data/ground_truth/labels_videoN.json (3 clases)
5. Entrenamiento          src/train_classifier.py + src/train_detector.py
6. Evaluación             métricas integradas con split camera_holdout
```

### Entrenamiento

```bash
# Canal 2 — clasificador triclase
python src/train_classifier.py --split camera_holdout --epochs 30 --lr 5e-4 --patience 5

# Canal 1 — detector binario de persona
python src/train_detector.py --split camera_holdout --epochs 20 --lr 1e-3 --patience 5
```

Hiperparámetros: batch size 16, CrossEntropyLoss con pesos inversos a la
frecuencia de clase, early stopping con patience=5. Augmentación:
`RandomHorizontalFlip`, `ColorJitter`, `RandomRotation(5)`,
`RandomPerspective`, `RandomAffine`.

### Agregar un video nuevo

1. Convertir: `ffmpeg -i video.mp4 -an -c:v copy data/processed/video_nuevo.mp4`
2. Calibrar polígonos de asientos → `data/calibration/seats_video_nuevo.json`
3. Agregar entrada en `config/pipeline_config.yaml`
4. Extraer frames: `python src/preprocessing.py --video data/processed/video_nuevo.mp4 --output data/frames/video_nuevo --target-frames 300`
5. (Opcional) Etiquetar y reentrenar para incorporar la nueva perspectiva

## Estructura del repositorio

```
config/pipeline_config.yaml       configuración central del pipeline
data/calibration/                  polígonos ROI por cámara (seats_videoN.json)
data/frames/                       frames extraídos por video
data/frames/first_frames/          primer frame de cada cámara (referencia visual)
data/ground_truth/                 anotaciones manuales (labels_videoN.json)
models/weights/                    pesos entrenados (.pt) — no versionados
output/reports/                    reportes de evaluación
src/config.py                      carga de pipeline_config.yaml
src/preprocessing.py               extracción de frames con auto-FPS
src/calibrate_seats.py             calibración interactiva de polígonos (OpenCV)
src/seat_dataset.py                dataset multi-video con splits de validación
src/train_classifier.py            Canal 2: clasificador triclase
src/train_detector.py              Canal 1: detector binario persona/sin-persona
src/evaluate.py                    métricas y reporte markdown
src/render_demo.py                 renderiza video con cajas + estado superpuestos
src/occupancy_detect.py            enfoque YOLO previo (referencia histórica)
docs/metodologia.md                documento de metodología completo
Antecedentes/                      paper de referencia
```

## Por qué un clasificador y no YOLO

El enfoque original usaba YOLOv8n para detectar personas y objetos, cruzando
detecciones contra los polígonos de asiento. Alcanzó 86.9% accuracy pero
**reservado** tenía un techo duro: YOLO detecta mochilas/laptops con confianza
muy baja (~0.25–0.35) en video de baja resolución, limitando el recall de esa
clase a 45.5%.

El clasificador aprende directamente de las etiquetas cualquier señal visual
relevante (no solo categorías COCO) y sube **reservado** a >99% recall. El
trade-off es que no generaliza a escenas completamente distintas sin
recalibración — aceptable para cámaras fijas de una biblioteca específica.

`src/occupancy_detect.py` y `src/evaluate.py` se mantienen como referencia.

## Setup

```bash
pip install -r requirements.txt
```

Requiere `ffmpeg`/`ffprobe` en el PATH para la extracción de frames.
