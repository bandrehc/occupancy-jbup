# Metodología: Sistema de Detección de Ocupación de Asientos

**Proyecto:** Occupancy Detection — Biblioteca JBUP  
**Arquitectura:** Serial Dual-Channel (adaptado de *A Serial Dual-Channel Library Occupancy Detection System Based on Faster RCNN*)

---

## 1. Contexto y motivación

El objetivo del sistema es determinar automáticamente el estado de cada asiento visible en una grabación de cámara fija de la biblioteca: **libre** (vacío), **reservado** (objetos presentes pero sin persona) u **ocupado** (persona sentada).

El modelo anterior entrenado únicamente sobre `video1` alcanzó 99.2 % de precisión, pero no generalizaba a otros ángulos de cámara. La reformulación descrita en este documento extiende el enfoque a **9 videos con posiciones y ángulos de cámara distintos**, buscando un único modelo que generalice a perspectivas no vistas durante el entrenamiento.

---

## 2. Arquitectura del sistema

El sistema implementa un **pipeline serial de dos canales**, en el que el resultado del primero condiciona la entrada al segundo:

```
Frame completo
  └─► recortar ROI de cada asiento (según calibración de polígonos)
        └─► Canal 1 — ¿hay persona sentada?
              ├─ SÍ → estado = "ocupado"  (no pasa a Canal 2)
              └─ NO → Canal 2 — ¿hay objetos en el asiento?
                        ├─ SÍ → estado = "reservado"
                        └─ NO → estado = "libre"
```

### Canal 1 — Detector de persona (ResNet18 binario)

- **Arquitectura:** ResNet18 preentrenado en ImageNet, fine-tuning de `layer4` + cabeza `fc`
- **Salida:** 2 clases — `sin_persona` (0) / `persona` (1)
- **Etiquetas derivadas** de las anotaciones existentes: `"ocupado"` → 1, `"libre"` y `"reservado"` → 0
- **Script:** `src/train_detector.py`
- **Pesos:** `models/weights/person_detector.pt`

### Canal 2 — Clasificador de estado (ResNet18 triclase)

- **Arquitectura:** ResNet18 preentrenado en ImageNet, fine-tuning de `layer4` + cabeza `fc`
- **Salida:** 3 clases — `libre` / `ocupado` / `reservado`
- **Script:** `src/train_classifier.py`
- **Pesos:** `models/weights/seat_classifier.pt`

> **Nota de implementación actual:** Canal 2 opera sobre todas las entradas (no solo los asientos que Canal 1 descartó como ocupados). La integración serial completa se implementará en `occupancy_detect.py`.

---

## 3. Datos de origen

| Video | Duración útil | Asientos | Ángulo |
|-------|--------------|----------|--------|
| video1 | ~75 min | 4 | frontal lateral |
| video2 | ~3 min | 5 | — |
| video3 | ~24 min | 4 | — |
| video4 | ~7 min | 4 | — |
| video5 | ~7 min | 4 | — |
| video6 | ~26 min | 5 | — |
| video7 | ~65 min | 4 | — |
| video9cut1 | ~97 min | 4 | distinto de video9cut2 |
| video9cut2 | ~43 min | 4 | distinto de video9cut1 |

Los videos se almacenan en `data/processed/` (excluidos del repositorio por tamaño). Cada video mantiene **una perspectiva de cámara fija** durante toda su duración.

---

## 4. Fase 1 — Preprocesamiento de video

Cada video se convirtió a formato sin audio apto para procesamiento:

```bash
ffmpeg -i data/raw/videoN.mp4 -an -c:v copy data/processed/videoN_cut.mp4
```

Se aplicó recorte temporal para eliminar el inicio (colocación de cámara) y el final (retiro de cámara) usando `data/processed/` como salida.

---

## 5. Fase 2 — Extracción de frames

`src/preprocessing.py` extrae frames uniformemente distribuidos en el tiempo, calculando el FPS necesario para alcanzar exactamente `target_frames` dentro de la duración útil del video:

```
fps = target_frames / (duración_total - trim_start - trim_end)
```

**Meta:** 300 frames por video, independientemente de su duración.

```bash
python src/preprocessing.py \
  --video data/processed/videoN_cut.mp4 \
  --output data/frames/videoN \
  --target-frames 300
```

Los frames se guardan como `frame_NNNNNN.jpg` a resolución 1280×720. Cada directorio incluye un `manifest.csv` con timestamp de cada frame.

---

## 6. Fase 3 — Calibración de ROI por cámara

> **Este paso es obligatorio para cualquier cámara nueva** y es la razón por la que el sistema **no puede operar sobre frames crudos sin calibración previa** (ver Sección 9).

Para cada video se delimitaron manualmente los polígonos de cada asiento visible usando la herramienta de calibración browser-based (`calibrate_seats`). El resultado es un archivo JSON por video:

```json
[
  {
    "seat_id": "A1",
    "polygon": [[11, 226], [206, 223], [211, 633], [5, 628]]
  },
  {
    "seat_id": "A2",
    "polygon": [[231, 631], [478, 628], [471, 217], [220, 218]]
  }
]
```

Los archivos se almacenan en `data/calibration/seats_videoN.json`. Las coordenadas están en píxeles sobre resolución 1280×720.

El sistema convierte internamente cada polígono a bounding box con padding de 10 px:

```python
x0 = max(0, min(xs) - 10)
y0 = max(0, min(ys) - 10)
x1 = min(1280, max(xs) + 10)
y1 = min(720, max(ys) + 10)
```

---

## 7. Fase 4 — Anotación manual

Sobre los frames extraídos se etiquetó manualmente cada asiento en cada frame usando la herramienta de etiquetado (`Seat Labeler`), con atajos de teclado Q/W/E para las tres clases y avance automático entre asientos.

**Estadísticas del dataset:**

| Video | Entries | Libre | Reservado | Ocupado |
|-------|---------|-------|-----------|---------|
| video1 | 1,196 | 521 | 198 | 477 |
| video2 | 1,500 | 705 | 19 | 776 |
| video3 | 1,200 | 0 | 149 | 1,051 |
| video4 | 1,200 | 459 | 386 | 355 |
| video5 | 1,200 | 0 | 600 | 600 |
| video6 | 1,435 | 549 | 74 | 812 |
| video7 | 328 | 187 | 2 | 139 |
| video9cut1 | 1,200 | 180 | 124 | 896 |
| video9cut2 | 1,200 | 0 | 126 | 1,074 |
| **TOTAL** | **10,459** | **2,601** | **1,678** | **6,180** |

Las anotaciones se almacenan en `data/ground_truth/labels_videoN.json`:

```json
[
  {"frame_file": "frame_000001.jpg", "seat_id": "A1", "status": "libre"},
  {"frame_file": "frame_000001.jpg", "seat_id": "A2", "status": "ocupado"}
]
```

---

## 8. Fase 5 — Entrenamiento

### Split de evaluación

Se usa **leave-one-camera-out** (`camera_holdout`): todos los frames del video holdout se reservan exclusivamente para validación, nunca aparecen en entrenamiento. Esto mide generalización real a cámaras no vistas.

**Holdout elegido: `video4`** — distribuye equitativamente las tres clases (459/386/355), garantizando que la evaluación cubra libre, reservado y ocupado.

| Split | Videos | Entries |
|-------|--------|---------|
| Train | video1,2,3,5,6,7,9cut1,9cut2 | ~9,259 |
| Val | video4 | 1,200 |

### Preprocesamiento de imagen (siguiendo el paper)

- Redimensionado a 224×224
- Ecualización HSV canal V (normalización de iluminación)
- Normalización ImageNet (mean/std)

### Augmentación (solo train)

```python
RandomHorizontalFlip()
ColorJitter(brightness=0.25, contrast=0.25)
RandomRotation(5)
RandomPerspective(distortion_scale=0.2, p=0.3)  # robustez multi-ángulo
RandomAffine(degrees=10, translate=(0.05, 0.05))
```

### Hiperparámetros

| Parámetro | Canal 1 | Canal 2 |
|-----------|---------|---------|
| Backbone | ResNet18 | ResNet18 |
| Capas entrenadas | layer4 + fc | layer4 + fc |
| Epochs máx. | 20 | 30 |
| Early stopping | patience=5 | patience=5 |
| Batch size | 16 | 16 |
| Learning rate | 1e-3 | 5e-4 |
| Loss | CrossEntropy ponderada | CrossEntropy ponderada |

Los pesos de clase son inverso de frecuencia para compensar el desbalance entre clases.

### Comandos de entrenamiento

```bash
# Canal 2 — clasificador triclase
python src/train_classifier.py --split camera_holdout --epochs 30 --lr 5e-4 --patience 5

# Canal 1 — detector binario de persona
python src/train_detector.py --split camera_holdout --epochs 20 --lr 1e-3 --patience 5
```

---

## 9. Cómo procesar un nuevo video

### Requisito fundamental: la calibración de polígonos es obligatoria

> **El sistema NO puede operar sobre frames crudos sin calibración previa.**
>
> La razón es arquitectural: tanto Canal 1 como Canal 2 reciben como entrada un **recorte (crop) del asiento individual**, no el frame completo. Sin los polígonos que delimitan dónde está cada asiento en el encuadre, el sistema no sabe qué región recortar.

Esto contrasta con detectores genéricos como YOLO, que buscan objetos libremente en toda la imagen. En este sistema la posición de los asientos es conocida y fija para cada cámara, lo que permite recortes precisos con mucho menos cómputo y mayor resistencia a falsos positivos.

### Pasos para un video nuevo

#### Paso 1 — Convertir y recortar el video

```bash
# Quitar audio y ajustar duración si es necesario
ffmpeg -i video_nuevo.mp4 -an -c:v copy data/processed/video_nuevo.mp4
```

#### Paso 2 — Extraer primer frame para calibración

```bash
ffmpeg -i data/processed/video_nuevo.mp4 -vframes 1 -q:v 2 \
  data/frames/first_frames/video_nuevo_frame0.jpg
```

#### Paso 3 — Calibrar los polígonos de asientos

Abre la herramienta de calibración (artefacto browser-based) y:

1. Carga el primer frame del video
2. Dibuja un polígono sobre cada asiento visible haciendo clic en sus esquinas
3. Asigna un `seat_id` a cada asiento (ej. `A1`, `A2`, …)
4. Exporta el JSON y guárdalo como `data/calibration/seats_video_nuevo.json`

El archivo resultante debe tener este formato:

```json
[
  {"seat_id": "A1", "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]},
  {"seat_id": "A2", "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}
]
```

Las coordenadas son en píxeles sobre la resolución original del video (1280×720).

#### Paso 4 — Añadir el video al config

Edita `config/pipeline_config.yaml` añadiendo una entrada en `videos`:

```yaml
videos:
  # ... entradas existentes ...
  video_nuevo:
    seats_config: data/calibration/seats_video_nuevo.json
    frames_dir: data/frames/video_nuevo
```

#### Paso 5 — Extraer frames

```bash
python src/preprocessing.py \
  --video data/processed/video_nuevo.mp4 \
  --output data/frames/video_nuevo \
  --target-frames 300
```

#### Paso 6 — Inferencia (cuando esté implementado `occupancy_detect.py`)

Una vez integrados ambos canales, el flujo de predicción frame a frame será:

```python
from config import load_config
from seat_dataset import seat_bboxes

cfg = load_config("config/pipeline_config.yaml")
bboxes = seat_bboxes("data/calibration/seats_video_nuevo.json")

# Para cada frame:
#   1. Recortar cada seat_id según su bbox
#   2. Pasar crop a Canal 1 → si persona: estado = "ocupado"
#   3. Si no persona → pasar crop a Canal 2 → "reservado" o "libre"
```

### Resumen: ¿qué se necesita para un video nuevo?

| Elemento | ¿Obligatorio? | Descripción |
|----------|--------------|-------------|
| Video procesado (sin audio) | ✅ Sí | `data/processed/video_nuevo.mp4` |
| Polígonos de asientos calibrados | ✅ Sí | `data/calibration/seats_video_nuevo.json` |
| Frames extraídos | ✅ Sí (para inferencia frame a frame) | `data/frames/video_nuevo/` |
| Anotaciones manuales | ❌ No (solo si se va a re-entrenar) | `data/ground_truth/labels_video_nuevo.json` |
| Re-entrenamiento | ❌ No | Los modelos actuales generalizan a nuevas cámaras |

---

## 10. Estructura del repositorio

```
occupancy-jbup/
├── config/
│   └── pipeline_config.yaml       # configuración central
├── data/
│   ├── calibration/               # polígonos ROI por video (seats_videoN.json)
│   ├── frames/
│   │   ├── first_frames/          # primer frame de cada cámara (referencia)
│   │   └── videoN/                # frames extraídos por video
│   └── ground_truth/              # anotaciones manuales (labels_videoN.json)
├── models/
│   └── weights/                   # pesos entrenados (.pt) — excluidos del repo
├── output/
│   └── reports/                   # reportes de evaluación
├── src/
│   ├── config.py                  # carga de pipeline_config.yaml
│   ├── preprocessing.py           # extracción de frames con auto-FPS
│   ├── seat_dataset.py            # dataset multi-video con soporte de video_id
│   ├── train_classifier.py        # Canal 2: triclase libre/reservado/ocupado
│   ├── train_detector.py          # Canal 1: binario persona/sin-persona
│   └── evaluate.py                # métricas y reporte markdown
└── Antecedentes/                  # paper de referencia
```
