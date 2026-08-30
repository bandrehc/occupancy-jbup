# occupancy-jbup

Detección de ocupación de puestos de estudio en la biblioteca JBUP a partir de
video de cámara fija: extracción de frames, calibración manual de zonas de
interés (ROI) por asiento, etiquetado de ground truth, y un clasificador de
imagen (ResNet18 fine-tuned) que predice el estado de cada asiento —
`libre` / `ocupado` / `reservado` — directamente sobre el recorte de su ROI.

## Estado actual

Pipeline funcional de punta a punta. Modelo de producción entrenado sobre las
1200 etiquetas manuales disponibles (300 frames × 4 asientos); accuracy
estimado por validación cruzada (dos splits independientes, ver
"Por qué un clasificador y no YOLO" más abajo):

| clase | precisión | recall | f1 |
|---|---|---|---|
| libre | 0.985–0.990 | 1.000 | 0.992–0.995 |
| ocupado | 1.000 | 0.989–0.990 | 0.995 |
| reservado | 0.977–0.987 | 0.977–0.987 | 0.977–0.987 |

**Accuracy: 99.2%** en ambas validaciones (`output/reports/evaluation_classifier_tail.md`,
`output/reports/evaluation_classifier_interleaved.md`). El modelo desplegado
(`models/weights/seat_classifier.pt`) se reentrenó sobre el 100% de los datos
una vez validado el enfoque — ver sección de entrenamiento.

### Limitación conocida

El modelo aprende la apariencia específica de **estos 4 asientos desde esta
cámara**. Si la cámara se mueve o cambia el mobiliario, hay que recalibrar
(`seats.json`), re-etiquetar esa nueva vista y reentrenar — no generaliza a
otra escena como sí lo haría un detector de personas genérico. Es un
trade-off aceptado a propósito (ver comparación), no un descuido.

## Pipeline

```
1. Extracción de frames       src/preprocessing.py (extract_frames)
2. Calibración de asientos    data/calibration/seats.json (polígonos ROI por asiento)
3. Etiquetado ground truth    data/ground_truth/*.json (manual, 3 clases)
4. Entrenamiento               src/train_classifier.py (ResNet18 sobre los recortes ROI)
5. Evaluación                  integrada en train_classifier.py (mismo formato que evaluate.py)
```

### 1. Extracción de frames

```
python -c "
import sys; sys.path.insert(0, 'src')
from preprocessing import extract_frames
extract_frames('data/raw/<video>.mp4', 'data/frames')
"
```

Parámetros en `config/pipeline_config.yaml` → `sampling`: fps, ancho de salida,
y `trim_start_seconds`/`trim_end_seconds` para excluir los extremos del video
donde se manipula la cámara (colocación/retiro).

### 2. Calibración de asientos

`data/calibration/seats.json` — un polígono por asiento (silla + zona de
torso). Herramienta interactiva: `python src/calibrate_seats.py` (ventana
OpenCV). Solo hace falta recalibrar si la cámara se mueve.

### 3. Ground truth manual

`data/ground_truth/labels.json` — lista de `{frame_file, timestamp_seconds,
seat_id, status}`, con `status` en `libre` / `ocupado` / `reservado`.
Etiquetado hecho a mano sobre recortes de cada ROI (no el frame completo), un
frame a la vez.

Criterio para `reservado`: silla vacía pero con objetos personales visibles
(mochila, laptop, etc.) que indican que el puesto está tomado aunque nadie
esté sentado en ese instante. Una persona cruzando el fondo sin sentarse no
cambia el estado del asiento afectado.

### 4. Entrenamiento

```
python src/train_classifier.py --split full --epochs 15
```

ResNet18 preentrenado en ImageNet, con solo `layer4` y `fc` reentrenables (el
resto congelado — con ~1000 ejemplos, ajustar la red completa arriesga
sobreajuste). Cabeza reemplazada por una capa de 3 clases. `CrossEntropyLoss`
con pesos inversos a la frecuencia de cada clase (hay más `libre` que
`reservado`). Aumento de datos: flip horizontal, jitter de brillo/contraste,
rotación ±5°.

`--split full` entrena con el 100% de los datos (modelo de producción, sin
held-out — usar solo después de haber validado el enfoque). Para validar antes
de confiar en un cambio (más datos, otra arquitectura, etc.), usar:

```
python src/train_classifier.py --split tail          # último 20% del video como validación
python src/train_classifier.py --split interleaved   # 1 de cada 5 frames, repartido en todo el video
```

Ambos splits separan por `frame_file` completo (nunca al azar por ejemplo
individual): los 4 recortes de un mismo frame están altamente correlacionados,
así que mezclarlos entre train/val infla la métrica artificialmente.

### 5. Evaluación

Integrada al final de `train_classifier.py` cuando hay held-out (`--split
tail`/`interleaved`): genera un reporte con el mismo formato que
`src/evaluate.py` (accuracy, precisión/recall/f1 por clase, matriz de
confusión), para que sea comparable 1:1 contra el pipeline de YOLO.

## Cómo agregar más sesiones de grabación

El pipeline está pensado para crecer con más capturas sin tocar código:

1. Extrae los frames de la sesión nueva en su **propia subcarpeta**:
   `data/frames/<nombre_sesion>/` (por ejemplo `data/frames/2026-09-15/`).
2. Si la cámara sigue en la misma posición, reusa `seats.json` tal cual. Si se
   movió, recalibra (`calibrate_seats.py`) y reemplázalo.
3. Etiqueta esos frames nuevos (misma mecánica manual) y guarda el resultado
   como **un archivo `.json` nuevo** dentro de `data/ground_truth/` — por
   ejemplo `data/ground_truth/labels_2026-09-15.json`. Los `frame_file` deben
   quedar con el prefijo de su subcarpeta, ej. `"2026-09-15/frame_000001.jpg"`.
4. Reentrena: `python src/train_classifier.py --split full --epochs 15`.
   Por defecto, `--labels data/ground_truth` (el directorio) junta **todos**
   los `.json` que encuentre ahí — no hace falta fusionar archivos a mano.

Antes de confiar en el modelo resultante, vale la pena correr `--split tail`
(deja la sesión más nueva fuera del entrenamiento) para confirmar que lo
aprendido en la sesión anterior generaliza a la nueva, y no solo memorizó la
primera.

## Por qué un clasificador y no YOLO

El enfoque original (todavía en el repo, `src/occupancy_detect.py`) corría
YOLO (detección de personas + objetos COCO) y cruzaba cada detección contra
los polígonos de asiento con una heurística de solape + banda de confianza
dudosa + recorte de refinamiento. Funcionaba (86.9% accuracy tras varias
rondas de calibración — ver `output/reports/evaluation.md`), pero:

- **`reservado` tenía un techo duro**: YOLOv8n detecta mochilas/laptops con
  confianza muy baja (~0.25–0.35 máx.) en este video borroso → 45.5% recall,
  sin importar cuánto se ajustara el umbral.
- El clasificador no tiene ese techo: aprende directamente de las 1200
  etiquetas cualquier señal visual usada al etiquetar (no solo "hay una
  mochila/laptop/cartera/maleta de COCO"), y sube `reservado` a 97.7-98.7%
  recall.
- Es la herramienta correcta para el objetivo real: **no** hacía falta
  detectar objetos ni generalizar a otra escena — solo clasificar el estado
  de 4 asientos fijos, y eso es exactamente lo que optimiza un clasificador
  entrenado en la tarea, en vez de derivarlo de un detector genérico.

El trade-off (documentado arriba en "Limitación conocida") es que YOLO
generaliza a cámaras/salas nuevas sin reentrenar y el clasificador no. Para
este proyecto (cámara fija de esta biblioteca) esa generalización no se
necesita, así que no vale el costo en precisión.

`occupancy_detect.py` y `evaluate.py` se mantienen en el repo como referencia
y punto de comparación, no como pipeline activo.

## Estructura

```
config/pipeline_config.yaml    parámetros del pipeline (sampling, detección YOLO, matching)
data/raw/                      video(s) fuente (no versionado)
data/frames/                   frames extraídos por sesión + manifest.csv
data/calibration/seats.json    polígonos ROI por asiento
data/ground_truth/*.json       etiquetado manual, un archivo por sesión
data/processed/                videos/artefactos de demo renderizados
models/weights/                 pesos de YOLO y del clasificador (no versionado)
output/reports/                 predicciones + reportes de evaluación
src/preprocessing.py            extracción de frames, normalización de iluminación
src/calibrate_seats.py          calibración interactiva de polígonos (OpenCV)
src/seat_dataset.py             dataset de recortes ROI + splits de validación
src/train_classifier.py         entrenamiento del clasificador (pipeline activo)
src/render_demo.py              renderiza un tramo de video con cajas + estado superpuestos
src/occupancy_detect.py         detección YOLO + heurística de solape (referencia histórica)
src/evaluate.py                 evaluación contra ground truth (usado por ambos enfoques)
```

## Setup

```
pip install -r requirements.txt
```

Requiere `ffmpeg`/`ffprobe` en el PATH para la extracción de frames.
