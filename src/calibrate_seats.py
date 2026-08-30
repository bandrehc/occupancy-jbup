"""Herramienta interactiva para dibujar los polígonos de ROI de cada asiento
sobre un frame de referencia y guardarlos en data/calibration/seats.json.

Uso:
    python src/calibrate_seats.py
    python src/calibrate_seats.py --frame data/frames/frame_000086.jpg --output data/calibration/seats.json

Controles:
    click izquierdo   agrega un punto al polígono del asiento actual
    u                 deshace el último punto agregado
    n                 cierra el polígono actual (mínimo 3 puntos) y empieza el siguiente asiento
    z                 deshace el último asiento ya cerrado
    s                 guarda seats.json y termina
    q / ESC           sale sin guardar
"""

import argparse
import json
import os

import cv2

DEFAULT_FRAME = "data/frames/frame_000086.jpg"
DEFAULT_OUTPUT = "data/calibration/seats.json"

WINDOW_NAME = "Calibracion de asientos"


class CalibrationState:
    def __init__(self):
        self.completed_seats = []  # list of {"seat_id": str, "polygon": [[x,y],...]}
        self.current_points = []

    def next_seat_id(self):
        return f"A{len(self.completed_seats) + 1}"


def mouse_callback(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.current_points.append([x, y])


def render(base_frame, state):
    frame = base_frame.copy()

    for seat in state.completed_seats:
        pts = seat["polygon"]
        for i in range(len(pts)):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % len(pts)])
            cv2.line(frame, p1, p2, (0, 200, 0), 2)
        cx = sum(p[0] for p in pts) // len(pts)
        cy = sum(p[1] for p in pts) // len(pts)
        cv2.putText(frame, seat["seat_id"], (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    for i, p in enumerate(state.current_points):
        cv2.circle(frame, tuple(p), 4, (0, 0, 255), -1)
        if i > 0:
            cv2.line(frame, tuple(state.current_points[i - 1]), tuple(p), (0, 0, 255), 2)
    if len(state.current_points) >= 3:
        cv2.line(frame, tuple(state.current_points[-1]), tuple(state.current_points[0]), (0, 0, 255), 1)

    hint = f"Asiento actual: {state.next_seat_id()} | puntos: {len(state.current_points)} | click=punto n=cerrar u=deshacer z=deshacer asiento s=guardar q=salir"
    cv2.putText(frame, hint, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base_frame = cv2.imread(args.frame)
    if base_frame is None:
        raise FileNotFoundError(f"No se pudo leer el frame de referencia: {args.frame}")

    state = CalibrationState()

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, state)

    print(__doc__)

    while True:
        cv2.imshow(WINDOW_NAME, render(base_frame, state))
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u"):
            if state.current_points:
                state.current_points.pop()

        elif key == ord("n"):
            if len(state.current_points) >= 3:
                state.completed_seats.append({
                    "seat_id": state.next_seat_id(),
                    "polygon": state.current_points,
                })
                state.current_points = []
            else:
                print("Necesitas al menos 3 puntos para cerrar un poligono.")

        elif key == ord("z"):
            if state.completed_seats:
                removed = state.completed_seats.pop()
                print(f"Asiento {removed['seat_id']} eliminado.")

        elif key == ord("s"):
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(state.completed_seats, f, indent=2)
            print(f"Guardado {len(state.completed_seats)} asientos en {args.output}")
            break

        elif key == ord("q") or key == 27:
            print("Salida sin guardar.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
