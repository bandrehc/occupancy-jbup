# Evaluación del clasificador (ResNet18)

Pares evaluados: 240 (sin predicción: 0)
Accuracy global: 0.992

## Por clase

| clase | precision | recall | f1 | support |
|---|---|---|---|---|
| libre | 0.984 | 1.000 | 0.992 | 63 |
| ocupado | 1.000 | 0.990 | 0.995 | 98 |
| reservado | 0.987 | 0.987 | 0.987 | 79 |

## Matriz de confusión (filas=real, columnas=predicho)

| real \ pred | libre | ocupado | reservado |
|---|---|---|---|
| libre | 63 | 0 | 0 |
| ocupado | 0 | 97 | 1 |
| reservado | 1 | 0 | 78 |
