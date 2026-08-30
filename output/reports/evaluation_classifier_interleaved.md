# Evaluación del clasificador (ResNet18)

Pares evaluados: 240 (sin predicción: 0)
Accuracy global: 0.992

## Por clase

| clase | precision | recall | f1 | support |
|---|---|---|---|---|
| libre | 0.990 | 1.000 | 0.995 | 104 |
| ocupado | 1.000 | 0.989 | 0.995 | 93 |
| reservado | 0.977 | 0.977 | 0.977 | 43 |

## Matriz de confusión (filas=real, columnas=predicho)

| real \ pred | libre | ocupado | reservado |
|---|---|---|---|
| libre | 104 | 0 | 0 |
| ocupado | 0 | 92 | 1 |
| reservado | 1 | 0 | 42 |
