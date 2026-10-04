# Evaluación del clasificador (ResNet18)

Pares evaluados: 1200 (sin predicción: 0)
Accuracy global: 0.999

## Por clase

| clase | precision | recall | f1 | support |
|---|---|---|---|---|
| libre | 0.000 | 0.000 | 0.000 | 0 |
| ocupado | 1.000 | 0.999 | 1.000 | 1074 |
| reservado | 0.992 | 1.000 | 0.996 | 126 |

## Matriz de confusión (filas=real, columnas=predicho)

| real \ pred | libre | ocupado | reservado |
|---|---|---|---|
| libre | 0 | 0 | 0 |
| ocupado | 0 | 1073 | 1 |
| reservado | 0 | 0 | 126 |
