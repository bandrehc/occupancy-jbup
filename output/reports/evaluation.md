# Evaluación de ocupación

Pares evaluados: 1200 (sin predicción: 0)
Accuracy global: 0.869

## Por clase

| clase | precision | recall | f1 | support |
|---|---|---|---|---|
| libre | 0.787 | 0.989 | 0.876 | 524 |
| ocupado | 0.962 | 0.910 | 0.935 | 478 |
| reservado | 1.000 | 0.455 | 0.625 | 198 |

## Matriz de confusión (filas=real, columnas=predicho)

| real \ pred | libre | ocupado | reservado |
|---|---|---|---|
| libre | 518 | 6 | 0 |
| ocupado | 43 | 435 | 0 |
| reservado | 97 | 11 | 90 |
