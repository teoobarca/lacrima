# Open-set evaluation — honest heldout of `SucheOko`

Heldout class: **SucheOko** (n=14 scans, 2 persons)

Known classes trained: ['ZdraviLudia', 'Diabetes', 'PGOV_Glaukom', 'SklerozaMultiplex']

## Threshold reliability

- AUROC (in-sample known vs OOD unknown): **0.8483**
- AUROC (OOF    known vs OOD unknown): **0.6220** (honest number)
- 4-class OOF weighted F1 (no heldout): 0.7420 (macro 0.6944)

## Threshold scan

| label | T | TPR unknown | FPR known (in-sample) | FPR known (OOF) |
|---|---|---|---|---|
| T=p10-correct | 0.775 | 0.214 | 0.000 | 0.146 |
| T=p25-correct | 0.899 | 0.500 | 0.000 | 0.301 |
| T=p50-correct | 0.990 | 0.714 | 0.212 | 0.531 |
| T=0.40 | 0.400 | 0.000 | 0.000 | 0.000 |
| T=0.50 | 0.500 | 0.000 | 0.000 | 0.000 |
| T=0.60 | 0.600 | 0.071 | 0.000 | 0.035 |
| T=0.70 | 0.700 | 0.143 | 0.000 | 0.111 |

## Unknown-class confidently-misclassified breakdown

- SklerozaMultiplex: 12/14
- ZdraviLudia: 1/14
- Diabetes: 1/14
