# Pitch figures — Hack Košice 2026

| File | Description |
|------|-------------|
| `01_class_distribution.png` | Bar charts: scans per class + unique patients per class (highlighting SucheOko's 2-patient limit). |
| `02_class_morphology_grid.png` | 5×3 grid of preprocessed AFM height maps (afmhot), three samples per class, deterministic seed=42. |
| `03_umap_embedding.png` | UMAP of mean-pooled DINOv2-S scan embeddings, coloured by class (left) and by patient (right). |
| `04_confusion_matrix.png` | Row-normalized LOPO confusion matrix for the DINOv2-S linear probe; diagonal cells in green, errors in red. |
| `05_per_class_metrics.png` | Per-class precision / recall / F1 from the LOPO probe; support counts above each group, weighted-F1 baseline as red dashed line. |
| `06_morphology_comparison.png` | Iconic single-scan morphology per class with biological / Masmali annotations for the pitch slide. |

All figures rendered at 150 dpi with seaborn `darkgrid` theme.
