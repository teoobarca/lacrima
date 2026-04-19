# Pitch figures — Hack Košice 2026

| File | Description |
|------|-------------|
| `01_class_distribution.png` | Bar charts: scans per class + unique patients per class (highlighting SucheOko's 2-patient limit). |
| `02_class_morphology_grid.png` | 5×3 grid of preprocessed AFM height maps (afmhot), three samples per class, deterministic seed=42. |
| `03_umap_embedding.png` | UMAP of mean-pooled DINOv2-S scan embeddings, coloured by class (left) and by patient (right). |
| `04_confusion_matrix.png` | Row-normalized LOPO confusion matrix for the DINOv2-S linear probe; diagonal cells in green, errors in red. |
| `05_per_class_metrics.png` | Per-class precision / recall / F1 from the v4 ensemble (person-LOPO), overlaid with top-1/2/3 accuracy reference lines. |
| `06_morphology_comparison.png` | Iconic single-scan morphology per class with biological / Masmali annotations for the pitch slide. |
| `07_topk_and_calibration.png` | Top-k accuracy (scan vs patient), F1 (scan vs patient), and pre/post-Platt reliability diagrams for the v4 champion. |
| `08_gradcam_per_class.png` | Grad-CAM saliency overlays on DINOv2-B 90 nm (two best-scoring scans per class) — pattern commentary in `reports/CLASS_FINGERPRINTS.md`. |
| `09_biomarker_fingerprint.png` | Per-class z-score fingerprint on 12 handcrafted AFM features (roughness + GLCM + fractal D) — numeric values in `09_biomarker_table.csv`. |

All figures rendered at 150 dpi with seaborn `darkgrid` theme.

Companion text: [`../CLASS_FINGERPRINTS.md`](../CLASS_FINGERPRINTS.md) explains what each class's Grad-CAM activation pattern is doing and how it cross-checks against the handcrafted biomarker row.
