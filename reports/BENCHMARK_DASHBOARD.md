# Benchmark Dashboard — person-level LOPO F1

Single-classifier (Logistic Regression, `class_weight='balanced'`) evaluation on person-level LOPO with all cached feature sets.

Last updated automatically by `scripts/benchmark_dashboard.py`.

| config                                             |   f1_weighted |   f1_macro |   dim |   seconds |
|:---------------------------------------------------|--------------:|-----------:|------:|----------:|
| ✨ MULTI-SCALE 90+45+BiomedCLIP-TTA (W7 candidate) |        0.6887 |     0.5541 |  2048 |    0.0000 |
| ★ ENSEMBLE TTA v2 (L2 + geom-mean, SHIPPED)        |        0.6562 |     0.5382 |  1280 |    0.0000 |
| ENSEMBLE TTA v1 (arith-mean, no L2)                |        0.6458 |     0.5154 |  1280 |    0.0000 |
| dinov2_vitb14 TTA (D4, mean-pool)                  |        0.6434 |     0.5275 |   768 |    1.3123 |
| dinov2_vitb14 45nm tiled (mean-pool)               |        0.6433 |     0.5038 |   768 |    1.3026 |
| ENSEMBLE DINOv2-B + BiomedCLIP (tiled, no TTA)     |        0.6346 |     0.4934 |  1280 |    0.0000 |
| dinov2_vitb14 tiled (mean-pool)                    |        0.6150 |     0.4910 |   768 |    1.2205 |
| biomedclip TTA (D4, mean-pool)                     |        0.6135 |     0.4838 |   512 |    1.5445 |
| dinov2_vits14 tiled (mean-pool)                    |        0.5927 |     0.4782 |   384 |    0.7734 |
| biomedclip tiled (mean-pool)                       |        0.5795 |     0.4336 |   512 |    1.4311 |
| dinov2_vits14 single-crop                          |        0.5726 |     0.4442 |   384 |    0.8476 |
| dinov2_vitb14 single-crop                          |        0.5698 |     0.4215 |   768 |    1.0462 |
| biomedclip single-crop                             |        0.5695 |     0.4358 |   512 |    1.4415 |
| handcrafted (94 feat) + LR                         |        0.4882 |     0.3707 |    94 |    0.0000 |
| TDA (1015 feat) + LR                               |        0.4863 |     0.3722 |  1015 |    0.0000 |
| advanced features + LR                             |        0.4707 |     0.3435 |   443 |    0.0000 |
