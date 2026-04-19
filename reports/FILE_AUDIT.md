# File audit — orphans + broken links

Automated audit performed 2026-04-18 at the same time the project indexes were created. Cross-referenced all `.md` files against:

- `INDEX.md` (root)
- `reports/INDEX.md`
- `reports/pitch/INDEX.md`
- every `.py`, `.md`, `.ipynb`, `.txt`, `.json`, `.yaml`, `.toml` under the repo (excluding `.venv`, `.git`, `__pycache__`, `cache`)

## Summary

| Category | Count |
|---|---:|
| Markdown files in repo | 57 |
| Orphans (no reference anywhere) | 0 |
| Broken intra-repo links in the three INDEX docs | 0 |
| Python scripts under `scripts/` | 56 |
| Scripts mentioned in root `INDEX.md` | 56 |

## Broken links

None. All Markdown `[text](...)` references in the three index files resolve to an existing file.

## Orphan Markdown files

None. Every `.md` under the repo (other than this very audit document and the index documents themselves) is referenced by at least one `INDEX.md` or another `.md`/`.py` file.

## Non-Markdown artifacts

The following auto-generated companion files live under `reports/` but are not individually indexed — they are read programmatically by the corresponding Markdown report:

- JSON: `advanced_features_results.json`, `autoresearch_h{1,2,3,4,8}_*.json`, `channel_survey_summary.json`, `data_audit.json`, `ensemble_results.json`, `error_analysis.json`, `metrics_upgrade.json`, `multichannel_results.json`, `multichannel_v2_results.json`, `multiscale_results.json`, `multiscale_tta_results.json`, `synthetic_aug_results.json`
- CSV: `channel_survey.csv`, `error_cases.csv`, `raw_afm_probe.csv`, `pitch/09_biomarker_table.csv`
- Parquet: `best_oof_predictions.parquet`
- Image dirs: `samples/*.png`, `raw_samples/*.png`, `pitch/*.png`

All of these are implicitly covered by the corresponding textual `.md` report.

## Scripts coverage

All 56 scripts in `scripts/*.py` are listed in the root `INDEX.md` under **Code modules → `scripts/`**, grouped by phase (audit, baselines, ensemble, autoresearch, specialists, multi-channel/multi-scale, advanced features, red-team, evaluation, pitch, synthetic, Wave 11 specialists).

## Model bundles coverage

All five bundles under `models/` are listed with their F1 and role in both `INDEX.md` and `README.md`:

- `ensemble_v4_multiscale/` (SHIPPED CHAMPION, 0.6887)
- `ensemble_v2_tta/` (superseded, 0.6562)
- `ensemble_v1_tta/` (superseded, 0.6458)
- `ensemble_v1/` (superseded, 0.6346)
- `dinov2b_tiled_v1/` (baseline, 0.6150)

## Reproducing this audit

```bash
.venv/bin/python - <<'PY'
import re
from pathlib import Path
root = Path('.')
mdfiles = [p for p in root.rglob('*.md')
           if not any(part in ('.venv','.git','__pycache__') for part in p.parts)]
# ... (full script: scans INDEX.md + repo-wide for references, prints orphans)
PY
```

(Full audit script is the inline `python` block in the session that created this document.)
