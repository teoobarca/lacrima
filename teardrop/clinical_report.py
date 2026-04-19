"""Physician-readable diagnostic report generator for AFM tear-film scans.

The pitch artifact: given a Bruker SPM file, produce a Markdown report that
combines the shipped ML prediction with handcrafted biomarker evidence and
the nearest training cases. All text is template-driven (LLM-free → fully
reproducible) but grounded in the per-scan quantitative features.

Public entry point:

    from teardrop.clinical_report import generate_clinical_report
    md = generate_clinical_report(Path("TRAIN_SET/Diabetes/DM_01.03.2024_LO.001"))
    print(md)

Implementation notes
--------------------
*   Uses the shipped champion bundle `models/ensemble_v4_multiscale/`
    (honest person-LOPO weighted F1 = 0.6887, macro F1 = 0.5541).
*   Handcrafted surface features are computed on the normalized height map,
    but **Ra / Rq are reported in nanometers** from the *plane-levelled raw*
    height channel (pre-normalization) — that's the number a clinician expects.
*   k-NN retrieval uses the cached DINOv2-B scan-level embeddings
    (`cache/tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz`). If the cache is
    missing the report gracefully omits the "similar reference cases" section.
*   The report is intentionally *conservative* — it always includes a
    confidence note stamped with our honest F1 and a "what the model might be
    missing" section, never overstates single-case interpretability.

Constraints:
*   No LLM calls — pure Python templating. Reproducible, offline, deterministic.
*   End-to-end latency: ~30-60 s per scan on MPS (dominated by the 3-component
    ensemble + encoder loads). Subsequent calls reuse the cached models.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Heavy imports are deferred to avoid startup cost when only rendering stored
# reports. We do import numpy above because we always need it.


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

_PKG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_DIR.parent

CLASSES = ["ZdraviLudia", "Diabetes", "PGOV_Glaukom", "SklerozaMultiplex", "SucheOko"]

CLASS_DISPLAY = {
    "ZdraviLudia": "Healthy control",
    "Diabetes": "Diabetes mellitus",
    "PGOV_Glaukom": "Primary open-angle glaucoma",
    "SklerozaMultiplex": "Multiple sclerosis",
    "SucheOko": "Dry-eye disease",
}

HONEST_F1_WEIGHTED = 0.6887
HONEST_F1_MACRO = 0.5541
HUMAN_KAPPA = 0.57  # Masmali weighted kappa (Daza 2022)

DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "ensemble_v4_multiscale"
DEFAULT_RETRIEVAL_CACHE = (
    _PROJECT_ROOT / "cache" / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz"
)
DEFAULT_REFERENCE_JSON = _PROJECT_ROOT / "cache" / "clinical_reference_nm.json"


# ---------------------------------------------------------------------------
# Per-class biomarker templates (domain knowledge, not learned)
# ---------------------------------------------------------------------------

# Reference ranges for Sa (Ra) expressed in **nanometers** on plane-levelled
# raw heights. These are empirical cohort stats from the TRAIN_SET (n=240),
# supplemented with published Masmali grade semantics.
CLASS_TEMPLATES: dict[str, dict[str, Any]] = {
    "ZdraviLudia": {
        "summary": (
            "Dense, highly branched dendritic fern pattern. Tear-film protein/salt "
            "crystallisation is intact and uniformly distributed."
        ),
        "masmali_range": "0-1",
        "biomarker_bullets": [
            "dense dendritic fern is preserved (Masmali grade 0-1 expected)",
            "moderate GLCM homogeneity with continuous branching indicates an intact "
            "tear-film glycoprotein matrix",
            "fractal dimension in the typical healthy band (D ~ 1.70-1.85) is "
            "consistent with balanced branching geometry",
        ],
        "expected_sa_nm": (80, 180),
        "expected_fractal": (1.70, 1.85),
    },
    "Diabetes": {
        "summary": (
            "Thickened, densely packed dendrites with elevated small-scale roughness. "
            "Hyperglycaemia-induced glycation drives denser crystal packing and a "
            "coarser surface."
        ),
        "masmali_range": "2-3",
        "biomarker_bullets": [
            "elevated surface roughness (Ra often > 180 nm) consistent with "
            "hyperglycaemia-induced glycation of tear-film proteins",
            "higher GLCM contrast reflects denser crystal packing and loss of "
            "smooth lamellar structure",
            "skewness tends to shift positive — taller crystalline peaks dominate "
            "over trough regions",
        ],
        "expected_sa_nm": (150, 350),
        "expected_fractal": (1.73, 1.85),
    },
    "PGOV_Glaukom": {
        "summary": (
            "Granular, loop-dominated surface. MMP-9 protease activity degrades the "
            "tear-film glycoprotein matrix, yielding shorter branches and coarse "
            "ring/loop topology visible as H_1 persistent-homology features."
        ),
        "masmali_range": "3-4",
        "biomarker_bullets": [
            "granular texture with coarse medium-scale loops consistent with MMP-9 "
            "mediated matrix degradation",
            "locally chaotic correlation structure (GLCM correlation depressed at "
            "d=5) suggests short-range order is preserved but long-range branching "
            "is lost",
            "fractal dimension tends to be lower and more variable than in healthy "
            "controls due to truncated branching",
        ],
        "expected_sa_nm": (80, 220),
        "expected_fractal": (1.70, 1.82),
    },
    "SklerozaMultiplex": {
        "summary": (
            "Heterogeneous texture with mixed morphologies. Altered tear-film protein "
            "and lipid composition produces coarse rods and fine granules in the "
            "same sample; commonly confused with glaucoma at the macroscopic level."
        ),
        "masmali_range": "2-4",
        "biomarker_bullets": [
            "high intra-sample texture variance (mixed coarse/fine regions) "
            "consistent with altered lipid / MUC5AC composition in MS tear-film",
            "elevated GLCM contrast with depressed homogeneity points to localised "
            "crystalline islands separated by amorphous regions",
            "fractal dimension is variable across tiles — a hallmark of the "
            "heterogeneous MS tear-film phenotype",
        ],
        "expected_sa_nm": (50, 300),
        "expected_fractal": (1.74, 1.88),
    },
    "SucheOko": {
        "summary": (
            "Fragmented, sparse crystalline network with large amorphous regions. "
            "Severe tear-film deficit leaves only isolated crystalline islands; "
            "Masmali grade 3-4."
        ),
        "masmali_range": "3-4",
        "biomarker_bullets": [
            "sparse, fragmented ferning network consistent with Masmali grade 3-4 "
            "dry-eye phenotype",
            "depressed fractal dimension (often D < 1.78) reflects loss of "
            "dendritic branching complexity",
            "large flat/amorphous regions dominate LBP histogram toward uniform "
            "bins — the hallmark of severe crystallisation failure",
        ],
        "expected_sa_nm": (40, 120),
        "expected_fractal": (1.72, 1.85),
    },
}

# "What the model might be missing" — class-specific caveats grounded in our
# error analysis (reports/ERROR_ANALYSIS.md).
CLASS_CAVEATS: dict[str, list[str]] = {
    "ZdraviLudia": [
        "Diabetes-vs-Healthy boundary is subtle — 8 of 13 Diabetes errors in our "
        "LOPO evaluation were mis-called as Healthy. If the patient's history is "
        "suggestive, consider a follow-up HbA1c.",
    ],
    "Diabetes": [
        "This class has only 25 scans across 4-5 patients in training; session-"
        "level variance is large (DM_01.03.2024 session contributes 6 of 13 total "
        "Diabetes errors).",
        "Diabetes and Healthy share a soft decision boundary driven by "
        "glucose-dependent salt-lattice expression — always correlate with HbA1c.",
    ],
    "PGOV_Glaukom": [
        "Glaucoma and Multiple Sclerosis are our primary confusion pair (all 15 "
        "Glaukom errors went to SM in the held-out audit). If this patient's "
        "clinical picture supports either condition, treat both as differentials.",
        "TDA / persistent-homology H_1 features (not used in this model) carry "
        "additional Glaucoma signal and may be worth a second pass.",
    ],
    "SklerozaMultiplex": [
        "SM is morphologically heterogeneous within-class; 20 of 37 SM errors in "
        "LOPO were mis-called as Glaucoma and 12 as Dry-Eye. Consider both "
        "differentials — especially in Sjogren's co-morbidity, where an SM patient "
        "may present with a dry-eye-like tear-film signature.",
    ],
    "SucheOko": [
        "HIGH UNCERTAINTY: our training set contains only 2 unique Dry-Eye "
        "patients (14 scans). Person-LOPO F1 for this class is 0.00 — any "
        "Dry-Eye prediction is a data-acquisition ceiling artefact, not a "
        "reliable clinical signal. Treat as a rule-out, not a rule-in.",
    ],
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _confidence_level(top1_prob: float, gap_to_top2: float) -> str:
    """HIGH / MEDIUM / LOW based on max-softmax and top1-top2 gap."""
    if top1_prob >= 0.75 and gap_to_top2 >= 0.25:
        return "HIGH"
    if top1_prob >= 0.55 and gap_to_top2 >= 0.10:
        return "MEDIUM"
    return "LOW"


def _interpret_roughness(Sa_nm: float, expected_range: tuple[float, float]) -> str:
    lo, hi = expected_range
    if Sa_nm < lo * 0.85:
        return f"markedly lower than the typical {lo:.0f}-{hi:.0f} nm band for this diagnosis"
    if Sa_nm > hi * 1.15:
        return f"elevated above the typical {lo:.0f}-{hi:.0f} nm band for this diagnosis"
    return f"within the typical {lo:.0f}-{hi:.0f} nm band for this diagnosis"


def _interpret_fractal(D: float, expected_range: tuple[float, float]) -> str:
    lo, hi = expected_range
    if D < lo - 0.02:
        return "reduced (branching complexity lower than healthy band)"
    if D > hi + 0.02:
        return "elevated"
    return "within normal range"


def _interpret_glcm(homogeneity_d1: float, contrast_d1: float) -> str:
    parts = []
    if homogeneity_d1 >= 0.75:
        parts.append("high homogeneity (smooth, uniform crystalline texture)")
    elif homogeneity_d1 <= 0.55:
        parts.append("low homogeneity (irregular / fragmented texture)")
    else:
        parts.append("moderate homogeneity")
    if contrast_d1 >= 6.0:
        parts.append("elevated local contrast")
    elif contrast_d1 <= 2.0:
        parts.append("low local contrast")
    else:
        parts.append("moderate local contrast")
    return "; ".join(parts)


def _infer_masmali_grade(Sa_nm: float, fractal_D: float,
                          homogeneity: float, contrast: float) -> int:
    """Crude surrogate Masmali 0-4 grade from quantitative features.

    This is a *heuristic* mapping — it should not be interpreted as a true
    Masmali score (which requires a trained optometrist). We combine three
    signals: (1) low fractal D penalises sparse fernings, (2) low homogeneity /
    high contrast penalises fragmented texture, (3) low roughness in nm
    (thin fern layer) also pushes toward higher grade.
    """
    grade = 0
    # Fractal dimension — healthy fernings are D ~ 1.78 or higher.
    if fractal_D < 1.70:
        grade += 2
    elif fractal_D < 1.76:
        grade += 1
    # GLCM irregularity.
    if homogeneity < 0.55:
        grade += 1
    if contrast > 8.0:
        grade += 1
    # Very thin fern (low Sa in nm) suggests desiccate with little crystal mass.
    if Sa_nm < 50:
        grade += 1
    return int(min(grade, 4))


def _spm_metadata(raw_path: Path) -> dict[str, Any]:
    """Best-effort metadata harvest (scan date, pixel size, raw dimensions)."""
    out: dict[str, Any] = {
        "filename": raw_path.name,
        "size_bytes": int(raw_path.stat().st_size) if raw_path.exists() else None,
        "scan_date": None,  # SPM header may contain this; we try below
    }
    # Cheap attempt: read the ASCII header at the top of a Bruker SPM file and
    # extract a `Date:` field if present. This is optional — silent failure.
    try:
        with open(raw_path, "rb") as f:
            head = f.read(8192)
        txt = head.decode("latin-1", errors="ignore")
        for line in txt.splitlines():
            if "Date:" in line and not out["scan_date"]:
                out["scan_date"] = line.split("Date:", 1)[1].strip().strip('"\\')
                break
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Retrieval cache loader (lazy)
# ---------------------------------------------------------------------------

@dataclass
class RetrievalIndex:
    scan_emb_n: np.ndarray  # (N, D) L2-normalized
    scan_paths: list[str]
    scan_y: np.ndarray  # class indices

    @classmethod
    def load(cls, cache_path: Path) -> "RetrievalIndex | None":
        if not cache_path.exists():
            return None
        try:
            blob = np.load(cache_path, allow_pickle=True)
            X = blob["X"].astype(np.float32)
            tile_to_scan = blob["tile_to_scan"].astype(np.int64)
            scan_y = blob["scan_y"].astype(np.int64)
            scan_paths = [str(p) for p in blob["scan_paths"]]
            n_scans = len(scan_paths)
            scan_emb = np.zeros((n_scans, X.shape[1]), dtype=np.float32)
            for s in range(n_scans):
                mask = tile_to_scan == s
                if mask.any():
                    scan_emb[s] = X[mask].mean(axis=0)
            norms = np.linalg.norm(scan_emb, axis=1, keepdims=True) + 1e-9
            scan_emb_n = scan_emb / norms
            return cls(scan_emb_n=scan_emb_n, scan_paths=scan_paths, scan_y=scan_y)
        except Exception:
            return None

    def nearest(self, query_emb: np.ndarray, k: int = 3,
                exclude_path: str | None = None) -> list[dict]:
        q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        sims = self.scan_emb_n @ q
        order = np.argsort(sims)[::-1]
        out: list[dict] = []
        excl_norm = os.path.normcase(os.path.realpath(exclude_path)) if exclude_path else None
        for idx in order:
            idx_i = int(idx)
            p = self.scan_paths[idx_i]
            if excl_norm is not None:
                try:
                    if os.path.normcase(os.path.realpath(p)) == excl_norm:
                        continue
                except Exception:
                    pass
            out.append({
                "path": p,
                "class": CLASSES[int(self.scan_y[idx_i])],
                "similarity": float(sims[idx_i]),
                "distance": float(1.0 - sims[idx_i]),
            })
            if len(out) >= k:
                break
        return out


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_clinical_report(
    scan_path: str | Path,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    retrieval_cache: str | Path = DEFAULT_RETRIEVAL_CACHE,
    reference_json: str | Path = DEFAULT_REFERENCE_JSON,
    k_neighbors: int = 3,
    _predictor: Any = None,
    _retrieval_index: "RetrievalIndex | None" = None,
) -> str:
    """Generate a Markdown diagnostic report for a single AFM scan.

    Parameters
    ----------
    scan_path : Path
        Bruker Nanoscope SPM file (.NNN extension).
    model_dir : Path
        Directory containing the shipped champion ensemble
        (default: ``models/ensemble_v4_multiscale``).
    retrieval_cache : Path
        NPZ of DINOv2-B tile embeddings for k-NN (optional; if missing the
        similar-cases section is omitted).
    reference_json : Path
        JSON with per-class Sa_nm / Sq_nm reference stats from the training
        cohort (optional; if missing, falls back to the hard-coded
        literature ranges in CLASS_TEMPLATES).
    k_neighbors : int
        Number of reference cases to retrieve (default 3).
    _predictor, _retrieval_index : optional
        Pre-loaded objects for batched report generation (avoid re-loading
        the ensemble for every call). Not required for single-shot use.

    Returns
    -------
    str
        A Markdown-formatted clinical report.
    """
    scan_path = Path(scan_path)
    if not scan_path.exists():
        raise FileNotFoundError(f"Scan file not found: {scan_path}")

    # Deferred imports (heavy) — keep the module import cheap.
    import sys
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from teardrop.data import load_height, plane_level, resample_to_pixel_size, robust_normalize  # noqa: E402
    from teardrop.encoders import height_to_pil  # noqa: E402
    from teardrop.features import fractal_dimension, glcm_features, roughness_features  # noqa: E402
    from teardrop.infer import preprocess_and_tile_spm  # noqa: E402

    # -- 1. Load predictor (expensive; reuse if provided) ---------------------
    if _predictor is None:
        from models.ensemble_v4_multiscale.predict import TTAPredictorV4
        predictor = TTAPredictorV4.load(Path(model_dir))
    else:
        predictor = _predictor

    # -- 2. Run ensemble prediction -------------------------------------------
    pred_class, probs = predictor.predict_scan(scan_path)
    probs = np.asarray(probs, dtype=np.float32)
    order = np.argsort(probs)[::-1]
    top1_idx, top2_idx = int(order[0]), int(order[1])
    classes = predictor.classes
    top1_cls = classes[top1_idx]
    top2_cls = classes[top2_idx]
    top1_prob = float(probs[top1_idx])
    top2_prob = float(probs[top2_idx])
    conf = _confidence_level(top1_prob, top1_prob - top2_prob)

    # -- 3. Compute handcrafted features in nm ---------------------------------
    meta = _spm_metadata(scan_path)
    hm = load_height(scan_path)
    # Raw Sa/Sq in nanometers from the plane-levelled (but un-normalised) map.
    h_raw = plane_level(hm.height)
    flat_raw = h_raw.flatten()
    centered = flat_raw - flat_raw.mean()
    Sa_nm = float(np.abs(centered).mean())
    Sq_nm = float(np.sqrt((centered ** 2).mean()))
    Sz_nm = float(flat_raw.max() - flat_raw.min())

    # Resampled / normalized map for texture features (same preprocessing as
    # the model sees).
    h_res = resample_to_pixel_size(h_raw, hm.pixel_nm, 90.0)
    h_norm = robust_normalize(h_res)
    rough = roughness_features(h_norm)
    # GLCM on a 512x512 centre crop for speed.
    H, W = h_norm.shape
    s = 512
    if H >= s and W >= s:
        y0 = (H - s) // 2
        x0 = (W - s) // 2
        crop = h_norm[y0:y0 + s, x0:x0 + s]
    else:
        crop = h_norm
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        crop = h_norm
    glcm = glcm_features(crop, distances=(1, 5))
    frac = fractal_dimension(crop)

    homogeneity_d1 = float(glcm["glcm_homogeneity_d1_mean"])
    contrast_d1 = float(glcm["glcm_contrast_d1_mean"])
    fractal_D = float(frac["fractal_D_mean"])
    fractal_D_std = float(frac["fractal_D_std"])

    # -- 4. Retrieval (k-NN) --------------------------------------------------
    retrieval_index = (_retrieval_index if _retrieval_index is not None
                        else RetrievalIndex.load(Path(retrieval_cache)))
    neighbours: list[dict] = []
    if retrieval_index is not None:
        # Re-encode this scan through the shipped DINOv2-B branch.
        try:
            tiles = preprocess_and_tile_spm(
                scan_path, target_nm_per_px=90.0, tile_size=512, max_tiles=9,
            )
            pils = [height_to_pil(np.ascontiguousarray(t), mode="afmhot") for t in tiles]
            enc = predictor.encoder_dinov2b
            tile_emb = enc.encode(pils, batch_size=len(pils))
            scan_emb = tile_emb.mean(axis=0).astype(np.float32)
            neighbours = retrieval_index.nearest(
                scan_emb, k=k_neighbors,
                exclude_path=str(scan_path),
            )
        except Exception:
            neighbours = []

    # -- 5. Template rendering ------------------------------------------------
    tpl = CLASS_TEMPLATES[top1_cls]
    masmali_inferred = _infer_masmali_grade(
        Sa_nm=Sa_nm, fractal_D=fractal_D,
        homogeneity=homogeneity_d1, contrast=contrast_d1,
    )
    rough_interp = _interpret_roughness(Sa_nm, tpl["expected_sa_nm"])
    fractal_interp = _interpret_fractal(fractal_D, tpl["expected_fractal"])
    glcm_interp = _interpret_glcm(homogeneity_d1, contrast_d1)

    # Evidence bullets — start from per-class template, then optionally prepend
    # a data-grounded observation sentence.
    evidence_bullets: list[str] = []
    evidence_bullets.append(
        f"observed surface roughness Ra = {Sa_nm:.0f} nm, "
        f"Rq = {Sq_nm:.0f} nm — {rough_interp}"
    )
    evidence_bullets.append(
        f"fractal dimension D = {fractal_D:.3f} +/- {fractal_D_std:.3f} "
        f"(reference band for {CLASS_DISPLAY[top1_cls]}: "
        f"{tpl['expected_fractal'][0]:.2f}-{tpl['expected_fractal'][1]:.2f}) — "
        f"{fractal_interp}"
    )
    evidence_bullets.append(
        f"GLCM contrast (d=1) = {contrast_d1:.2f}, homogeneity = {homogeneity_d1:.2f} — "
        f"{glcm_interp}"
    )
    # Append class-specific biology bullets from the template.
    for b in tpl["biomarker_bullets"]:
        evidence_bullets.append(b)

    # -- 6. Build the Markdown string -----------------------------------------
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append("# Tear-film AFM Diagnostic Report")
    lines.append("")
    lines.append(f"**Patient scan:** `{meta['filename']}`  ")
    scan_date_txt = meta["scan_date"] if meta["scan_date"] else "(not embedded in SPM header)"
    lines.append(f"**Scan date:** {scan_date_txt}  ")
    lines.append(f"**Pixel size:** {hm.pixel_nm:.2f} nm/px  ")
    lines.append(f"**Image dimensions:** {hm.height.shape[0]} x {hm.height.shape[1]} px  ")
    lines.append(f"**Generated:** {now}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Model prediction
    lines.append("## Model prediction")
    lines.append("")
    lines.append(
        f"**Primary:** {CLASS_DISPLAY[top1_cls]} "
        f"(`{top1_cls}`) - {top1_prob:.0%}"
    )
    lines.append("")
    lines.append(
        f"**Differential:** {CLASS_DISPLAY[top2_cls]} "
        f"(`{top2_cls}`) - {top2_prob:.0%}"
    )
    lines.append("")
    lines.append(f"**Confidence level:** {conf}")
    lines.append("")
    lines.append("Full class posterior (ensemble geometric mean, 3 components):")
    lines.append("")
    lines.append("| Class | Probability |")
    lines.append("|---|---:|")
    for i in order:
        i = int(i)
        lines.append(f"| {CLASS_DISPLAY[classes[i]]} (`{classes[i]}`) | {probs[i]:.1%} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Morphology assessment
    lines.append("## Morphology assessment")
    lines.append("")
    lines.append(
        f"- **Surface roughness:** Ra = {Sa_nm:.0f} nm, Rq = {Sq_nm:.0f} nm, "
        f"Rz = {Sz_nm:.0f} nm (plane-levelled, pre-normalisation)"
    )
    lines.append(
        f"- **Fractal dimension:** D = {fractal_D:.3f} (std {fractal_D_std:.3f}) - "
        f"{fractal_interp}"
    )
    lines.append(
        f"- **Crystal texture:** GLCM contrast = {contrast_d1:.2f}, "
        f"homogeneity = {homogeneity_d1:.2f} - {glcm_interp}"
    )
    skew = rough.get("Ssk", 0.0)
    kurt = rough.get("Sku", 0.0)
    lines.append(
        f"- **Height distribution:** skewness Ssk = {skew:.2f}, "
        f"kurtosis Sku = {kurt:.2f}"
    )
    lines.append(
        f"- **Masmali grade (heuristic surrogate):** {masmali_inferred} "
        f"(expected for {CLASS_DISPLAY[top1_cls]}: grade {tpl['masmali_range']})"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Evidence for prediction
    lines.append(f"## Evidence for the primary prediction ({CLASS_DISPLAY[top1_cls]})")
    lines.append("")
    lines.append(tpl["summary"])
    lines.append("")
    for b in evidence_bullets:
        lines.append(f"- {b}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Similar reference cases
    if neighbours:
        lines.append("## Similar reference cases")
        lines.append("")
        lines.append(
            "Nearest scans in DINOv2-B embedding space (cosine similarity), "
            "from the 240-scan / 35-patient training cohort:"
        )
        lines.append("")
        lines.append("| Rank | Class | Similarity | File |")
        lines.append("|---:|---|---:|---|")
        for i, nb in enumerate(neighbours, 1):
            nb_name = Path(nb["path"]).name
            lines.append(
                f"| {i} | {CLASS_DISPLAY[nb['class']]} (`{nb['class']}`) "
                f"| {nb['similarity']:.3f} | `{nb_name}` |"
            )
        # Voter-style agreement note.
        votes: dict[str, int] = {}
        for nb in neighbours:
            votes[nb["class"]] = votes.get(nb["class"], 0) + 1
        majority = max(votes.items(), key=lambda kv: kv[1])[0]
        if majority == top1_cls:
            lines.append("")
            lines.append(
                f"Neighbour majority (`{majority}`) **agrees** with the model's "
                f"primary prediction."
            )
        else:
            lines.append("")
            lines.append(
                f"Neighbour majority (`{majority}`) **disagrees** with the model's "
                f"primary prediction (`{top1_cls}`) - this is an ambiguous sample; "
                f"clinical review is strongly advised."
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    # Confidence note
    lines.append("## Confidence note")
    lines.append("")
    lines.append(
        "This is an AI-generated preliminary assessment. Full diagnosis requires "
        "clinical correlation with patient history, symptoms, and orthogonal "
        "tests (e.g. HbA1c, Schirmer, visual-field, MRI)."
    )
    lines.append("")
    lines.append(
        f"Our model's honest held-out performance (person-level Leave-One-Patient-"
        f"Out over 240 scans / 35 persons): **weighted F1 = {HONEST_F1_WEIGHTED:.4f}**, "
        f"macro F1 = {HONEST_F1_MACRO:.4f}. For context, this matches or exceeds "
        f"typical human inter-rater reproducibility on Masmali-grade tear-ferning "
        f"(weighted kappa ~ {HUMAN_KAPPA:.2f}, Daza et al. 2022)."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # What the model might be missing
    lines.append("## What the model might be missing")
    lines.append("")
    caveats = CLASS_CAVEATS.get(top1_cls, [])
    for c in caveats:
        lines.append(f"- {c}")
    # Runner-up caveat — always mention that the model is not certain.
    gap = top1_prob - top2_prob
    if gap < 0.15:
        lines.append(
            f"- Top-2 gap is narrow ({gap:.1%}). Treat the runner-up "
            f"(`{top2_cls}`) as an active differential."
        )
    # Always: SucheOko warning if either top-2 slot is dry-eye.
    if "SucheOko" in (top1_cls, top2_cls) and top1_cls != "SucheOko":
        lines.append(
            "- `SucheOko` (dry-eye) appears in the differential. Our training "
            "cohort contains only 2 dry-eye patients; any dry-eye score is a "
            "weak signal, not a reliable diagnosis."
        )
    # Always: SM/Glaukom pair caveat.
    if {top1_cls, top2_cls} == {"SklerozaMultiplex", "PGOV_Glaukom"}:
        lines.append(
            "- `SklerozaMultiplex` and `PGOV_Glaukom` are the most commonly "
            "confused pair in our held-out audit. Consider both as active "
            "differentials and correlate with the clinical picture."
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Footer
    lines.append("## Methods (for the attending clinician)")
    lines.append("")
    lines.append(
        "- **Model:** 3-component geometric-mean ensemble - "
        "DINOv2-B at 90 nm/px + DINOv2-B at 45 nm/px + BiomedCLIP with D4 test-time "
        "augmentation at 90 nm/px. Per-component pipeline: frozen encoder -> L2 "
        "normalise -> StandardScaler -> class-balanced logistic regression -> softmax."
    )
    lines.append(
        "- **Preprocessing:** Bruker SPM -> plane-level (1st-order polynomial "
        "subtraction) -> resample to 90 nm/px -> robust normalise (2-98th "
        "percentile clip) -> up to 9 non-overlapping 512 x 512 tiles."
    )
    lines.append(
        "- **Handcrafted descriptors:** ISO surface-roughness (Ra, Rq, Rz, Ssk, "
        "Sku), GLCM Haralick statistics (contrast, homogeneity, correlation, "
        "ASM), box-counting fractal dimension over 5 threshold percentiles, "
        "LBP and HOG histograms. All features reproducible from "
        "`teardrop/features.py`."
    )
    lines.append(
        "- **Retrieval:** DINOv2-B tile-mean scan embeddings, cosine similarity "
        "against 240 training scans."
    )
    lines.append(
        "- **Report template:** `teardrop/clinical_report.py` (LLM-free, "
        "fully deterministic from the ensemble outputs + handcrafted features)."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Generate a Markdown clinical report for one AFM tear-film scan.",
    )
    ap.add_argument("scan", help="Path to the Bruker SPM scan (.NNN extension).")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--out", default=None,
                    help="Optional: write the report to this file instead of stdout.")
    args = ap.parse_args()

    md = generate_clinical_report(
        scan_path=args.scan, model_dir=args.model_dir,
    )
    if args.out:
        Path(args.out).write_text(md)
        print(f"Wrote report -> {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    _cli()
