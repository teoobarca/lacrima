"""LLM-reasoning classification layer.

Takes quantitative AFM tear-ferning features + domain knowledge, formats them
as a medical case summary, sends to the Claude API, and parses back a JSON
object with per-class probabilities + human-readable reasoning.

This is the INTERPRETABILITY layer — it is not expected to beat the DINOv2
embedding baseline on pure F1, but it produces text rationales that a
traditional classifier cannot.

Functions
---------
features_to_prompt(feats, domain_context) -> str
classify_with_llm(feats, model, temperature) -> dict  (class_probs / reasoning / key_features_used)

Environment
-----------
Requires ``ANTHROPIC_API_KEY``. If unset, :func:`classify_with_llm` raises
RuntimeError — callers are expected to handle it gracefully and abort.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

import anthropic

# -----------------------------------------------------------------------
# Class list + domain knowledge
# -----------------------------------------------------------------------

CLASSES = ["ZdraviLudia", "Diabetes", "PGOV_Glaukom", "SklerozaMultiplex", "SucheOko"]

DOMAIN_CONTEXT = """\
You are a clinician-scientist specialising in tear-film biomarkers, analysing
Atomic Force Microscopy (AFM) height maps of dried tear-ferning patterns. You
classify each sample into one of five categories. The expected tear-ferning
morphology for each class (drawn from Masmali grading and published ocular
biomarker literature) is:

1. ZdraviLudia  (healthy)
   - Dense, highly-branched dendritic fern (Masmali grade I-II)
   - Low surface roughness (Sa/Sq modest)
   - Fractal dimension typically 1.70 - 1.85
   - Moderate GLCM homogeneity, moderate correlation
   - LBP histogram balanced between uniform and edge patterns

2. Diabetes
   - Thicker dendritic branches, higher packing density
   - Elevated tear osmolarity -> denser lattice, more "solid" regions
   - Higher Sa and Sq than healthy (coarser surface)
   - GLCM contrast elevated; dissimilarity elevated
   - Skewness (Ssk) often shifted positive (taller peaks)

3. PGOV_Glaukom (primary open-angle glaucoma)
   - Granular structure, scattered particles instead of classic dendrites
   - MMP-9 protease activity degrades the glycoprotein matrix
   - Shorter, thicker branches; fewer end-points
   - GLCM correlation LOW (locally chaotic texture)
   - Fractal dimension lower and noisier than healthy

4. SklerozaMultiplex (multiple sclerosis)
   - HETEROGENEOUS morphology within-class (protein/lipid alteration)
   - Mixed crystal morphologies — coarse rods OR fine granules in same sample
   - High intra-sample variance in GLCM and LBP
   - Fractal D variable; roughness variable
   - Often confused visually with PGOV_Glaukom

5. SucheOko (dry eye disease)
   - Fragmented, SPARSE network (Masmali grade III-IV)
   - Lower branching, more amorphous / empty regions
   - Fractal D DEPRESSED (typically < 1.65)
   - Lower roughness in fern regions but high variance overall
   - LBP histogram skewed toward flat/uniform bins
"""

# The 20 most discriminative features for compact prompting. Chosen from
# GLCM / roughness / fractal / LBP families so each disease has something
# to latch onto.
KEY_FEATURES = [
    "Sa", "Sq", "Ssk", "Sku",
    "glcm_contrast_d1_mean", "glcm_contrast_d5_mean",
    "glcm_homogeneity_d1_mean", "glcm_homogeneity_d5_mean",
    "glcm_correlation_d1_mean", "glcm_correlation_d5_mean",
    "glcm_ASM_d1_mean", "glcm_energy_d1_mean",
    "glcm_dissimilarity_d3_mean",
    "fractal_D_mean", "fractal_D_std",
    "lbp_0", "lbp_10", "lbp_25",
    "hog_mean", "hog_std",
]

# -----------------------------------------------------------------------
# Simple rate-limiting (bounded concurrency + small floor between calls)
# -----------------------------------------------------------------------

_RATE_LOCK = threading.Lock()
_LAST_CALL_AT = [0.0]
_MIN_INTERVAL_S = 0.15  # ~6-7 rps ceiling; Haiku/Sonnet handle this trivially


def _throttle() -> None:
    """Ensure at least _MIN_INTERVAL_S between consecutive calls (thread-safe)."""
    with _RATE_LOCK:
        now = time.time()
        wait = _MIN_INTERVAL_S - (now - _LAST_CALL_AT[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL_AT[0] = time.time()


# -----------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------

def _fmt_num(v: Any) -> str:
    """Format numeric feature for the prompt — 4 sig figs."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(f) >= 1e4 or (0 < abs(f) < 1e-3):
        return f"{f:.3e}"
    return f"{f:.4g}"


def features_to_prompt(feats: dict, domain_context: str = DOMAIN_CONTEXT) -> str:
    """Build the user-side prompt with the 20-feature case summary + instructions."""
    lines = ["## Quantitative features (AFM tear-ferning scan)"]
    for k in KEY_FEATURES:
        if k in feats:
            lines.append(f"- {k}: {_fmt_num(feats[k])}")

    case = "\n".join(lines)
    classes_json_example = ", ".join(f'"{c}": 0.0' for c in CLASSES)

    return f"""{domain_context}

---

{case}

---

## Your task

Given the above quantitative features for a single AFM tear-ferning scan,
classify it into one of the five classes.

Respond **strictly** with a single JSON object (no prose, no markdown fence,
no leading/trailing text) with the following exact shape:

{{
  "class_probs": {{{classes_json_example}}},
  "reasoning": "<2-4 sentences citing specific feature values that drove the decision>",
  "key_features_used": ["<feature_name_1>", "<feature_name_2>", ...]
}}

Requirements:
- class_probs must contain all 5 keys exactly as shown (case-sensitive).
- Probabilities must be non-negative and sum to 1.0 (+/- 0.01).
- "reasoning" must cite at least 2 specific feature values from the case.
- "key_features_used" lists the feature names you weighed most (<= 5 items).
"""


# -----------------------------------------------------------------------
# LLM call + response parsing
# -----------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a careful clinical-research scientist. Respond only with the "
    "single JSON object requested — no prose, no markdown code fence, no "
    "apologies. Valid JSON only."
)


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of the model response, tolerating code fences."""
    text = text.strip()
    # strip ```json ... ``` fence if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # find first { ... last matching }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def _normalize_probs(probs: dict) -> dict:
    """Fill any missing class with 0 and renormalise to sum=1.0."""
    out = {c: float(probs.get(c, 0.0)) for c in CLASSES}
    s = sum(out.values())
    if s <= 0:
        # fall back to uniform
        return {c: 1.0 / len(CLASSES) for c in CLASSES}
    return {c: v / s for c, v in out.items()}


def _get_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running the "
            "LLM-reasoning classifier."
        )
    return anthropic.Anthropic(api_key=key)


def classify_with_llm(
    feats: dict,
    model: str = "claude-haiku-4-5",
    temperature: float = 0.0,
    max_retries: int = 3,
    max_tokens: int = 1024,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """Classify one scan via the Claude API.

    Parameters
    ----------
    feats : dict
        Feature dict (e.g. one row of cache/features_handcrafted.parquet).
    model : str
        Anthropic model ID. Default claude-haiku-4-5 (cheap, fast).
    temperature : float
        Sampling temperature. Default 0.0 for determinism.
    max_retries : int
        Retries on transient errors or malformed JSON.
    max_tokens : int
        Output token cap — 1024 is plenty for the required JSON.
    client : anthropic.Anthropic, optional
        Reuse an existing client (avoids re-reading env var).

    Returns
    -------
    dict with keys: class_probs, reasoning, key_features_used,
    predicted_class, raw_response, usage, model, latency_s.
    """
    if client is None:
        client = _get_client()

    prompt = features_to_prompt(feats)
    last_err: Exception | None = None

    for attempt in range(max_retries):
        _throttle()
        t0 = time.time()
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.RateLimitError as e:
            last_err = e
            # exponential backoff
            wait = 2 ** attempt
            time.sleep(wait)
            continue
        except anthropic.APIStatusError as e:
            last_err = e
            if e.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise

        latency = time.time() - t0
        text = "".join(b.text for b in resp.content if b.type == "text")

        try:
            parsed = _extract_json(text)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            continue

        probs = _normalize_probs(parsed.get("class_probs", {}))
        predicted = max(probs, key=probs.get)

        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", 0),
            "output_tokens": getattr(resp.usage, "output_tokens", 0),
        }

        return {
            "class_probs": probs,
            "reasoning": str(parsed.get("reasoning", "")),
            "key_features_used": list(parsed.get("key_features_used", [])),
            "predicted_class": predicted,
            "raw_response": text,
            "usage": usage,
            "model": model,
            "latency_s": latency,
        }

    raise RuntimeError(
        f"classify_with_llm failed after {max_retries} retries: {last_err}"
    )


# Haiku 4.5 pricing per Anthropic docs (cached 2026-04): $1/MTok input, $5/MTok output
HAIKU_INPUT_USD_PER_MTOK = 1.00
HAIKU_OUTPUT_USD_PER_MTOK = 5.00

# Sonnet 4.6 pricing (cached 2026-04): $3/MTok input, $15/MTok output
SONNET_INPUT_USD_PER_MTOK = 3.00
SONNET_OUTPUT_USD_PER_MTOK = 15.00


def estimate_cost_usd(usage: dict, model: str) -> float:
    """Per-call USD estimate from usage dict."""
    if "haiku" in model:
        in_rate = HAIKU_INPUT_USD_PER_MTOK
        out_rate = HAIKU_OUTPUT_USD_PER_MTOK
    elif "sonnet" in model:
        in_rate = SONNET_INPUT_USD_PER_MTOK
        out_rate = SONNET_OUTPUT_USD_PER_MTOK
    else:
        # conservative default
        in_rate = SONNET_INPUT_USD_PER_MTOK
        out_rate = SONNET_OUTPUT_USD_PER_MTOK
    return (
        usage.get("input_tokens", 0) * in_rate / 1_000_000
        + usage.get("output_tokens", 0) * out_rate / 1_000_000
    )
