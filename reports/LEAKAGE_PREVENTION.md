# VLM Filename-Leakage Prevention — Runtime-Enforced

**Status:** shipped 2026-04-18.
**Module:** `teardrop/safe_paths.py`.
**Tests:** `tests/test_no_leak.py` (12 tests, all green).

---

## TL;DR

We got bitten **twice in one day** by the same bug: VLM driver scripts
wrote tile / collage images to paths like
`cache/vlm_tiles/Diabetes__37_DM.png`, then passed that path string to
Claude via `claude -p`. Claude reads the class name directly off the
filename → honest F1 collapses when the filename is obfuscated.

The fix is two-part:

1. **A single source of truth** (`teardrop.safe_paths.safe_tile_path`) for
   every VLM image path — deterministic, class-free, person-free,
   raw-name-free filenames of the form
   `cache/vlm_safe/<subdir>/scan_<idx>.png`.
2. **A runtime guard** (`assert_prompt_safe`) that every VLM script calls
   immediately before `subprocess.run(["claude", "-p", …, prompt])`. It
   raises `PromptLeakError` if the prompt contains any of the three
   known leak patterns.

Both are lint-checked by `tests/test_no_leak.py`. A future script that
silently rolls its own path scheme will fail CI before it ships.

---

## The two incidents

| Wave | Script | Caught F1 | Honest F1 | Delta |
|---|---|---|---|---|
| **9** | `scripts/vlm_direct_classify.py` (initial) | 0.88 | 0.28 | –0.60 |
| **14** | `scripts/vlm_few_shot*.py` (all variants) | 0.8873 | 0.26 | –0.62 |

### Wave 9 root cause

`cache/vlm_tiles/{cls}__{stem}.png` was rendered by
`vlm_direct_classify.py`. The class-prefixed filename was then
interpolated into the prompt (`The image at {img_path} shows …`).

### Wave 14 root cause

All `vlm_few_shot*.py` variants composed a labeled anchor collage with
filename `cache/vlm_few_shot_collages/{s.cls}__…png`. The collage itself
labels the anchors by class (that is intended — the anchors ARE labeled
references) but the **query** image tile and the **collage file path**
also contained the query's true class, so the model could just read the
filename and skip the vision task entirely.

---

## Module API (`teardrop/safe_paths.py`)

```python
from teardrop.safe_paths import (
    SAFE_ROOT,                # cache/vlm_safe/
    safe_tile_path,           # obfuscated PNG path
    safe_manifest_path,       # sibling manifest.json (labels live here)
    assert_prompt_safe,       # runtime guard; raises PromptLeakError
    scan_file_for_leaks,      # utility: does this path string leak?
    PromptLeakError,
    RAW_NAME_FRAGMENTS,
)
```

### `safe_tile_path(index: int, subdir: str) -> Path`

Returns `<SAFE_ROOT>/<subdir>/scan_<index:04d>.png`. The returned path
contains **no class, no person, no raw name**.

```python
p = safe_tile_path(42, subdir="direct")
# -> cache/vlm_safe/direct/scan_0042.png
```

### `safe_manifest_path(subdir: str) -> Path`

Returns `<SAFE_ROOT>/<subdir>/manifest.json` (creating parents). This is
the **only** place labels should be stored. The manifest is never passed
into a prompt.

### `assert_prompt_safe(prompt: str, extra_forbidden: Iterable[str] = ()) -> None`

Three-layer runtime check (raises `PromptLeakError` on first hit):

1. **Path-context class names.** Flags class names that appear adjacent
   to a path separator (`/Diabetes/`), a double-underscore
   (`Diabetes__37`) or at string boundaries. Prose mentions like
   “Possible classes: ZdraviLudia, Diabetes, …” are allowed — they’re
   mandatory for the prompt to describe the taxonomy.
2. **Raw-filename fragments.** Rejects `_DM`, `-SM-`, `_PGOV`, `_Zdrav`
   and friends — but only when they are NOT part of a legitimate class
   name spelling (so `PGOV_Glaukom` in prose is fine, `21_LV_PGOV.104`
   is not).
3. **Legacy leaky directories.** `vlm_tiles/` (non-honest),
   `vlm_few_shot_collages/`, `vlm_few_shot_k3_collages/`,
   `vlm_zero_shot_opus_tiles/`, `vlm_variant2_anchors_honest/`.
4. **Caller-supplied extras.** Pass `extra_forbidden=["raw_name_x"]` to
   belt-and-brace an unusual case.

### `scan_file_for_leaks(path: Path, forbidden_words=None) -> list[str]`

Returns forbidden substrings found in `str(path)`. Defaults to
`CLASSES + RAW_NAME_FRAGMENTS`. Useful as a sanity check right after
rendering.

---

## Retrofit summary

Every VLM script now uses `safe_tile_path` / `SAFE_ROOT` and calls
`assert_prompt_safe` before `subprocess.run(["claude", "-p", …])`:

- `scripts/vlm_direct_classify.py` — tiles now under
  `cache/vlm_safe/direct/`, sha1-hashed.
- `scripts/vlm_honest_parallel.py` — already safe
  (`cache/vlm_tiles_honest/`); added `assert_prompt_safe`.
- `scripts/vlm_few_shot.py` — tiles + collages moved to
  `cache/vlm_safe/few_shot/`. Anchor visible IDs are sha1 hashes so the
  raw filename can’t be OCR’d off the collage.
- `scripts/vlm_few_shot_sonnet.py` — ditto, `cache/vlm_safe/few_shot_sonnet/`.
- `scripts/vlm_few_shot_sonnet_full_240.py` — imports the safe helpers.
- `scripts/vlm_few_shot_k3.py` — tiles + collages under
  `cache/vlm_safe/few_shot_k3/`.
- `scripts/vlm_few_shot_k2_extend.py` — same pattern.
- `scripts/vlm_few_shot_opus.py` — same pattern.
- `scripts/vlm_few_shot_full_240.py` — same pattern.
- `scripts/vlm_zero_shot_opus.py` — tiles under `cache/vlm_safe/zero_shot_opus/`.
- `scripts/vlm_prompt_variants.py` — tile/anchor dir moved to
  `cache/vlm_safe/prompt_variants/`. Anchor + query tiles both
  sha1-named.
- `scripts/vlm_self_consistency.py` — uses safe helpers from
  `vlm_few_shot`; adds guard.
- `scripts/vlm_sanity_isolated.py` — already uses `mktemp /
  scan_XXXX.png`; added guard.
- `scripts/vlm_model_comparison.py` — uses `SAFE_ROOT / "direct"`,
  sha1-named tiles.
- `scripts/expert_council.py` — uses `cache/vlm_tiles_honest/`; added
  guard.
- `scripts/llm_gated_reasoner.py` — text-only LLM prompt; added guard as
  belt-and-braces.

---

## Guidelines for new VLM scripts

1. **Never** write an image that is passed to `claude -p` with
   `f"{cls}__…"`, `f"{sample.cls}__…"`, or any other class / person /
   raw-name interpolation.
2. **Always** build the image path via
   `teardrop.safe_paths.safe_tile_path(index, subdir)` or a sha1 hash
   under `SAFE_ROOT / "<your_subdir>"`.
3. **Store labels in a manifest** at
   `safe_manifest_path(subdir)`. Never echo a manifest entry into a
   prompt.
4. **Call `assert_prompt_safe(prompt)` immediately before** every
   `subprocess.run(["claude", "-p", …, prompt])`. No exceptions.
5. **Scrub visual text too.** If you render labels onto an image tile /
   collage (anchor IDs, filenames), hash them first — OCR is a real
   leak channel.
6. Add a test case to `tests/test_no_leak.py` for any new prompt
   renderer so it's continuously monitored.

---

## Pitch framing

> “We were bitten twice by the same filename-leakage bug in our VLM
> pipelines — honest F1 collapsed from 0.88 to 0.26 once we obfuscated
> the filenames. Rather than rely on vigilance, we shipped
> **runtime-enforced leakage prevention**: every `claude -p` call routes
> through `assert_prompt_safe`, which raises on class-in-path,
> raw-filename-fragment, or legacy-directory references. A lint test
> (`tests/test_no_leak.py`) sweeps the whole scripts tree on every
> change. Future contributors can't reintroduce the bug without the
> tests failing first.”
