"""Runtime-enforced leakage prevention for VLM pipelines.

Background
----------
We were bitten TWICE (2026-04-18) by the same bug: VLM scripts wrote image
files with paths like ``cache/vlm_tiles/Diabetes__37_DM.png`` or
``cache/vlm_few_shot_collages/<CLASS>__<scan>.png`` and then passed that
path string to Claude via the ``claude -p`` CLI. Claude trivially reads the
class name from the filename → inflated F1 (Wave 9: 88 % → 28 % honest,
Wave 14: 88.73 % → 26 % honest).

This module is a single-source-of-truth for VLM-safe paths and a runtime
``assert_prompt_safe`` guard that every VLM script MUST call immediately
before invoking ``claude -p``.

API
---
``safe_tile_path(index, subdir)``
    Deterministic, obfuscated PNG path of the form
    ``<repo>/cache/vlm_safe/<subdir>/scan_0042.png`` — no class, no person,
    no raw filename.

``safe_manifest_path(subdir)``
    Sibling ``manifest.json`` inside the same subdir. Store the mapping
    ``scan_0042 → {true_class, person, raw_path}`` here. NEVER pass the
    manifest contents into a prompt.

``PromptLeakError``
    Runtime exception raised by ``assert_prompt_safe`` on detected leak.

``assert_prompt_safe(prompt, extra_forbidden=())``
    Inspect a prompt (the exact string passed to ``claude -p``) for leak
    patterns. Raises :class:`PromptLeakError` on the first hit.

    The check is **context-aware**: class names appearing as part of a
    *path-like* token (``/Diabetes/``, ``Diabetes__37.png``, etc.) are
    forbidden, but class names appearing in generic prose (``"Possible
    classes: Diabetes, …"``) are allowed. This is what every current VLM
    prompt relies on — the taxonomy must be described to the model.

``scan_file_for_leaks(path, forbidden_words)``
    Utility: check that a rendered PNG path does not contain leak words.
    Useful as a belt-and-braces assertion right after rendering.

Design rules for new VLM scripts
--------------------------------
1. Always build tile / collage paths via :func:`safe_tile_path`.
2. Put labels / person ids / raw names in :func:`safe_manifest_path`, not
   in filenames or prompts.
3. Call :func:`assert_prompt_safe` on the exact prompt string immediately
   before ``subprocess.run(["claude", "-p", ..., prompt])``.
4. Never ``f"{cls}__…"`` a filename used as a VLM input.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from teardrop.data import CLASSES

__all__ = [
    "safe_tile_path",
    "safe_manifest_path",
    "PromptLeakError",
    "assert_prompt_safe",
    "scan_file_for_leaks",
    "SAFE_ROOT",
    "RAW_NAME_FRAGMENTS",
]


# Root for all obfuscated VLM inputs. Scripts may pass their own subdir
# (e.g. ``"direct"``, ``"few_shot"``, ``"council"``).
SAFE_ROOT = Path(__file__).resolve().parent.parent / "cache" / "vlm_safe"


# Substring fragments frequently found in raw TRAIN_SET filenames that
# leak class information by themselves. Collected from the real dataset:
#   * Diabetes:        ``DM_01.03.2024_LO.37``, ``37_DM.038``
#   * SM:              ``20_3-SM-LV-18.016``, ``1-SM-PM-18.002``
#   * PGOV_Glaukom:    ``21_LV_PGOV.104``
#   * ZdraviLudia:     ``2L-Zdrav.001``, ``8P_Zdrav.015``
#   * SucheOko:        ``SO_12.09.2024.081`` (rare — "SO" alone is too
#                      ambiguous to blacklist here)
RAW_NAME_FRAGMENTS: tuple[str, ...] = (
    "_DM",
    "-DM",
    "_SM-",
    "-SM-",
    "_Sm",
    "-Sm",
    "_Gla",
    "-Gla",
    "_PGOV",
    "-PGOV",
    "_Zdrav",
    "-Zdrav",
    "_SucheOko",
    "_SO_",
)


# Path-like tokens we treat as leaky when a class name appears inside them.
# Anything between a path separator / double-underscore / dot and a class
# name is considered a path context.
_PATH_CONTEXT_PREFIX = r"(?:[/\\]|__|^)"
_PATH_CONTEXT_SUFFIX = r"(?:[/\\]|__|\.|$)"


class PromptLeakError(RuntimeError):
    """Raised by :func:`assert_prompt_safe` when a leak pattern is found."""


def safe_tile_path(index: int, subdir: str = "default") -> Path:
    """Return an obfuscated PNG path: ``<SAFE_ROOT>/<subdir>/scan_<idx>.png``.

    The filename contains no class, person, or raw-name info — only the
    caller-supplied ``index``.
    """
    d = SAFE_ROOT / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d / f"scan_{int(index):04d}.png"


def safe_manifest_path(subdir: str) -> Path:
    """Return ``<SAFE_ROOT>/<subdir>/manifest.json`` (and ensure parent exists).

    The manifest is the ONLY place where labels live. It must never be
    embedded in a prompt.
    """
    d = SAFE_ROOT / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d / "manifest.json"


def _snippet(prompt: str, needle: str, radius: int = 40) -> str:
    idx = prompt.find(needle)
    if idx < 0:
        return "<not found>"
    lo = max(0, idx - radius)
    hi = min(len(prompt), idx + len(needle) + radius)
    return prompt[lo:hi].replace("\n", " ")


def _class_leak_in_path_context(prompt: str) -> tuple[str, str] | None:
    """Return (class_name, snippet) if any class appears in path-context.

    "Path context" means the class name is preceded or followed by a path
    separator (``/`` or ``\\``), a double underscore, or it is at the
    start / end of the string. Generic prose mentions are allowed.
    """
    for cls in CLASSES:
        # Fast negative check
        if cls not in prompt:
            continue
        pattern = re.compile(
            _PATH_CONTEXT_PREFIX + re.escape(cls) + _PATH_CONTEXT_SUFFIX
        )
        m = pattern.search(prompt)
        if m:
            return cls, _snippet(prompt, m.group(0))
    return None


def _fragment_hits_outside_class_names(prompt: str, frag: str) -> list[int]:
    """Indices where ``frag`` appears NOT as part of a class name spelling.

    Every class name in :data:`CLASSES` is temporarily masked to ``*``s
    before the search, so ``_PGOV`` inside ``PGOV_Glaukom`` doesn't count
    as a leak. A lone ``21_LV_PGOV.104`` still does.
    """
    masked = prompt
    for cls in CLASSES:
        masked = masked.replace(cls, "*" * len(cls))
    hits: list[int] = []
    start = 0
    while True:
        i = masked.find(frag, start)
        if i < 0:
            break
        hits.append(i)
        start = i + 1
    return hits


def assert_prompt_safe(prompt: str, extra_forbidden: Iterable[str] = ()) -> None:
    """Raise :class:`PromptLeakError` if ``prompt`` looks leaky.

    Checks (in order):

    1. Class names appearing in a path-like context (``/Diabetes/``,
       ``Diabetes__37.png``, trailing ``…Diabetes``).
    2. Raw-filename fragments (``_DM``, ``-SM-``, ``_PGOV``, ``_Zdrav`` …)
       that are NOT part of a valid class-name spelling. ``PGOV_Glaukom``
       in prose is fine; ``21_LV_PGOV`` is not.
    3. Reference to the leaky legacy directories ``vlm_tiles/``,
       ``vlm_few_shot_collages/``, ``vlm_few_shot_k3_collages/``, or
       ``vlm_zero_shot_opus_tiles/``. Safe scripts must use ``vlm_safe/``
       or ``vlm_tiles_honest/``.
    4. Any ``extra_forbidden`` word substring.

    Call this BEFORE every ``subprocess.run(["claude", "-p", …, prompt])``.
    """
    # 1) class names in path context
    hit = _class_leak_in_path_context(prompt)
    if hit is not None:
        cls, snip = hit
        raise PromptLeakError(
            f"PROMPT LEAK: class '{cls}' appears in a PATH-LIKE context. "
            f"Rename the file via safe_tile_path(). Snippet: ...{snip}..."
        )

    # 2) raw-filename fragments, excluding legitimate class-name prose
    for frag in RAW_NAME_FRAGMENTS:
        indices = _fragment_hits_outside_class_names(prompt, frag)
        if indices:
            idx = indices[0]
            lo = max(0, idx - 40)
            hi = min(len(prompt), idx + len(frag) + 40)
            snip = prompt[lo:hi].replace("\n", " ")
            raise PromptLeakError(
                f"PROMPT LEAK: raw-filename fragment '{frag}' in prompt. "
                f"Use teardrop.safe_paths.safe_tile_path() to obfuscate. "
                f"Snippet: ...{snip}..."
            )

    # 3) legacy leaky directories
    for leaky_dir in (
        "vlm_tiles/",
        "vlm_tiles\\",
        "vlm_few_shot_collages/",
        "vlm_few_shot_collages\\",
        "vlm_few_shot_k3_collages/",
        "vlm_few_shot_k3_collages\\",
        "vlm_zero_shot_opus_tiles/",
        "vlm_variant2_anchors_honest/",
    ):
        if leaky_dir in prompt:
            raise PromptLeakError(
                f"PROMPT LEAK: legacy leaky directory '{leaky_dir}' in prompt. "
                f"Use cache/vlm_safe/<subdir>/ via safe_tile_path(). "
                f"Snippet: ...{_snippet(prompt, leaky_dir)}..."
            )

    # 4) caller-supplied extras
    for word in extra_forbidden:
        if word and word in prompt:
            raise PromptLeakError(
                f"PROMPT LEAK: forbidden word '{word}' in prompt. "
                f"Snippet: ...{_snippet(prompt, word)}..."
            )


def scan_file_for_leaks(path: Path, forbidden_words: Iterable[str] | None = None) -> list[str]:
    """Return the list of forbidden substrings found in ``str(path)``.

    ``forbidden_words`` defaults to ``CLASSES + RAW_NAME_FRAGMENTS``. Empty
    list = no leaks.
    """
    if forbidden_words is None:
        forbidden_words = list(CLASSES) + list(RAW_NAME_FRAGMENTS)
    s = str(path)
    return [w for w in forbidden_words if w and w in s]
