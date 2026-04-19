"""Pre-flight sanity tests for VLM leakage prevention.

Run:
    .venv/bin/python tests/test_no_leak.py
    # or via pytest
    .venv/bin/python -m pytest tests/test_no_leak.py -v

Suite
-----
1. ``test_safe_paths_imports``
   Smoke test that :mod:`teardrop.safe_paths` exposes the expected API.
2. ``test_assert_prompt_safe_accepts_clean``
   Legitimate prompts (generic class-taxonomy prose + obfuscated tile
   paths) pass.
3. ``test_assert_prompt_safe_rejects_leaky``
   Every known leak pattern from the two 2026-04-18 incidents is
   rejected with :class:`PromptLeakError`.
4. ``test_all_vlm_script_prompts_safe``
   Renders a sample prompt from every VLM script's actual
   ``PROMPT_TEMPLATE`` (or equivalent) and asserts it passes
   :func:`assert_prompt_safe`.
5. ``test_no_leaky_patterns_in_scripts``
   Static lint: grep every ``scripts/vlm_*.py`` and
   ``scripts/expert_council.py`` for known leaky idioms
   (``f"{cls}__…"``, writes into ``cache/vlm_tiles/`` etc.). Fails if
   any survive.
"""
from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from teardrop.data import CLASSES  # noqa: E402
from teardrop.safe_paths import (  # noqa: E402
    PromptLeakError,
    SAFE_ROOT,
    assert_prompt_safe,
    safe_manifest_path,
    safe_tile_path,
    scan_file_for_leaks,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPTS_DIR = REPO / "scripts"

# Scripts that MUST be free of leaky idioms. These all issue `claude -p`
# subprocess calls with an image path or a feature prompt.
LEAK_CHECK_SCRIPTS = sorted(
    [
        SCRIPTS_DIR / name
        for name in (
            "vlm_direct_classify.py",
            "vlm_honest_parallel.py",
            "vlm_few_shot.py",
            "vlm_few_shot_sonnet.py",
            "vlm_few_shot_sonnet_full_240.py",
            "vlm_few_shot_k3.py",
            "vlm_few_shot_k2_extend.py",
            "vlm_few_shot_opus.py",
            "vlm_few_shot_full_240.py",
            "vlm_zero_shot_opus.py",
            "vlm_prompt_variants.py",
            "vlm_self_consistency.py",
            "vlm_sanity_isolated.py",
            "vlm_model_comparison.py",
            "expert_council.py",
            "llm_gated_reasoner.py",
        )
    ]
)

# Regex lints — these patterns, if present outside a comment string,
# indicate a leaky filename construction. They are tightly scoped so they
# do NOT fire on the safe alternatives we now use.
LEAKY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "class-prefixed filename: f\"{cls}__…\" / f\"{s.cls}__…\"",
        re.compile(r"f\"\{(?:[a-zA-Z_][\w\.]*\.)?cls\}__"),
    ),
    (
        "class-prefixed filename: f\"{sample.cls}__…\"",
        re.compile(r"f\"\{sample\.cls\}__"),
    ),
    (
        "direct write to cache/vlm_tiles/ (leaky legacy dir)",
        re.compile(r"(?<!vlm_safe/)cache/vlm_tiles/(?!honest)"),
    ),
    (
        "direct write to cache/vlm_few_shot_collages/ (leaky legacy dir)",
        re.compile(r"cache/vlm_few_shot_collages/"),
    ),
    (
        "direct write to cache/vlm_few_shot_k3_collages/ (leaky legacy dir)",
        re.compile(r"cache/vlm_few_shot_k3_collages/"),
    ),
    (
        "anchor_info with raw path name (visible-OCR leak)",
        re.compile(r"anchor_info\.append\(\(\s*cls\s*,\s*a_tile\s*,\s*a_sample\.raw_path\.name\s*\)"),
    ),
]


# ---------------------------------------------------------------------------
# Utilities: render sample prompts from each script
# ---------------------------------------------------------------------------


def _sample_safe_img_path(subdir: str = "test") -> Path:
    return safe_tile_path(0, subdir)


def _render_direct_classify_prompt() -> str:
    from scripts.vlm_direct_classify import PROMPT_TEMPLATE
    return PROMPT_TEMPLATE.format(img_path=str(_sample_safe_img_path()))


def _render_honest_parallel_prompt() -> str:
    from scripts.vlm_honest_parallel import PROMPT_TEMPLATE
    return PROMPT_TEMPLATE.format(img_path=str(_sample_safe_img_path()))


def _render_few_shot_prompt() -> str:
    from scripts.vlm_few_shot import PROMPT_TEMPLATE
    return PROMPT_TEMPLATE.format(img_path=str(_sample_safe_img_path()))


def _render_few_shot_sonnet_prompt() -> str:
    from scripts.vlm_few_shot_sonnet import PROMPT_TEMPLATE
    return PROMPT_TEMPLATE.format(img_path=str(_sample_safe_img_path()))


def _render_few_shot_k3_prompt() -> str:
    from scripts.vlm_few_shot_k3 import PROMPT_TEMPLATE
    return PROMPT_TEMPLATE.format(img_path=str(_sample_safe_img_path()))


def _render_zero_shot_opus_prompt() -> str:
    from scripts.vlm_zero_shot_opus import PROMPT_TEMPLATE
    return PROMPT_TEMPLATE.format(img_path=str(_sample_safe_img_path()))


def _render_prompt_variants_prompts() -> list[str]:
    from scripts.vlm_prompt_variants import (
        prompt_v1_minimal, prompt_v3_cot, prompt_v4_expert,
    )
    img = _sample_safe_img_path()
    return [prompt_v1_minimal(img), prompt_v3_cot(img), prompt_v4_expert(img)]


def _render_self_consistency_prompts() -> list[str]:
    from scripts.vlm_self_consistency import build_prompt, PROMPT_INTRO
    img = _sample_safe_img_path()
    return [build_prompt(v, img) for v in PROMPT_INTRO.keys()]


def _render_sanity_isolated_prompt() -> str:
    from scripts.vlm_sanity_isolated import PROMPT_TEMPLATE
    return PROMPT_TEMPLATE.format(img_path=str(_sample_safe_img_path()))


def _render_expert_council_prompt() -> str:
    from scripts.expert_council import JUDGE_PROMPT_TEMPLATE, CLASS_SIGNATURES
    return JUDGE_PROMPT_TEMPLATE.format(
        img_path=str(_sample_safe_img_path()),
        v4_line="Top-1: ZdraviLudia (p=0.70)",
        v4_probs="ZdraviLudia=0.7 Diabetes=0.1 PGOV_Glaukom=0.1 SklerozaMultiplex=0.05 SucheOko=0.05",
        knn_votes="ZdraviLudia:3 Diabetes:1 PGOV_Glaukom:1",
        knn_probs="ZdraviLudia=0.6 Diabetes=0.2 PGOV_Glaukom=0.2",
        knn_top1="ZdraviLudia",
        knn_top1_w=0.6,
        xgb_line="Top-1: ZdraviLudia (p=0.55)",
        xgb_probs="ZdraviLudia=0.55 Diabetes=0.15",
        fractal_D=1.78, fractal_D_std=0.05,
        Sq=8.2, Sa=6.0, Ssk=0.1,
        glcm_contrast=4.2, glcm_homog=0.42, glcm_corr=0.35,
        masmali_grade=1,
        hog_mean=0.21,
        lac_slope=-0.48,
        class_signatures=CLASS_SIGNATURES,
    )


def _render_llm_gated_reasoner_prompt() -> str:
    from scripts.llm_gated_reasoner import build_prompt
    from teardrop.llm_reason import DOMAIN_CONTEXT
    # fake features
    feats = {"fractal_D": 1.77, "Sq": 8.0, "glcm_contrast": 4.2, "hog_mean": 0.21}
    neighbours = {"ZdraviLudia": [0, 1], "Diabetes": [2]}
    feats_by_idx = {0: feats, 1: feats, 2: feats}
    return build_prompt(
        query_feats=feats,
        top1_cls="ZdraviLudia",
        top1_p=0.55,
        top2_cls="Diabetes",
        top2_p=0.30,
        neighbours=neighbours,
        feats_by_idx=feats_by_idx,
        domain_txt=DOMAIN_CONTEXT,
    )


SCRIPT_PROMPT_RENDERERS: list[tuple[str, callable]] = [
    ("vlm_direct_classify", _render_direct_classify_prompt),
    ("vlm_honest_parallel", _render_honest_parallel_prompt),
    ("vlm_few_shot", _render_few_shot_prompt),
    ("vlm_few_shot_sonnet", _render_few_shot_sonnet_prompt),
    ("vlm_few_shot_k3", _render_few_shot_k3_prompt),
    ("vlm_zero_shot_opus", _render_zero_shot_opus_prompt),
    ("vlm_sanity_isolated", _render_sanity_isolated_prompt),
    ("expert_council", _render_expert_council_prompt),
    ("llm_gated_reasoner", _render_llm_gated_reasoner_prompt),
    # prompt_variants + self_consistency expand to multiple prompts:
    ("vlm_prompt_variants[v1]", lambda: _render_prompt_variants_prompts()[0]),
    ("vlm_prompt_variants[v3]", lambda: _render_prompt_variants_prompts()[1]),
    ("vlm_prompt_variants[v4]", lambda: _render_prompt_variants_prompts()[2]),
    ("vlm_self_consistency[A]", lambda: _render_self_consistency_prompts()[0]),
    ("vlm_self_consistency[B]", lambda: _render_self_consistency_prompts()[1]),
    ("vlm_self_consistency[C]", lambda: _render_self_consistency_prompts()[2]),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class SafePathsAPITest(unittest.TestCase):
    def test_imports(self):
        self.assertTrue(callable(assert_prompt_safe))
        self.assertTrue(callable(safe_tile_path))
        self.assertTrue(callable(safe_manifest_path))
        self.assertTrue(issubclass(PromptLeakError, RuntimeError))
        p = safe_tile_path(7, "test_api")
        self.assertTrue(p.name == "scan_0007.png")
        self.assertIn("vlm_safe", str(p))
        self.assertNotIn("Diabetes", str(p))

    def test_scan_file_for_leaks(self):
        leaky = Path("/cache/vlm_tiles/Diabetes__37_DM.png")
        hits = scan_file_for_leaks(leaky)
        self.assertIn("Diabetes", hits)
        self.assertIn("_DM", hits)
        safe = safe_tile_path(0, "test_scan")
        self.assertEqual(scan_file_for_leaks(safe), [])


class AssertPromptSafeTest(unittest.TestCase):
    def test_accepts_clean_prompt(self):
        good = (
            "Possible classes: ZdraviLudia, Diabetes, PGOV_Glaukom, "
            "SklerozaMultiplex, SucheOko. Image at "
            f"{safe_tile_path(0, 'clean_test')}"
        )
        assert_prompt_safe(good)  # must not raise

    def test_rejects_Diabetes_path_context(self):
        bad = "read cache/vlm_tiles/Diabetes__37_DM.png and classify."
        with self.assertRaises(PromptLeakError):
            assert_prompt_safe(bad)

    def test_rejects_DM_fragment(self):
        bad = "Image at /tmp/foo_DM.png. Classify."
        with self.assertRaises(PromptLeakError):
            assert_prompt_safe(bad)

    def test_rejects_PGOV_fragment(self):
        bad = "Image at /tmp/21_LV_PGOV.104. Classify."
        with self.assertRaises(PromptLeakError):
            assert_prompt_safe(bad)

    def test_rejects_leaky_dir(self):
        bad = "Read cache/vlm_few_shot_collages/foo.png"
        with self.assertRaises(PromptLeakError):
            assert_prompt_safe(bad)

    def test_rejects_class_in_slash_context(self):
        bad = "Path /TRAIN_SET/Diabetes/37.png. Classify into one of the 5 classes."
        with self.assertRaises(PromptLeakError):
            assert_prompt_safe(bad)

    def test_extra_forbidden(self):
        with self.assertRaises(PromptLeakError):
            assert_prompt_safe("harmless prompt", extra_forbidden=["harmless"])


class AllScriptPromptsAreSafeTest(unittest.TestCase):
    def test_every_script_prompt_passes_assert_prompt_safe(self):
        failed: list[str] = []
        for name, renderer in SCRIPT_PROMPT_RENDERERS:
            try:
                prompt = renderer()
            except Exception as e:  # noqa: BLE001
                failed.append(f"{name}: RENDER FAILED with {type(e).__name__}: {e}")
                continue
            try:
                assert_prompt_safe(prompt)
            except PromptLeakError as e:
                failed.append(f"{name}: PROMPT LEAKED -> {e}")
        self.assertFalse(failed, "Prompt-safety failures:\n" + "\n".join(failed))


class ScriptStaticLintTest(unittest.TestCase):
    def test_no_leaky_patterns_in_any_script(self):
        hits: list[str] = []
        for path in LEAK_CHECK_SCRIPTS:
            if not path.exists():
                continue
            text = path.read_text()
            # Strip # comments and docstrings (very roughly) to reduce false
            # positives from examples in docstrings.
            try:
                tree = ast.parse(text)
                # Collect (lineno, col) spans of docstrings to skip.
                skip_lines: set[int] = set()
                for node in ast.walk(tree):
                    if (
                        isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    ):
                        ds = node.body[0]
                        if hasattr(ds, "end_lineno") and ds.end_lineno is not None:
                            for ln in range(ds.lineno, ds.end_lineno + 1):
                                skip_lines.add(ln)
            except SyntaxError:
                skip_lines = set()

            for lineno, line in enumerate(text.splitlines(), start=1):
                if lineno in skip_lines:
                    continue
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                for label, pat in LEAKY_PATTERNS:
                    if pat.search(line):
                        hits.append(f"{path.name}:{lineno}  [{label}]  {line.strip()[:160]}")
        self.assertFalse(hits, "Leaky patterns still present:\n" + "\n".join(hits))

    def test_every_vlm_script_calls_assert_prompt_safe(self):
        """Every script that builds a prompt + runs claude -p must import
        ``assert_prompt_safe`` AND call it. This is a belt-and-braces
        check against accidental removal.
        """
        # The PONG ping in expert_council.py / llm_gated_reasoner.py is a
        # fixed string, not a formatted prompt, so the assert is not
        # strictly required, but we still expect the import.
        required_imports = [
            "vlm_direct_classify.py",
            "vlm_honest_parallel.py",
            "vlm_few_shot.py",
            "vlm_few_shot_sonnet.py",
            "vlm_few_shot_k3.py",
            "vlm_zero_shot_opus.py",
            "vlm_prompt_variants.py",
            "vlm_self_consistency.py",
            "vlm_sanity_isolated.py",
            "expert_council.py",
            "llm_gated_reasoner.py",
        ]
        missing_import: list[str] = []
        missing_call: list[str] = []
        for name in required_imports:
            p = SCRIPTS_DIR / name
            if not p.exists():
                continue
            text = p.read_text()
            if "assert_prompt_safe" not in text:
                missing_import.append(name)
                continue
            # at least one call-site (not just the import line)
            call_sites = [
                ln for ln in text.splitlines()
                if "assert_prompt_safe(" in ln and "import" not in ln and "def " not in ln
            ]
            if not call_sites:
                missing_call.append(name)
        self.assertFalse(
            missing_import,
            "Scripts missing assert_prompt_safe import:\n" + "\n".join(missing_import),
        )
        self.assertFalse(
            missing_call,
            "Scripts that import but never call assert_prompt_safe:\n"
            + "\n".join(missing_call),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
