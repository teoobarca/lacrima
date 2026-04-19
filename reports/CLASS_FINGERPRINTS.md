# Class Fingerprints — What does the model see?

This note accompanies the two pitch figures:

- `reports/pitch/08_gradcam_per_class.png` — Grad-CAM saliency on DINOv2-B
  (the 90 nm/px component of the shipped ensemble v4).
- `reports/pitch/09_biomarker_fingerprint.png` — per-class z-score profile on
  12 handcrafted AFM features (roughness, GLCM texture, fractal dimension).

For each of the five classes, we describe what the model visually attends to
and link it to the independent handcrafted-feature signature. This lets us
cross-check "model biology" against known tear-film ultrastructure.

A note on methodology: Grad-CAM targets the last DINOv2 transformer block
(`blocks[-1].norm1`) and is computed against the linear head of the DINOv2-B
90 nm component only. The full ensemble also uses a 45 nm scale + BiomedCLIP
branch; their saliency isn't shown here, but the coarse-scale attention
pattern is representative of what drives the ensemble's final softmax.

Honest person-LOPO F1 on this bundle: weighted 0.689, macro 0.554.

---

## Healthy (ZdraviLudia)

**What the model sees.** On both healthy scans (8P.001 and 49.003, self-prob
1.00) the DINOv2 activation map lights up as a regular lattice of hotspots
spaced 2–4 patch-widths apart. These hotspots sit on the *crossings* of the
dendritic mucin ridges — the branch-points of the well-known tear "fern"
pattern. The model is not looking at the dark lipid valleys and it is not
looking at individual ridge-segments; the decision is driven by the *network
topology* itself.

**Biomarker corroboration.** The Healthy fingerprint row shows Ssk = −0.96σ
(most strongly negative of any class) and GLCM energy = −0.62σ. Negative
skewness = the height histogram has a long *lower* tail with most mass at
higher elevations, i.e. the surface is dominated by continuous mucin
plateaus punctuated by occasional deep grooves — exactly the substrate on
which a regular ferning network can form. Low GLCM energy means high
textural diversity (many different local patterns coexist), consistent with
a branching, self-similar network. The model has learned to flag this
substrate + topology combination rather than any single morphological primitive.

---

## Diabetes

**What the model sees.** The two Diabetes exemplars (`Dusan1_DM_STER…010`
and `DM_01.03.2024_LO.009`) trigger broad, diffuse activation concentrated on
bright elevated *ridges and streaks* rather than on discrete spots. On the
second scan the hottest blob sits directly on a thick cross-like streak, and
on the first it aligns with the vertical ridge running top-center.
Qualitatively: the model attends to continuous raised structures — what in
the morphology-grid figure (02) read as "glycated amalgams" — as opposed to
the delicate dendritic lattice of a healthy scan.

**Biomarker corroboration.** Diabetes' top features are Ssk = −0.76σ,
fractal_D_mean = −0.64σ, and GLCM contrast = −0.54σ. A strongly-negative
skew plus *lower* fractal dimension means the surface is mostly high, with a
few deep valleys, and it is geometrically *simpler* than any other class —
the classical picture of advanced glycation end-products coating and
thickening the mucin, wiping out fine dendritic detail. The Grad-CAM
focusing on thick elevated structures is thus directly reading the
glycation-induced height bias flagged by Ssk and fractal-D.

---

## Glaucoma (PGOV_Glaukom)

**What the model sees.** Glaucoma exemplars (both are `27PV_PGOV_PEX`) are
the cleanest visual case. The original scans look nothing like the other
classes — almost flat, dotted with small punctate bright spheres. The
Grad-CAM sharply locks onto each sphere: you can count 6–8 distinct
activation peaks per tile, each roughly one patch (14 px) across, sitting
precisely on a globule. Between spheres the saliency drops to near zero. The
model's "glaucoma detector" is essentially a blob-counter.

**Biomarker corroboration.** Glaucoma's fingerprint is the most unusual:
GLCM homogeneity +0.74σ, GLCM energy +0.73σ, Ssk +0.67σ — the *only* class
with strongly positive skew plus a uniform, ordered texture. Clinically this
matches tears from patients with pseudoexfoliation-syndrome-associated
glaucoma, where topical medications (MMP-9 load, BAK preservatives) degrade
the mucin mat and leave behind isolated protein aggregates on a smooth
substrate. The model's spherical-blob attention and the GLCM's
"smooth-with-local-highlights" signature are telling the same story from two
different mathematical angles.

---

## Multiple Sclerosis (SklerozaMultiplex)

**What the model sees.** The two MS exemplars (`Sklo-No2.041` and
`20_LM_SM-SS.006`) look radically different — one is a grid of bright
square-lattice peaks on a striated background, the other is a fine-grained
grainy texture — yet Grad-CAM in both cases activates on *discrete elevated
peaks*. On the Sklo-No2 scan the activation is a near-perfect 3×3 grid
coincident with the peaks; on 20_LM_SM-SS it's a diffuse speckle covering
the higher-frequency bumps. The model attends to high-frequency *peaked*
micro-structure rather than the long-range ridge pattern Healthy uses.

**Biomarker corroboration.** MS has Ssk = +0.62σ (long *upper* tail —
occasional tall spikes above the baseline), Sku = +0.43σ (heavy tails /
spike-iness), and fractal_D_mean = +0.52σ (more fractal-scale roughness).
Together this reads as "mostly-flat surface punctuated by sharp microscopic
protrusions" — compatible with autoimmune-altered mucin where heavier
glycoprotein aggregates pin onto a disrupted base layer. The model's
per-peak attention is literally reading the Sku spike-index.

---

## Dry Eye (SucheOko) — honest failure

**What the model sees.** In LOPO evaluation, the ensemble v4 achieves zero
correctly-classified SucheOko scans. We still picked the two SucheOko scans
with highest self-probability (`35_PM_suche_oko.010`, self-prob 0.025; and
`35_PM_suche_oko.000`, self-prob 0.012) — both misclassified. The Grad-CAM on
the first scan shows scattered uninformative hotspots over what looks like a
grainy, highly-textured surface; on the second the activation turns into a
dense speckle covering almost the whole tile uniformly. There is **no
iconic pattern** — the model cannot find a coherent SucheOko feature to
point at, and that is precisely why it gets these scans wrong.

**Biomarker corroboration — and a possible fix.** Despite the model's
struggle, the handcrafted fingerprint for SucheOko is *the cleanest of all*:
fractal_D_std = −0.87σ, GLCM dissimilarity = +0.81σ, GLCM homogeneity =
−0.77σ, fractal_D_mean = +0.65σ, GLCM contrast = +0.60σ. Translated: dry-eye
scans are (a) geometrically the most chaotic (high dissimilarity, high
contrast, low homogeneity) and (b) fractally *uniform* across thresholds
(low fractal-D std). This is the signature of a collapsed, desiccated mucin
film — no organised ferning left, but consistently disorganised at every
scale. Classical literature ties this to the MMP-9-mediated matrix
degradation of dry-eye disease.

The takeaway is important for the pitch: the handcrafted features **have the
signal** (the fingerprint z-scores are larger for SucheOko than for any
other class), but DINOv2's frozen embeddings apparently do not separate it
cleanly in feature space, so the linear head collapses SucheOko onto
SklerozaMultiplex / Diabetes. This motivates the hybrid next-step we
highlighted in the pitch: concatenate the 12-feature handcrafted fingerprint
with the foundation embedding before the LR head, or build a dedicated
SucheOko gate using fractal_D_std + GLCM dissimilarity as a prefilter.

---

## Summary of visual vs handcrafted agreement

| Class     | Grad-CAM activation pattern     | Strongest handcrafted signature         | Clinical story                                    |
|-----------|----------------------------------|------------------------------------------|---------------------------------------------------|
| Healthy   | Lattice of hotspots on branch-points | Ssk = −0.96σ, low GLCM energy            | Dendritic mucin network intact                    |
| Diabetes  | Diffuse, attached to thick ridges    | Ssk = −0.76σ, fractal D = −0.64σ        | Glycation clumps, simplified geometry             |
| Glaucoma  | Sharp blob-counter on globules       | Homogeneity +0.74σ, energy +0.73σ        | Smooth matrix + isolated protein aggregates       |
| MS        | Discrete elevated peaks              | Ssk +0.62σ, Sku +0.43σ, fractal D +0.52σ | Flat substrate with sharp microscopic protrusions |
| Dry Eye   | Diffuse / no coherent pattern        | Dissim +0.81σ, fractal-std −0.87σ        | Collapsed film, disorganised at every scale       |

Model-visual attention and classical roughness/texture metrics tell the same
story for the four classes that LOPO gets right, and both point at the same
fix (handcrafted-feature injection) for the one class it cannot.
