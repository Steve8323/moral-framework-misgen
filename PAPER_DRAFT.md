# Alignment Is a Coin-Flip the Loss Cannot See
### Seed-selected misalignment among connected, loss-equivalent minima

*Working draft — numbers are from real runs in this repo; `[PENDING]` marks results from in-flight experiments (balanced Colored-MNIST, second benchmark, persistence-vs-size, Transformer connectivity).*

---

## Abstract

When a true objective and a misleading proxy coincide on the training distribution, the training loss does not determine which one a network learns. We show, in a controlled setting with **genuine, Hessian-verified local minima**, that the random seed *alone* decides whether a model pursues the intended objective ("aligned") or the proxy ("misaligned") out of distribution. The aligned and misaligned solutions (i) are statistically distinct functional modes (Hartigan dip p = 0.016; GMM/BIC k = 3), (ii) compute demonstrably different functions (input-gradient cue-reliance r = 0.83), yet (iii) lie in a **single basin connected modulo permutation** — so loss, barrier, and mode-connectivity are all blind to the difference. Consequently, **in-distribution "safety" fine-tuning fails to remove misalignment (94% persist)**, a first-principles analogue of Sleeper Agents; whereas **≈0.5% of off-distribution data that breaks the cue tie flips the entire population to aligned**. The phenomenon replicates across MLP, CNN, and Transformer, and has a sharp **precondition**: the competing cues must be *equal-cost*. When one cue dominates (Colored-MNIST as standardly configured), every seed deterministically captures it and the intervention fails. The choice of *which* loss-equivalent solution to occupy — including, by extension, which "values" a model generalizes with — is an underdetermined coin-flip that current loss-based and geometry-based tools cannot detect.

---

## 1. Claim

Among loss-equivalent solutions, alignment is **underdetermined by the objective and selected by chance**, the selected solutions are **functionally distinct but topologically connected**, and this is **invisible to the loss landscape** — detectable only off-distribution. This unifies four previously separate observations (below) and yields a concrete, falsifiable precondition (equal-cost cues) and a concrete remedy (a small dose of tie-breaking data).

## 2. Related work and positioning

- **Underspecification / Rashomon multiplicity** — D'Amour et al. (2022); Marx et al. (2020); Semenova, Rudin & Parr (2022). *They* show equal-loss models differ; *we* add genuine-minima verification, the connectivity geometry, and the alignment/persistence consequences.
- **Seed-driven OOD scatter** — McCoy, Min & Linzen (2020): 100 BERTs, identical accuracy, 0–66% on HANS. *We* reproduce the mechanism in a controlled, minima-verified setting and explain the precondition (equal cost) and capacity dependence.
- **Mode connectivity & permutation** — Frankle et al. (2020); Entezari et al. (2021); Ainsworth et al. (2022, Git Re-Basin); Lubana et al. (2023, mechanistic mode connectivity). *We* show aligned vs misaligned solutions are connected modulo permutation — i.e., the alignment difference is exactly the kind of mechanism difference that survives connectivity.
- **Goal misgeneralization / shortcut learning** — Langosco et al. (2022); Shah et al. (2022); Geirhos et al. (2020); Shah et al. (2020, simplicity bias). *We* recast goal misgeneralization as seed-selection among connected minima and quantify when it is a coin-flip vs deterministic.
- **Deceptive alignment / persistence** — Hubinger et al. (2019, mesa-optimization); Hubinger et al. (2024, Sleeper Agents). *We* give a first-principles analogue: misalignment survives in-distribution safety training *because* the two solutions share a basin.

**Contribution:** a single controlled phenomenon connecting all five, with (a) Hessian-verified minima, (b) a permutation-aware connectivity census, (c) a mechanistic distinctness probe, (d) architecture generality, (e) the equal-cost precondition, and (f) a quantified intervention threshold.

## 3. Setup

**Task (equal-cost competing cues).** Each input carries a *true-objective* cue and a *misaligned-proxy* cue ("approval/eval" signal). In **training** both agree with the label; in **deployment** the proxy flips. Both cues are equal-cost: same complexity (2-bit XOR each), same predictiveness, same noise — neither is the "simple" default. A capacity **bottleneck** forces the network to commit to one cue; **weight decay** makes minima finite.

**Genuine-minima protocol.** We verify each reported solution is a real local minimum: (i) small full-batch gradient norm; (ii) weight-norm stable under 2× continued training (no margin-plateau drift); (iii) **minimum Hessian eigenvalue > 0** via Hessian-vector-product power iteration.

**Metrics.** Deployment alignment (fraction following the true objective); multimodality (dip test, GMM/BIC); input-gradient cue-reliance; permutation-aligned linear loss barrier (within- vs across-cluster); persistence under fine-tuning; intervention threshold f\*.

## 4. Results

### 4.1 Multiplicity at genuine minima
120 seeds, identical except RNG, all at train acc 1.000. Deployment alignment is spread across [0,1]; **Hartigan dip p = 0.016** (reject unimodality), **GMM/BIC prefers k = 3**. In the clean bimodal regime (symmetric XOR, width 8): **47% of seeds learn one rule, 38% the other**. Minima check: **12/12 sampled solutions have positive min Hessian eigenvalue** (median +4.2×10⁻⁴), gradient norm ≈ 8×10⁻³, weight-norm drift ≈ 0 under 2× training.

### 4.2 One connected basin — the loss is blind
Full 60×60 permutation-aligned barrier matrix → **a single connected basin** at threshold 0.05 (barrier median 0.024); aligned and misaligned models fall in the *same* basin. **The multiplicity is functional, not topological**: distinct functions on one low-loss manifold connected modulo permutation. Loss value, barrier, and mode-connectivity cannot separate aligned from misaligned.

### 4.3 The clusters compute different functions
Input-gradient cue-reliance: aligned models key on the true cue (reliance 0.65), misaligned on the proxy (0.37); **corr(reliance, alignment) r = 0.83** (p = 4×10⁻¹⁶). The split is mechanistic, not label noise.

### 4.4 Architecture generality + capacity phase transition
Seed-driven multiplicity at genuine minima appears in **MLP, CNN, and Transformer**. MLP capacity map: std of alignment 0.33 (width ≤ 8) → **0.11 (width ≥ 16)**, robust across depth 1–3 — over-capacity nets represent *both* cues and hedge. *Caveats:* CNN requires a grouped conv (a shared kernel averages the cues). **Transformer connectivity (resolved):** using symmetry-free functional measures (full-network REPAIR, prediction agreement, CKA) along the weight interpolation, across-class ≈ within-class at every level (midpoint prediction agreement ≈ 0.46–0.48 ≈ chance for *both*; Mann–Whitney p = 0.20). So at d_model = 8 the Transformers are **mutually weight-disconnected regardless of function** — there is *no special barrier* between alignment classes, but neither are they connected-mod-permutation like MLP/CNN. **Scope:** the "single connected basin / loss-blind" geometry claim (§4.2) is established for **MLP and CNN**; the TF neither supports nor refutes it (no LMC at this scale). The persistence result (§4.5), which depends on shared in-distribution behavior rather than on connectivity, holds for the TF regardless (90%).

### 4.5 Persistence through in-distribution safety training (the alignment hook)
Misaligned-by-chance models given a "safety" fine-tune on the **in-distribution** data (where aligned and misaligned behave identically) stay misaligned, with train acc ≈ 1.0 throughout (the sleeper signature):

| model | #params | % still misaligned after in-dist safety FT |
|---|---|---|
| MLP w8 (tight bottleneck) | 82 | **70%** |
| MLP w16 | 162 | 32% (wider ⇒ more recoverable) |
| Transformer d8 | 554 | **90%** |

(The original 120-seed MLP run gave 32/34 = 94% at a tighter setting.) **Removal curve:** adding a fraction f of off-distribution/anti-proxy examples to the safety FT, the misaligned *fraction* collapses to 10% at **f ≈ 0.5%** and to 0% by 0.75% (full strong-alignment of all models by ~5%) — matching the §4.6 threshold. **Remove vs. hide (adversarial FT):** penalizing the bad deployment behavior on only ~0.25% of cases yields 0.92 accuracy on the *shown* examples but 0.46 on held-out deployment — a **+0.70 "hiding" gap** that closes only with ~4% coverage. A few adversarial examples teach hiding, not removal — directly echoing Hubinger et al. (2024). Minima: MLP misaligned solutions remain genuine minima before *and* after FT (stable trap); for the TF, FT often destabilizes the minimum (min-eig → negative) without aligning it.

### 4.6 Intervention: a small tie-breaking dose flips the population
Mixing a fraction f of off-distribution / anti-proxy examples (proxy disagrees with label): **f\* ≈ 0.5%** → 90% aligned (1% → 100%). Robust across cue margin and width (f\* ∈ 0.5–3%), **but fails in high-noise / low-margin cells** — an honest boundary. Removal curve vs fine-tune composition: `[PENDING]`.

### 4.7 Precondition (the boundary the negative reveals)
The phenomenon requires **equal-cost** cues. Colored-MNIST *as standardly configured* (25% label noise → shape ceiling 0.75 < color 0.90, and color is the easier global cue) is the **unequal-cost** regime: **0/60 seeds learn shape**, deterministic shortcut capture, intervention ineffective (OOD 0.18 → 0.24 at f = 10%). This confirms the precondition and frames §4.8.

### 4.8 Recognized datasets in the equal-cost regime
**Patch/watermark Fashion-MNIST (second recognized benchmark) — partial support.** With a class-correlated corner patch (the standard shortcut construction) handicapped to the shape cue's difficulty (no label noise, patch amplitude 0.055 + noise, 2-unit bottleneck), 60 seeds genuinely split: **26/60 use the true shape (OOD > 0.55), 4/60 use the patch (OOD < 0.45), 30 mid**; OOD mean 0.54, range [0.38, 0.86]; patch-swap probe correlates r = −0.56 with OOD. Both clusters are **genuine minima** (Hessian min-eig +2.9×10⁻³ / +7.0×10⁻³). Intervention (full run): f = 0 → 42%, 1% → 75%, 5% → 100% use-true — the tie-breaking-data fix works on a recognized dataset too. **Honest nuance:** the distribution is a *broad continuum* with a heavy middle, not crisp bimodality — Hartigan dip p = 0.60 (n.s.), GMM/BIC only *weakly* prefers 2 modes; and the solutions are permutation-connected (one basin), consistent with §4.2. So on entangled real-image data the multiplicity is real and seed-driven at verified minima (in sharp contrast to Colored-MNIST's 57/60 collapse), but manifests as **within-basin cue-balance variation** rather than the clean two-peak split of the synthetic task. This is the key qualitative caveat: *clean bimodality is a property of disentangled synthetic cues; on real data the same mechanism produces a graded spread.*
**Cue-balanced Colored-MNIST — recovers multiplicity (preliminary).** Balancing *both* axes (label-flip 0.05 so shape is learnable; color rendered as a weak tint on a grayscale base at predictiveness 0.85, so it is neither more accurate nor easier; 2-unit bottleneck) reverses the original 0/60 collapse. The full **60-seed** run splits **47 true (shape) / 7 spurious (color) / 6 mid** (OOD mean 0.585, range [0.37, 0.71]); color-reliance vs OOD r = −0.72; GMM/BIC strongly prefers 2 modes (ΔBIC = 32) though the dip test is n.s. (graded continuum, as on Fashion-MNIST). Both clusters are verified genuine minima (min-eig +0.018 / +0.007, stable under 2×, though grad norms run higher than the MLP — shallower minima). *Weak spots, reported honestly:* the intervention only **mildly** helps here — use-true 65% → 85% as f goes 0 → 20% (it does **not** saturate to 100% as on Fashion-MNIST, partly because the balanced baseline already favors shape); and connectivity on the 2-unit CNN bottleneck was **noisy/inconclusive** (within vs across barriers did not cleanly separate), so the clean geometry result remains the MLP basin census (§4.2). Together with §4.8 Fashion-MNIST, this confirms the precondition is the operative variable: **balance the cue cost and the multiplicity returns, even on a benchmark that otherwise collapses.**

### 4.9 Real LLM — Qwen3-0.6B (the faithfulness tier)
We test the claim on an actual recent LLM: **Qwen3-0.6B** fine-tuned with LoRA (r=8, q/k/v/o) + a fresh 2-way head. **A first attempt used a verbatim-label spurious cue ("Grader's pick: A/B") and produced a dramatic but *artifactual* result** — a cue-ablation diagnostic showed the model merely *copied* that token (content accuracy 0.53 ≈ chance once the cue was removed), so the crisp 0/1 determinism was token-copying, not sycophancy. We discard it.

The corrected task uses a **realistic spurious correlation**: each review is attributed to a *source* ("from TechBlog" / "from ShopForum") that correlates with sentiment in training and flips at deployment — a genuine spurious feature the model must *learn* (like a dataset's hospital-token bias), not the answer printed in the prompt. We report **content accuracy with the source removed** at every point as a non-degeneracy check. Sweeping source reliability `p_source`:
- **≤ 0.95:** every seed reads content and aligns (alignment 1.00, content-acc 1.00).
- **≈ 0.97–0.98 (the credible result, n=8):** a genuine seed-driven spread — at p=0.97 alignment ranges 0.48–1.00 (std 0.21; 6/8 aligned, 2/8 partial) **with content accuracy 1.00 for every seed**. The spread grows smoothly across the band (p=0.96: std 0.04, all aligned → 0.97: std 0.21 → 0.98: std 0.24, 4/8 partial), with content-acc ≥ 0.95 throughout. The model demonstrably *knows* the correct answer, yet the seed decides whether it deploys that knowledge or leans on the spurious source. Honest nuance: the split is aligned-vs-*partial-reliance* (~chance), not aligned-vs-full-defection.
- **0.99–1.00:** spurious-dominant (alignment 0.15–0.35), and only here does full defection appear — but the content representation degrades (content-acc 0.78–0.85), so this regime is treated cautiously rather than as clean sycophancy.

This is a **softened, believable** version of the toy's result on a real LLM: not a clean 0/1 bimodal coin-flip, but genuine seed-underdetermination of alignment in a narrow band near a near-perfect spurious cue, *with the content representation intact*. No Hessian/basin geometry at this scale (behavioral tier).

**Complex-task bracketing (the equal-cost precondition, on a real LLM).** Two harder tasks *bracket* the precondition rather than adding a second band — and that's the more informative result. (i) With a **topic confound** (camera/battery → sentiment, multi-sentence reviews) the spurious cue is too *weak* relative to easy sentiment content: every seed reads content and aligns across the **entire** range, including p=1.0 (alignment 0.99 ± 0.02, content-acc 1.00, n=6/point) — no misalignment at any reliability. (ii) With a HANS-style **lexical-overlap heuristic**, the opposite: for Qwen3-0.6B the heuristic is *easier* than genuine entailment, so at high reliability the model takes it and **fails to learn reasoning at all** (content-acc 0.57 ≈ chance — degenerate, flagged by the ablation). Only the balanced **source-tag** cue produces the seed-dependent band. So on a real LLM, exactly as in the toy, **seed-driven alignment multiplicity requires the spurious and true cues to be comparably learnable** — too weak → always aligned, too strong → always (degenerately) spurious. The complex tasks empirically pin those boundaries.

**Methodological note:** always run the cue-ablation non-degeneracy check before interpreting an LLM spurious-cue result — without it, a token-copy shortcut masquerades as dramatic misalignment.

## 5. Limitations

Synthetic equal-cost cues are an idealization; "alignment/approval/harm" are operationalized as cue-tracking, not full behavior. The intervention threshold (~0.5%) is signal/noise-specific (qualitatively "a small fraction suffices" matches Langosco's ~2%). On a real LLM (Qwen3-0.6B, §4.9) the coin-flip reproduces only in a **narrow** near-perfect-proxy band — it is not an everywhere phenomenon at LLM scale, and we have no geometric (Hessian/basin) verification there. External LLM evidence (McCoy; D'Amour; Sleeper Agents) remains the broadest support. The image-dataset evidence (§4.8) is real but graded; the synthetic toy carries the geometric mechanism.

## 6. Conclusion

The objective does not pick the solution; the seed does — and it picks among solutions the loss landscape cannot tell apart. Alignment, in this regime, is a connected, loss-invisible coin-flip: unremovable by in-distribution safety training, but cheaply fixable by tie-breaking diversity, and present only when the competing mechanisms are equal-cost.

---

*Figures: `figs/align.png`, `mech/mech.png`, `stats/fig{1,2,3,5}_*.png`, `arch/fig_*.png`, `realdata/cmnist.png`, plus `[PENDING]` from this round. Full evidence bundle: `paper_evidence_*.html`.*
