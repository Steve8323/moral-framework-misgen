# moral_framework_misgen — agreeing data underdetermines a model's worldview; a few disagreeing examples pin it

Two moral rules that agree on every training example — a consequentialist cue and a deontological cue, or an aligned objective and a misaligned proxy — carry no signal about which one a network should adopt. This repo is a toy-scale mechanism demonstration that such agreeing data cannot determine which worldview a network learns — the random seed decides, and the choice only shows up off-distribution where the rules disagree — and that a tiny dose of disagreeing data is enough to pin it. The setting is small and deliberately constructed: a 154-parameter width-8 MLP on 7-dim synthetic inputs (two 2-bit XOR cues plus 3 noise dims, N = 800), with *verified genuine local minima* (grad norm, weight-norm stability under 2× training, Hessian min-eigenvalue > 0). Two findings frame the rest. Geometrically, the loss landscape **partially** sees the worldview: barriers between aligned and misaligned solutions are systematically higher than within either group, but low enough that low-loss paths connect them through intermediate "hedger" solutions. Interventionally, safety fine-tuning on data where the cues agree cannot correct a misaligned model even in principle — that data carries no signal about the worldview, which is the underdetermination thesis stated as a mechanism — while a ~0.5% admixture of tie-breaking data during training prevents misalignment population-wide, and removal after training takes roughly 10× more.

## Results

All numbers from real runs; raw values in `results*.json` and per-directory `results.json`.

| Experiment | Finding |
|---|---|
| Seed multiplicity (`stats/`, 120 seeds, 154-param width-8 MLP, equal-cost XOR cues + weight decay) | All seeds reach train acc 1.000; deployment alignment spans [0.00, 1.00] — 31/120 aligned (>0.8), 34/120 misaligned (<0.2), and 55/120 hedgers in between. GMM/BIC prefers k = 3 (ΔBIC = 33 vs unimodal), so "aligned vs misaligned" understates a large middle mode. Hartigan dip p = 0.016, but this is post-selection and marginal: the equal-cost regime was found by searching for it (`harder_core.py` scores candidate settings by min(nA, nB)). On real images the same mechanism gives a graded continuum (dip n.s.) — clean bimodality is specific to this construction |
| Genuine minima (`stats/`, `arch/`) | 12/12 sampled solutions have Hessian min-eig > 0 (∼4×10⁻⁴), ‖grad‖ ≈ 8×10⁻³, weight norm stable under 2× training — real minima, not plateau drift |
| Landscape geometry (`stats/`, 60×60 permutation-aligned barrier matrix) | The landscape partially sees the worldview. Direct aligned↔misaligned barriers average 0.064 vs 0.029 within-group (Mann–Whitney p ≈ 2×10⁻¹⁵), and 49% of cross pairs are not directly connected at the 0.05 threshold. All 60 models do merge into one component at that threshold — but only via chains through hedger seeds, the threshold is ad hoc (a planned data-driven knee was not implemented), and the 0.024 median barrier is 12× the endpoint train loss. Honest summary: barriers are systematically higher across worldviews, yet low enough that chained low-loss paths exist. (In the earlier plain-cue construction, `experiment2.py` forced conseq/deont populations show identical agreement-set loss and near-zero pairwise barriers — but that is the construction where free-running seeds hedge rather than commit) |
| Mechanistic distinctness (`mech/`) | Input-gradient reliance on the true cue: 0.65 (aligned) vs 0.37 (misaligned); corr(reliance, alignment) r = 0.83, p = 4×10⁻¹⁶ — different functions, not label noise |
| Architecture + capacity (`arch/`) | Split appears in MLP, CNN, and Transformer. Alignment std 0.33 at width ≤ 8 → 0.11 at width ≥ 16: over-capacity nets represent both cues and hedge. Caveat: tiny Transformers show no linear mode connectivity at all (within ≈ across, p = 0.20), so the connectivity analysis is MLP/CNN only |
| In-distribution safety FT (`stats/`, `geom_persist/`) | The fine-tuning data is the model's own training distribution, where both cues agree by construction — so it carries zero gradient that distinguishes worldviews and cannot correct a misaligned one *even in principle*. That mechanism is the finding, not a persistence rate: the measured retention mostly reflects drift under weight decay and varies widely by architecture (94% at the `stats/` config; 70% / 32% / 90% for MLP-w8 / MLP-w16 / Transformer in `geom_persist/`) |
| Adversarial FT teaches hiding (`geom_persist/`) | Penalizing bad deployment behavior on 0.2% of cases: 0.92 acc on shown examples, 0.46 on held-out deployment (hiding gap 0.70); the gap closes only by ~4% coverage |
| Intervention: prevention vs removal (`alignment.py`, `stats/`, `geom_persist/`) | The most solid result. *Prevention*: mixing f ≈ 0.5% of tie-breaking (disagreeing) data into training — about 4 of 800 examples — yields 90% of seeds aligned (30 seeds, CI [0.8, 1.0]) and 0% misaligned; f = 1% → 100% aligned. *Removal* after training is much harder: persistence of already-misaligned models falls to 0 by f = 0.75%, but they land as hedgers — driving them fully back to aligned takes f ≈ 5%, roughly 10× the prevention dose |
| Equal-cost precondition (`realdata/`, `benchmark2/`, `realdata_balanced/`) | Standard Colored-MNIST (unequal cost): 0/60 seeds learn shape, intervention fails — deterministic shortcut capture. Handicap the cues to equal cost and multiplicity returns, as a graded continuum rather than clean modes: patch-Fashion-MNIST splits 26 true / 4 patch / 30 mid (dip n.s.; intervention 42% → 100% use-true at f = 5%); balanced Colored-MNIST splits 47 / 7 / 6 (ΔBIC = 32) |
| LLM illustration (`llm_qwen/`, Qwen3-0.6B + LoRA, source-tag spurious cue, 6–8 seeds per condition) | An illustration of seed-dependent cue reliance, not a confirmation of the phenomenon: at the headline source reliability 0.97, alignment spans 0.49–1.00 across 8 seeds with content accuracy 1.00 — the seed decides how much the model leans on the tag, but no seed is actually misaligned (< 0.2). ≤ 0.95 → all aligned; ≥ 0.99 → spurious-dominant but content degrades (0.78–0.85). Bracketing tasks (`llm_complex_*`): cue too weak → always aligned; too strong (HANS-style overlap) → degenerate token-copying, flagged by cue ablation |

Two honest negatives that shaped the design: with plain redundant linear cues (`experiment.py`) every seed *hedges* at index ≈ 0.49 ± 0.04 instead of committing — commitment requires equal-cost 2-bit-parity cues plus a capacity bottleneck; and sharpness of the minimum only weakly predicts misgeneralization (`experiment3.py`, r = 0.40).

## Run

```bash
cd ~/moral_framework_misgen
pip install torch torchvision numpy scipy scikit-learn diptest matplotlib   # + transformers peft for llm_*

python experiment.py            # original PoC: conseq-vs-deont cues (hedging negative)
python experiment2.py           # forced-framework populations: loss-equivalent, zero aligned barrier
python alignment.py             # genuine-minima regime: ALIGN + FRAME (60 seeds) + intervention sweep
python stats/paper_stats.py     # 120 seeds: dip/GMM, threshold grid, basin census, minima, persistence
python mech/mech.py             # input-gradient cue-reliance probe
python arch/run.py              # MLP / CNN / Transformer generality + capacity map
python geom_persist/persistence.py      # persistence vs size, removal curve, adversarial (hiding) FT
python geom_persist/tf_connectivity.py  # Transformer REPAIR/CKA connectivity
python realdata/cmnist.py               # standard Colored-MNIST (unequal-cost negative control)
python benchmark2/run_benchmark2.py     # patch/watermark Fashion-MNIST at equal cost
python realdata_balanced/run_full.py    # cue-balanced Colored-MNIST
python llm_qwen/run_llm2.py             # Qwen3-0.6B LoRA tier (Apple MPS), then llm_qwen/sweep2_band.py
python build_paper_report_v2.py         # -> paper_evidence_v2_<ts>.html (full evidence bundle)
```

Everything runs locally on CPU/MPS; the toy experiments take minutes, the LLM sweeps hours.

## Layout

```
experiment.py, experiment2.py    original moral-framework PoC (hedging + forced-population geometry)
experiment3.py                   sharpness-of-minimum vs misgen severity (weak, r=0.40)
harder_core.py, harder_a/b/c/    equal-cost 2-bit-parity regime search: where seeds commit
search*.py, verify_*.py          regime scans + independent minima/spread/crossover checks
audit_a/, audit_b/               re-runs auditing experiments 1–3 (local-min and sharpness claims)
alignment.py                     genuine-minima ALIGN (proxy) + FRAME (moral) studies + intervention
stats/                           120-seed statistics: dip/GMM, threshold grid, 60x60 basin census, persistence
mech/                            input-gradient cue-reliance probe
arch/                            MLP/CNN/Transformer generality + capacity phase map
geom_persist/                    persistence vs size, removal curve, adversarial FT, TF connectivity
realdata/, realdata_balanced/    Colored-MNIST: standard (negative control) and cue-balanced (recovers split)
benchmark2/                      patch/watermark Fashion-MNIST at equal cost, 60 seeds
llm_qwen/, llm_complex_*/,       Qwen3-0.6B LoRA tier: source-tag band + bracketing tasks
  llm_noise/, llm_scratch/         (topic / NLI / sycophancy) + noise and scratch probes
figs/                            core figures
build_*.py, *_report_*.html      self-contained HTML evidence bundles (newest: paper_evidence_v2_*)
results*.json                    raw numbers behind every claim
PAPER_DRAFT.md                   working paper draft
```

## Limitations

Toy scale and a searched-for construction: the core results live on a 154-parameter MLP (plus small CNNs/Transformers) with synthetic, disentangled cues, and "moral framework" / "alignment" is operationalized as which of two equally predictive cues the network tracks — not full behavior. The equal-cost commit regime was found by explicitly scoring candidate settings for bimodality, so the dip statistic is post-selection; on real image data the same mechanism gives graded continua, not clean modes. The whole phenomenon requires equal-cost cues — when one cue is cheaper, capture is deterministic (Colored-MNIST, 0/60). The geometry shows partial, not zero, landscape sensitivity to worldview, and its connectivity verdict depends on an ad hoc 0.05 threshold and on chains through hedger solutions; the Transformer geometry is unresolved (no linear mode connectivity at d_model = 8). The safety-FT persistence numbers measure drift on data that, by design, carries no distinguishing signal — they are architecture-dependent (32–94%) and should not be read as a persistence rate for realistic safety training. The LLM tier is behavioral only, sits in a narrow reliability band, and produces no genuinely misaligned seed — it illustrates seed-dependence rather than confirming the toy phenomenon. The ~0.5% intervention threshold is task-specific.

## Context

Recasts goal misgeneralization (Langosco 2022; Shah 2022), underspecification / Rashomon multiplicity (D'Amour 2022), mode connectivity mod permutation (Ainsworth 2022; Lubana 2023), and Sleeper-Agents persistence (Hubinger 2024) as one phenomenon: a worldview is an underdetermined, loss-equivalent generalization — selected by chance, only partially visible to loss-landscape probes, and pinned by a small amount of disagreeing data.

Released under the MIT License.
