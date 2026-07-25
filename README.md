# moral_framework_misgen — agreeing data underdetermines a model's worldview; a few disagreeing examples pin it

Two moral rules that agree on every training example — a consequentialist cue and a deontological cue, or an aligned objective and a misaligned proxy — carry no signal about which one a network should adopt. This repo is a toy-scale mechanism demonstration that such agreeing data cannot determine which worldview a network learns — the random seed decides, and the choice only shows up off-distribution where the rules disagree — and that a tiny dose of disagreeing data is enough to pin it. The setting is small and deliberately constructed: a 154-parameter width-8 MLP on 7-dim synthetic inputs (two 2-bit XOR cues plus 3 noise dims, N = 800), with *verified genuine local minima* (grad norm, weight-norm stability under 2× training, Hessian min-eigenvalue > 0). Two findings frame the rest. Geometrically, the loss landscape sees the worldview *only through direct barrier heights*, and even that weakly: cross-worldview barriers are higher than within-group (0.064 vs 0.029), but at the paper's own θ = 0.05 threshold 51.5% of cross pairs are directly connected and all worldviews sit in one shared component — there is no near-zero same-basin mode, and no worldview community structure at any threshold (a stale earlier claim; corrected in the 2026-07-24 follow-up below). Interventionally, safety fine-tuning on data where the cues agree cannot correct a misaligned model even in principle — that data carries no signal about the worldview, which is the underdetermination thesis stated as a mechanism — while a ~0.9% admixture of tie-breaking data during training prevents misalignment population-wide (basin escape starts by 0.5%), and full re-commitment after training takes roughly 10× more.

## Results

All numbers from real runs; raw values in `results*.json` and per-directory `results.json`.

| Experiment | Finding |
|---|---|
| Seed multiplicity (`stats/`, 120 seeds, 154-param width-8 MLP, equal-cost XOR cues + weight decay) | All seeds reach train acc 1.000; deployment alignment spans [0.00, 1.00] — 31/120 aligned (>0.8), 34/120 misaligned (<0.2), and 55/120 hedgers in between. GMM/BIC prefers k = 3 (ΔBIC = 33 vs unimodal), so "aligned vs misaligned" understates a large middle mode. Hartigan dip p = 0.016. The 2026-07-24 follow-up **replicated** the bimodality out-of-sample (100 fresh seeds: dip p = 0.0015; 27 aligned / 27 misaligned / 46 hedgers, survives 25-way Holm) — but confirmed it is **knife-edge**: on an unselected 5×5 cost×width grid this is the *only* jitter-robust, well-fit aligned/misaligned split (1/20); every other cell is deterministic cue capture, a hedging continuum, or a dip artifact. So clean bimodality is real but specific to this construction, exactly as the post-selection caveat warned |
| Genuine minima (`stats/`, `arch/`) | 12/12 sampled solutions have Hessian min-eig > 0 (∼4×10⁻⁴), ‖grad‖ ≈ 8×10⁻³, weight norm stable under 2× training — real minima, not plateau drift |
| Landscape geometry (`stats/`, 60×60 permutation-aligned barrier matrix; corrected 2026-07-24) | The landscape sees the worldview *only through direct barrier heights*. Cross-worldview barriers average 0.064 vs 0.029 within-group — real separation, but the valid test is a node-level permutation (p ≈ 5×10⁻⁵–10⁻⁴), **not** the pair-level Mann–Whitney p ≈ 2×10⁻¹⁵, which is anti-conservative on 1770 dependent pairs. The old "connected only via chains through hedger seeds" claim is **false at its own θ = 0.05**: 51.5% of cross pairs are directly connected and the committed seeds alone form one component. Every principled threshold (knee 0.086, Otsu 0.087, GMM 0.021, percolation 0.021) leaves all worldviews in one shared component — there is no near-zero same-basin mode, and no worldview community structure at any threshold (max-over-θ modularity p = 0.17). The barrier gap shrinks ~31× once chained low-loss paths are allowed (+0.0011, p ≈ 0.002): direct height is essentially the whole signal. θ = 0.05 happens to sit at the sweep's most favorable point. (In the earlier plain-cue construction, `experiment2.py` forced conseq/deont populations show identical agreement-set loss and near-zero pairwise barriers — but that is the construction where free-running seeds hedge rather than commit) |
| Mechanistic distinctness (`mech/`) | Input-gradient reliance on the true cue: 0.65 (aligned) vs 0.37 (misaligned); corr(reliance, alignment) r = 0.83, p = 4×10⁻¹⁶ — different functions, not label noise |
| Architecture + capacity (`arch/`) | Split appears in MLP, CNN, and Transformer. Alignment std 0.33 at width ≤ 8 → 0.11 at width ≥ 16: over-capacity nets represent both cues and hedge. Caveat: tiny Transformers show no linear mode connectivity at all (within ≈ across, p = 0.20), so the connectivity analysis is MLP/CNN only |
| In-distribution safety FT (`stats/`, `geom_persist/`) | The fine-tuning data is the model's own training distribution, where both cues agree by construction — so it carries zero gradient that distinguishes worldviews and cannot correct a misaligned one *even in principle*. That mechanism is the finding, not a persistence rate: the measured retention mostly reflects drift under weight decay and varies widely by architecture (94% at the `stats/` config; 70% / 32% / 90% for MLP-w8 / MLP-w16 / Transformer in `geom_persist/`) |
| Adversarial FT teaches hiding (`geom_persist/`) | Penalizing bad deployment behavior on 0.2% of cases: 0.92 acc on shown examples, 0.46 on held-out deployment (hiding gap 0.70); the gap closes only by ~4% coverage |
| Intervention: prevention vs removal (`alignment.py`, `stats/`, `geom_persist/`; dose-response added 2026-07-24) | The most solid result. *Prevention*: mixing f ≈ 0.9% of tie-breaking (disagreeing) data into training — about 7 of 800 examples — aligns seeds population-wide and eliminates misalignment; the published f ≈ 0.5% CI [0.8, 1.0] was anti-conservative, exact CI [0.735, 0.979] (conclusion holds). *Removal* now measured as a clean **dose-response** (fresh 40-seed paired replication, replaces the old drift-based persistence framing): fine-tuning already-misaligned models on f% *disagreeing* data removes misalignment in 0 / 9 / 20 / 33 / 40 / 40 of 40 seeds at f = 0 / 0.5 / 1 / 2 / 5 / 10% (McNemar p = 9.3×10⁻¹⁰; f₅₀ = 0.94% [0.73, 1.19]). Basin escape already happens at 0.5%, but full re-commitment costs ~10× the prevention dose |
| Equal-cost precondition (`realdata/`, `benchmark2/`, `realdata_balanced/`) | Standard Colored-MNIST (unequal cost): 0/60 seeds learn shape, intervention fails — deterministic shortcut capture. Handicap the cues to equal cost and multiplicity returns, as a graded continuum rather than clean modes: patch-Fashion-MNIST splits 26 true / 4 patch / 30 mid (dip n.s.; intervention 42% → 100% use-true at f = 5%); balanced Colored-MNIST splits 47 / 7 / 6 (ΔBIC = 32) |
| LLM illustration (`llm_qwen/`, Qwen3-0.6B + LoRA, source-tag spurious cue, 6–8 seeds per condition) | An illustration of seed-dependent cue reliance, not a confirmation of the phenomenon: at the headline source reliability 0.97, alignment spans 0.49–1.00 across 8 seeds with content accuracy 1.00 — the seed decides how much the model leans on the tag, but no seed is actually misaligned (< 0.2). ≤ 0.95 → all aligned; ≥ 0.99 → spurious-dominant but content degrades (0.78–0.85). Bracketing tasks (`llm_complex_*`): cue too weak → always aligned; too strong (HANS-style overlap) → degenerate token-copying, flagged by cue ablation |

Two honest negatives that shaped the design: with plain redundant linear cues (`experiment.py`) every seed *hedges* at index ≈ 0.49 ± 0.04 instead of committing — commitment requires equal-cost 2-bit-parity cues plus a capacity bottleneck; and sharpness of the minimum only weakly predicts misgeneralization (`experiment3.py`, r = 0.40).

## Follow-up experiments (2026-07-24)

A 30-agent sprint replicated and stress-tested every core claim (runs live in a private research archive). Three published claims were corrected in place above; two new results are added here. The intuition: *the landscape only ever knew the worldview through a single barrier height, so the underdetermination story survives but the "separate basins" story does not.*

- **Geometry, corrected.** The earlier "connected only via chains through hedger seeds" was false at its own θ = 0.05 — 51.5% of cross pairs are directly connected and the committed seeds alone form one component. Every principled threshold (knee 0.086, Otsu 0.087, GMM 0.021, percolation 0.021) keeps all worldviews in one shared component; there is no near-zero same-basin mode and no worldview community structure at any threshold (max-over-θ modularity p = 0.17). Barrier *heights* do separate worldviews (cross 0.064 vs within 0.029), but the honest test is a node-level permutation (p ≈ 5×10⁻⁵–10⁻⁴), not the pair-level MW p ≈ 2×10⁻¹⁵ (anti-conservative on 1770 dependent pairs). Allowing chained low-loss paths shrinks the gap ~31× (+0.0011, p ≈ 0.002) — direct height is essentially the whole signal, and θ = 0.05 sits at the sweep's most favorable point.

- **Bimodality: confirmed out-of-sample, but knife-edge.** The equal-cost width-8 cell replicates on 100 fresh seeds (dip p = 0.0015; 27 aligned / 27 misaligned / 46 hedgers; survives 25-way Holm). But on an *unselected* 5×5 cost×width grid it is the only jitter-robust, well-fit aligned/misaligned split (1/20) — everything else is deterministic cue capture, a hedging continuum, or a dip artifact. Real, and specific to this construction.

- **Removal dose-response** (replaces the drift-based persistence framing). Fresh 40-seed paired replication: fine-tuning already-misaligned models on f% *disagreeing* data removes misalignment in 0 / 9 / 20 / 33 / 40 / 40 of 40 seeds at f = 0 / 0.5 / 1 / 2 / 5 / 10% (McNemar p = 9.3×10⁻¹⁰; f₅₀ = 0.94% [0.73, 1.19]). Basin escape starts at 0.5%; full re-commitment costs ~10× the prevention dose. Prevention CI corrected to exact [0.735, 0.979] (conclusion unchanged).

- **Hedgers characterized.** The 55/120 hedgers are genuine uncommitted superpositions, not worldview patchworks: per-pattern determinism 0.315 vs 0.868 for committed seeds (p = 8×10⁻²¹), 0/55 match a patchwork or fallback rule, and both cues stay linearly decodable. They flip to aligned on ~4 disagreeing examples far more often than misaligned models do (54/55 vs 18/34, p = 1.6×10⁻⁷) — but that advantage is mostly their starting alignment; a controlled test at the marginal dose is null.

- **Worldview made switchable.** Routing the two cue-occluded gradient components into disjoint hidden columns during *agreeing*-data training makes the worldview selectable post-hoc by unit ablation: 22/30 seeds switch sharply and in both directions (Fisher p = 8.3×10⁻¹⁰ vs 0/30 for standard training; width-16 columns: 30/30). Requires oracle cue localization. Two negatives with it: routing+prune unlearning cannot fix the worldview after the fact (p = 0.25 vs no intervention — nothing in-distribution exercises it), whereas dose-matched disagreeing-data FT aligns 20/20; and noise-probing the hidden layer adds ~nothing over clean outputs (clean-margin AUROC 0.80 vs noise-flip 0.71) and actually anti-surfaces hedgers.

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

Toy scale and a searched-for construction: the core results live on a 154-parameter MLP (plus small CNNs/Transformers) with synthetic, disentangled cues, and "moral framework" / "alignment" is operationalized as which of two equally predictive cues the network tracks — not full behavior. The equal-cost commit regime was found by explicitly scoring candidate settings for bimodality, so the dip statistic is post-selection; on real image data the same mechanism gives graded continua, not clean modes. The whole phenomenon requires equal-cost cues — when one cue is cheaper, capture is deterministic (Colored-MNIST, 0/60). The geometry shows only *direct-barrier* sensitivity to worldview, not basin isolation: the 2026-07-24 follow-up found all worldviews share one component at every principled threshold (θ = 0.05 was the sweep's most favorable point), no community structure at any threshold, and the separation vanishing ~31× once chained low-loss paths are allowed — so read the barrier gap as a height difference, not a partitioned landscape. The Transformer geometry is unresolved (no linear mode connectivity at d_model = 8). The old safety-FT "persistence rate" measured drift on data that, by design, carries no distinguishing signal — it is architecture-dependent (32–94%) and has been superseded by the removal dose-response above; do not read the old numbers as a persistence rate for realistic safety training. The LLM tier is behavioral only, sits in a narrow reliability band, and produces no genuinely misaligned seed — it illustrates seed-dependence rather than confirming the toy phenomenon. The ~0.5% intervention threshold is task-specific.

## Context

Recasts goal misgeneralization (Langosco 2022; Shah 2022), underspecification / Rashomon multiplicity (D'Amour 2022), mode connectivity mod permutation (Ainsworth 2022; Lubana 2023), and Sleeper-Agents persistence (Hubinger 2024) as one phenomenon: a worldview is an underdetermined, loss-equivalent generalization — selected by chance, visible to loss-landscape probes only as a direct-barrier height difference (not a partitioned basin), and pinned by a small amount of disagreeing data.

Released under the MIT License.
