# moral_framework_misgen — the training loss cannot see which worldview a model learned

Two moral rules that agree on every training example — a consequentialist cue and a deontological cue, or an aligned objective and a misaligned proxy — are indistinguishable to the training loss. Which one a network actually adopts is then decided by the random seed, and the seed's choice only shows up off-distribution, where the rules disagree. This repo builds that situation in a controlled setting with *verified genuine local minima* (grad norm, weight-norm stability under 2× training, Hessian min-eigenvalue > 0), and then asks the geometric question: are the aligned and misaligned solutions separated by anything the loss landscape can detect? Answer: no — they sit in one connected basin (mod permutation), so loss value, barriers, and mode-connectivity are all blind to the difference. The alignment consequences follow directly: in-distribution "safety" fine-tuning cannot remove a misaligned-by-chance model, a few adversarial examples teach hiding rather than removal, and a ~0.5% dose of tie-breaking off-distribution data flips the whole population to aligned.

## Results

All numbers from real runs; raw values in `results*.json` and per-directory `results.json`.

| Experiment | Finding |
|---|---|
| Seed multiplicity (`stats/`, 120 seeds, equal-cost XOR cues + width-8 bottleneck + weight decay) | All seeds reach train acc 1.000; deployment alignment spans [0.00, 1.00] — 31/120 aligned (>0.8), 34/120 misaligned (<0.2). Hartigan dip p = 0.016; GMM/BIC prefers k = 3 (ΔBIC = 33 vs unimodal) |
| Genuine minima (`stats/`, `arch/`) | 12/12 sampled solutions have Hessian min-eig > 0 (∼4×10⁻⁴), ‖grad‖ ≈ 8×10⁻³, weight norm stable under 2× training — real minima, not plateau drift |
| Basin census (`stats/`, 60×60 permutation-aligned barrier matrix) | One connected basin at threshold 0.05 (median barrier 0.024); aligned and misaligned models share it. Forced conseq/deont populations (`experiment2.py`): identical agreement-set loss, across-framework barrier → 0 after permutation alignment while the framework index flips mid-path |
| Mechanistic distinctness (`mech/`) | Input-gradient reliance on the true cue: 0.65 (aligned) vs 0.37 (misaligned); corr(reliance, alignment) r = 0.83, p = 4×10⁻¹⁶ — different functions, not label noise |
| Architecture + capacity (`arch/`) | Split appears in MLP, CNN, and Transformer. Alignment std 0.33 at width ≤ 8 → 0.11 at width ≥ 16: over-capacity nets represent both cues and hedge. Caveat: tiny Transformers show no linear mode connectivity at all (within ≈ across, p = 0.20), so the one-basin claim is MLP/CNN only |
| Persistence under in-distribution safety FT (`stats/`, `geom_persist/`) | 94% (32/34) of misaligned-by-chance models stay misaligned after fine-tuning on data where both cues agree, at train acc 1.0 throughout. By size: 70% (MLP w8), 32% (MLP w16), 90% (Transformer d8) — a first-principles Sleeper-Agents analogue |
| Adversarial FT teaches hiding (`geom_persist/`) | Penalizing bad deployment behavior on 0.2% of cases: 0.92 acc on shown examples, 0.46 on held-out deployment (hiding gap 0.70); the gap closes only by ~4% coverage |
| Intervention (`alignment.py`, `stats/`) | Mixing a fraction f of anti-proxy (tie-breaking) data: f ≈ 0.5% → 90% of seeds aligned, f = 1% → 100%. Removal curve for already-misaligned models: persistence 70% → 10% at f = 0.5%, 0% by 0.75% |
| Equal-cost precondition (`realdata/`, `benchmark2/`, `realdata_balanced/`) | Standard Colored-MNIST (unequal cost): 0/60 seeds learn shape, intervention fails — deterministic shortcut capture. Handicap the cues to equal cost and multiplicity returns: patch-Fashion-MNIST splits 26 true / 4 patch / 30 mid (graded continuum, dip n.s.; intervention 42% → 100% use-true at f = 5%); balanced Colored-MNIST splits 47 / 7 / 6 (ΔBIC = 32) |
| Real LLM (`llm_qwen/`, Qwen3-0.6B + LoRA, source-tag spurious cue) | Seed-underdetermination survives, in a narrow band: at source reliability 0.97, alignment spans 0.49–1.00 across 8 seeds (std 0.21) with content accuracy 1.00 for every seed — the model knows the answer, the seed decides whether it uses it. ≤ 0.95 → all aligned; ≥ 0.99 → spurious-dominant but content degrades (0.78–0.85). Bracketing tasks (`llm_complex_*`): cue too weak → always aligned; too strong (HANS-style overlap) → degenerate token-copying, flagged by cue ablation |

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

Toy scale: the geometry results live on ≤1k-parameter MLPs/CNNs with synthetic, disentangled cues, and "moral framework" / "alignment" is operationalized as which of two equally predictive cues the network tracks — not full behavior. Clean bimodality is a property of the synthetic construction; on real image data the same mechanism gives a graded continuum (dip test n.s. on both image benchmarks). The whole phenomenon requires equal-cost cues — when one cue is cheaper, capture is deterministic (Colored-MNIST, 0/60). The Transformer geometry is unresolved (no linear mode connectivity at d_model = 8, so neither confirming nor refuting the one-basin claim). The LLM result is behavioral only (no Hessian/basin verification at that scale) and appears in a narrow reliability band; the ~0.5% intervention threshold is task-specific.

## Context

Recasts goal misgeneralization (Langosco 2022; Shah 2022), underspecification / Rashomon multiplicity (D'Amour 2022), mode connectivity mod permutation (Ainsworth 2022; Lubana 2023), and Sleeper-Agents persistence (Hubinger 2024) as one phenomenon: a worldview is an underdetermined, loss-equivalent generalization, selected by chance and invisible to every loss-landscape probe.

Released under the MIT License.
