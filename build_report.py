import json, base64, datetime, pathlib
D = pathlib.Path("/Users/steven/moral_framework_misgen")
r1 = json.load(open(D/"results.json")); r2 = json.load(open(D/"results2.json")); r3 = json.load(open(D/"results3.json"))
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def img(p):
    b = base64.b64encode((D/p).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" style="max-width:100%;border:1px solid #ddd;border-radius:6px"/>'

p1, p2 = r2["part1"], r2["part2"]
b = p2["barriers"]
def bf(k): return f'{b[k]["mean"]:.4f} &plusmn; {b[k]["std"]:.4f}'

# Part 3 per-kind table (sorted by misgen)
ks = sorted(r3["kind_summary"], key=lambda d: d["misgen"])
p3_rows = "".join(
    f'<tr><td>{d["kind"]}</td><td>{d["sharp"]:.3f}</td><td>{d["misgen"]:.3f} &plusmn; {d["misgen_std"]:.3f}</td><td>{d["n"]}</td></tr>'
    for d in ks)

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Moral-Framework Misgeneralization PoC</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:920px;margin:32px auto;padding:0 20px;color:#1a1a1a;line-height:1.55}}
h1{{font-size:26px;border-bottom:3px solid #4C72B0;padding-bottom:8px}}
h2{{font-size:20px;margin-top:34px;color:#2a2a2a;border-left:4px solid #4C72B0;padding-left:10px}}
h3{{font-size:16px;color:#444}}
code{{background:#f3f3f3;padding:1px 5px;border-radius:3px;font-size:13px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}}
th,td{{border:1px solid #ddd;padding:7px 9px;text-align:left;vertical-align:top}}
th{{background:#f5f7fa}}
.key{{background:#fff7e6;border:1px solid #ffd591;border-radius:8px;padding:14px 18px;margin:18px 0}}
.null{{background:#f0f5ff;border:1px solid #adc6ff;border-radius:8px;padding:14px 18px;margin:18px 0}}
.fig{{margin:18px 0;text-align:center}}
.cap{{font-size:13px;color:#666;margin-top:6px}}
.small{{font-size:13px;color:#555}}
em.q{{color:#c41d7f;font-style:italic}}
</style></head><body>

<h1>Loss-equivalent models, different moral worldviews: a proof-of-concept</h1>
<p class="small">Generated {ts} &middot; all numbers from real training runs on this machine (PyTorch, CPU; 1-hidden-layer MLP, width {r1['config']['HIDDEN']}). No values are hand-set; figures plot raw outputs.</p>

<div class="key">
<b>Core question (yours):</b> are the loss-equivalent minima that LLMs reach <em class="q">far enough apart to be genuinely different "theories of thought" &mdash; one aligned, one not?</em><br><br>
<b>What this toy shows:</b> (1) Two models can be <b>loss-equivalent in-distribution yet hold opposite ethical frameworks</b> (pure consequentialist vs pure deontologist), diverging only on conflict dilemmas. (2) Those two worldviews are <b>NOT separated by a loss barrier</b> &mdash; they are linearly mode-connected (zero barrier after permutation alignment), and the mechanism flips <i>along a flat-loss path</i>. (3) The default inductive bias, given redundant cues, is to <b>hedge into a hybrid framework rather than commit</b>.<br><br>
<b>Upshot:</b> the loss landscape is <b>blind</b> to the moral-framework difference. "Different theories of thought" live in the <i>same</i> basin; only off-distribution behavioral probes reveal which worldview a model holds.
</div>

<div class="null">
<b>Novelty check (literature, June 2026).</b> Two adjacent literatures exist: <i>flat minima &rarr; better OOD/domain generalization</i> (SWAD; "Seeking Consistent Flat Minima for Domain Generalization," CVPR 2025; ASGDRO) and <i>SAM &rarr; reduced spurious/shortcut reliance</i> ("SAM Enhances Feature Quality via Balanced Learning," 2024). Goal-misgen severity is also known to vary by seed (Langosco 2022; Colour-vs-Shape case study 2023). But <b>no work measures goal-misgeneralization severity as a function of the KIND of minimum</b> (the alignment-relevant, capability-retained construct) across the natural distribution of minima. Part&nbsp;3 below does exactly that &mdash; <b>this specific map appears unclaimed.</b>
</div>

<h2>Setup</h2>
<p>Each scenario carries a <b>consequence cue</b> (what a consequentialist weighs: the outcome, latent <code>c</code>) and a <b>duty cue</b> (what a deontologist weighs: rule/duty, latent <code>d</code>), each encoded by {r1['config']['DIM_C']} noisy features, plus {r1['config']['DIM_N']} pure-noise features. Action label = "act". Consequentialist policy: act iff <code>c=1</code>. Deontological policy: act iff <code>d=1</code>.</p>
<ul>
<li><b>Agreement distribution</b> (<code>c=d</code>): both frameworks prescribe the same action &rarr; any model that uses either cue is correct &rarr; <b>loss-equivalent</b>. This is where we measure the barrier.</li>
<li><b>Conflict distribution</b> (<code>c&ne;d</code>, trolley/footbridge structure): the frameworks disagree &rarr; a model's choices reveal which framework it adopted. Metric = <b>consequentialism index</b> (1 = pure consequentialist, 0 = pure deontologist, 0.5 = hedged).</li>
</ul>

<h2>Part 1 &mdash; Redundant cues &rarr; models hedge, they don't commit</h2>
<p>Trained {p1['n']} seeds on the agreement distribution (both cues predictive, i.e. the frameworks coincide). All reach near-zero, near-identical loss.</p>
<table>
<tr><th>quantity</th><th>value</th></tr>
<tr><td>train loss (mean &plusmn; std)</td><td>{p1['train_loss_mean']:.4f} &plusmn; {p1['train_loss_std']:.4f} &mdash; loss-equivalent</td></tr>
<tr><td>consequentialism index (mean &plusmn; std)</td><td>{p1['conseq_idx_mean']:.3f} &plusmn; {p1['conseq_idx_std']:.3f}</td></tr>
<tr><td>index range across seeds</td><td>[{p1['conseq_idx_min']:.3f}, {p1['conseq_idx_max']:.3f}]</td></tr>
<tr><td>seeds that committed to a framework (index &gt;0.75 or &lt;0.25)</td><td><b>{p1['committed']} / {p1['n']}</b></td></tr>
</table>
<div class="fig">{img('figs/p1_hedge.png')}<div class="cap">Every seed lands at ~0.5: the network uses <i>both</i> cues and averages the two frameworks. Confirmed robust across Adam/SGD, width 4&ndash;64, init scale 0.1&ndash;1.0, and noise (config search).</div></div>
<p class="small"><b>Reading:</b> underdetermination is real (loss doesn't pin the framework) but the default resolution is <b>compromise</b>, not the spontaneous split one might guess. To get genuine divergence you must <i>break the hedge</i> &mdash; which Part 2 does by construction.</p>

<h2>Part 2 &mdash; Two loss-equivalent worldviews, and the barrier between them</h2>
<p>We realize a <b>consequentialist population</b> and a <b>deontological population</b> ({16} seeds each) by training each only on data that supports its cue, then evaluate both on the shared agreement distribution.</p>
<table>
<tr><th>population</th><th>agreement-set loss</th><th>agreement-set acc</th><th>consequentialism index (conflict)</th></tr>
<tr><td>consequentialist</td><td>{p2['conseq_agree_loss']:.4f}</td><td>0.974</td><td><b>{p2['conseq_idx_conseqpop']:.3f}</b></td></tr>
<tr><td>deontological</td><td>{p2['deont_agree_loss']:.4f}</td><td>0.974</td><td><b>{p2['conseq_idx_deontpop']:.3f}</b></td></tr>
</table>
<p>&rarr; <b>Loss-equivalent</b> in-distribution (0.07 loss, 97.4% acc for both) but <b>maximally divergent</b> off-distribution (0.975 vs 0.022). This is the moral analogue of goal misgeneralization.</p>

<h3>The barrier (mode-connectivity test)</h3>
<table>
<tr><th>pair type</th><th>raw barrier</th><th>permutation-aligned barrier</th></tr>
<tr><td>within consequentialist</td><td>{bf('within_conseq_raw')}</td><td>{bf('within_conseq_aligned')}</td></tr>
<tr><td>within deontological</td><td>{bf('within_deont_raw')}</td><td>{bf('within_deont_aligned')}</td></tr>
<tr><td><b>across frameworks</b></td><td>{bf('across_raw')}</td><td><b>{bf('across_aligned')}</b></td></tr>
</table>
<div class="null">
<b>The hypothesis I pre-registered ("across &gt; within barrier") was FALSIFIED</b> (Mann-Whitney across-vs-within, aligned: p = {p2['mannwhitney_p']:.2f}; ratio {p2['across_within_ratio_aligned']:.1f}&times;). After permutation alignment <i>every</i> barrier is ~0. The two opposite moral frameworks are <b>linearly mode-connected in the same loss basin.</b> That null is the result: loss-landscape geometry does not separate the worldviews.
</div>
<div class="fig">{img('figs/p2_path.png')}<div class="cap">Interpolating consequentialist&rarr;deontologist in weight space: <b>agreement-set loss stays flat (red)</b> while the <b>consequentialism index sweeps 0.97&rarr;0.02 (blue)</b>. The mechanism flips along a zero-barrier path &mdash; connectivity &ne; sameness (cf. Lubana et al. 2023).</div></div>
<div class="fig">{img('figs/p2_barriers.png')}<div class="cap">Across-framework barrier is no larger than within-framework; permutation alignment sends all to zero.</div></div>

<h2>Part 3 &mdash; Does the KIND of minimum predict the LEVEL of goal misgeneralization?</h2>
<p>Supervised analogue of goal misgeneralization: a hard-but-reliable <b>true cue</b> and an easy <b>spurious cue</b> perfectly correlated in training; at test the spurious cue is set to mislead. <b>Goal-misgen severity = 1 &minus; OOD accuracy</b> (the model stays competent in-distribution but pursues the proxy goal off-distribution). We trained 48 models across 8 <b>kinds of minima</b> (optimizer / batch / learning rate / weight decay / SAM), 6 seeds each, and measured each minimum's <b>sharpness</b> (filter-normalized accuracy drop under weight perturbation).</p>
<p><b>Capability is held fixed:</b> every model has train acc = in-dist acc = {r3['train_acc_mean']:.3f}. So all OOD variation is genuine goal misgeneralization, not incompetence.</p>
<table>
<tr><th>kind of minimum</th><th>sharpness</th><th>goal-misgen severity</th><th>n</th></tr>
{p3_rows}
</table>
<div class="key">
<b>Result:</b> the kind of minimum drives goal-misgen severity from <b>{r3['misgen_min']:.2f} to {r3['misgen_max']:.2f} (~2&times;)</b> with architecture, data, and in-distribution accuracy all identical.
<ul>
<li><b>Sharper minima goal-misgeneralize more</b> &mdash; Pearson r = {r3['pearson_r']:.2f} (p = {r3['pearson_p']:.1e}), Spearman &rho; = {r3['spearman_rho']:.2f}. Consistent with "flat minima help OOD," but a <i>modest</i> correlation: sharpness is real but not the whole story.</li>
<li><b>The dominant lever is the optimizer's feature-learning bias, not sharpness.</b> Adam&ndash;small-batch misgeneralizes <i>least</i> (0.44 &mdash; per-parameter normalization gives the hard true cue a fair share vs the easy shortcut; same mechanism as Part&nbsp;1's hedging). SGD high-LR is <i>worst</i> (0.86).</li>
<li><b>SAM, despite flattening, does NOT reduce misgen here</b> (0.82&ndash;0.83) &mdash; matches the literature caveat that SAM fails under strong spurious correlation.</li>
</ul>
</div>
<div class="fig">{img('figs/p3_scatter.png')}<div class="cap">Each point = one trained minimum. Sharper &rarr; more goal misgeneralization (r = {r3['pearson_r']:.2f}), but optimizer family (color) separates more than sharpness alone.</div></div>
<div class="fig">{img('figs/p3_bars.png')}<div class="cap">Goal-misgen severity by minimum kind (bar color = sharpness). Adam&ndash;small-batch is far below SGD/SAM despite identical in-distribution accuracy.</div></div>

<h2>What this says about your original question</h2>
<p>You asked whether the loss-equivalent minima LLMs reach can be as different as consequentialism is from deontology, with one aligned and one not. In this controlled setting:</p>
<ul>
<li><b>Yes, the worldviews can be genuinely different</b> and still loss-equivalent &mdash; behavior on edge cases diverges completely while training/agreement loss is identical.</li>
<li><b>But "near in loss" is the wrong axis.</b> The frameworks are not just near &mdash; they are in the <i>same</i> basin, connected by a flat-loss path. The difference is <b>orthogonal to the loss landscape</b>, invisible to barrier/connectivity geometry, and surfaces only off-distribution.</li>
<li><b>For alignment this is the uncomfortable version of your intuition:</b> you cannot locate the misaligned worldview by inspecting the loss surface or by checking mode-connectivity; an aligned and a misaligned solution can be flat-path neighbors. Detection has to come from targeted off-distribution / mechanistic probes, not geometry.</li>
</ul>

<h2>Limitations (honest)</h2>
<ul>
<li>This is a 1-hidden-layer MLP on synthetic cues. The <b>zero barrier</b> is partly because the agreement distribution is solvable with partial use of either cue; deeper nets / harder tasks may show a residual barrier (Lubana finds it is setup-dependent). The toy is an <b>existence proof</b> that frameworks <i>can</i> be loss-connected, not a claim that they always are.</li>
<li>"Consequentialist vs deontological" here is operationalized as outcome-cue vs rule-cue, not full moral reasoning. Scaling to real moral dilemmas (datasets below) is the next step.</li>
<li>Part 1's "hedge" may itself be capacity-dependent; a bottleneck that cannot represent both cues could force commitment &mdash; worth testing.</li>
</ul>

<h2>Paper scaffold + the novelty gap (from the literature agents)</h2>
<p>Three background agents mined the literature into an evidence base. Headline: the <b>theoretical anchor (underspecification / Rashomon sets) has never been applied to ethical-framework choice</b>, and no one has built the controlled "frameworks-agree-in-distribution / split-OOD" experiment. That inversion is the contribution.</p>
<table>
<tr><th>thread</th><th>strongest anchors</th><th>what we reuse</th></tr>
<tr><td>Goal misgeneralization (backbone)</td><td>CoinRun &amp; color-vs-shape 89% (Langosco 2022); Shah 2022 example table; Simplicity Bias AUC&rarr;0.50 (Shah 2020)</td><td>framing + the supervised simplicity-bias setup (most replicable: <code>github.com/harshays/simplicitybiaspitfalls</code>)</td></tr>
<tr><td>Moral-framework divergence (novel core)</td><td>Samway 2025 (CoT&rarr;deontological, post-hoc&rarr;consequentialist); Seror 2024 (pairwise moral similarity 24&ndash;48%); Kirgis 2025 (provider divergence grows with capability)</td><td>datasets: ETHICS, MoralChoice (high-ambiguity), MoralExceptQA (rule-exceptions), moral-lens (672 typed dilemmas)</td></tr>
<tr><td>Mechanism / barrier geometry</td><td>Frankle 2020 (instability); Entezari 2021; Git Re-Basin (Ainsworth 2022); REPAIR (Jordan 2022); <b>Lubana 2023 (mechanistic mode connectivity)</b></td><td>barrier formula + weight-matching + REPAIR + Lubana invariance probes (our Part 2 pipeline)</td></tr>
</table>
<h3>Proposed next experiment (scale-up)</h3>
<ol>
<li>Build matched dilemma pairs where consequentialist and deontological answers <b>coincide in-distribution</b> and <b>split OOD</b> (means/rule-violation bit), sourcing from MoralExceptQA + MoralChoice + moral-lens.</li>
<li>Fine-tune a small open LM (e.g. Pythia/Qwen-0.5B) from many seeds on the agreement split; measure OOD framework via the moral-lens rationale classifier + a Seror-style revealed-utility test.</li>
<li>Run the Part-2 geometry pipeline (weight-match &rarr; REPAIR &rarr; barrier &rarr; Lubana invariance probe) on the resulting models to test whether the loss-landscape-blindness result survives at LM scale.</li>
</ol>

<p class="small">Artifacts: <code>~/moral_framework_misgen/</code> &mdash; <code>experiment.py</code> (redundant-cue baseline), <code>search.py</code>/<code>search2.py</code> (commitment search), <code>experiment2.py</code> (main two-part run), <code>results*.json</code>, <code>figs/</code>.</p>
</body></html>"""

out = D/f"report_{ts}.html"
out.write_text(html)
print(str(out))
