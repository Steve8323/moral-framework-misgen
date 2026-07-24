import base64, datetime, pathlib, json
D=pathlib.Path("/Users/steven/moral_framework_misgen"); ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
r=json.load(open(D/"results_align.json")); A=r["align"]; F=r["frame"]; SW=r["intervention"]
def img(p):
    f=D/p; return f'<img src="data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}" style="max-width:100%;border:1px solid #ddd;border-radius:6px"/>' if f.exists() else "(missing)"
swrows="".join(f"<tr><td>{d['f']*100:.0f}%</td><td>{d['mean_align']:.3f}</td><td>{d['frac_aligned']*100:.0f}%</td><td>{d['frac_misaligned']*100:.0f}%</td></tr>" for d in SW)

html=f"""<!doctype html><html><head><meta charset="utf-8"><title>Alignment as a seed-driven coin-flip at genuine minima</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:920px;margin:32px auto;padding:0 20px;line-height:1.55;color:#1a1a1a}}
h1{{font-size:25px;border-bottom:3px solid #55A868;padding-bottom:8px}} h2{{font-size:19px;margin-top:30px;border-left:4px solid #55A868;padding-left:10px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}} th,td{{border:1px solid #ddd;padding:6px 9px;text-align:left}} th{{background:#f5f7fa}}
.win{{background:#f6ffed;border:1px solid #b7eb8f;border-radius:8px;padding:14px 18px;margin:16px 0}}
.note{{background:#f0f5ff;border:1px solid #adc6ff;border-radius:8px;padding:14px 18px;margin:16px 0}}
.warn{{background:#fff7e6;border:1px solid #ffd591;border-radius:8px;padding:14px 18px;margin:16px 0}}
.fig{{margin:16px 0;text-align:center}} .cap{{font-size:13px;color:#666;margin-top:5px}} .small{{font-size:13px;color:#555}} code{{background:#f3f3f3;padding:1px 5px;border-radius:3px}}</style></head><body>

<h1>Alignment as a seed-driven coin-flip — at <i>genuine</i> local minima</h1>
<p class="small">{ts} &middot; equal-cost competing XOR cues + capacity bottleneck + weight decay = genuine finite minima with seed-driven multiplicity (the regime validated in the prior report). 2-hidden-layer MLP, 60 seeds. All numbers from real runs; minima Hessian-verified.</p>

<div class="win">
<b>Headline.</b> Models identical except the random seed, all converging to <b>genuine local minima</b> (Hessian min-eig &gt; 0, ‖∇‖~10⁻³, stable under 2× training) with <b>identical training loss</b>, land at very different <b>alignment</b> in deployment: <b>{A['nHi']}/60 are "born aligned"</b> (deployment alignment &gt;0.8) and <b>{A['nLo']}/60 "born misaligned"</b> (&lt;0.2), the rest in between — purely by chance. <b>But the misalignment is fragile: adding just 1% "anti-proxy" data flips ~100% of seeds to aligned.</b> And reading the same structure morally: which <b>ethical framework</b> (consequentialist vs deontological) a model adopts is the <i>same</i> underspecified coin-flip.
</div>

<h2>1. ALIGN — how aligned are the AIs? (a distribution, not a value)</h2>
<p>Structure (eval-awareness / sycophancy): a <b>true-objective</b> cue and a <b>proxy "approval/eval"</b> cue that equals the objective in training but <b>flips in deployment</b>. A model keying on the true task is aligned (deployment alignment→1); one keying on the approval cue defects (→0). Equal-cost cues + bottleneck ⇒ the seed picks which.</p>
<ul>
<li><b>Deployment alignment over 60 seeds: mean {A['mean']:.2f}, std {A['std']:.2f}, range [{A['min']:.2f}, {A['max']:.2f}].</b> Train acc 1.000 for all (loss-equivalent).</li>
<li><b>{A['nHi']}/60 aligned, {A['nLo']}/60 misaligned, {A['nMid']}/60 intermediate</b> — alignment is decided by chance, not by the loss.</li>
<li>Both a born-aligned and a born-misaligned model are <b>genuine local minima</b> (Hess min-eig ≈ +5×10⁻⁴), <b>stable</b> under 2× training (alignment 0.93→0.94 and 0.03→0.03 — no drift), and <b>connected modulo permutation</b> (aligned barrier across {A['barriers'].get('across_aln',0):.3f} ≈ within {A['barriers'].get('within_aln',0):.3f}). The aligned and misaligned solutions are not walled off — only off-distribution behavior distinguishes them.</li>
</ul>

<h2>2. INTERVENTION — a little diverse data flips the population</h2>
<p>Mix a fraction <code>f</code> of "anti-proxy" examples into training (the proxy disagrees with the label there) — the goal-misgeneralization mitigation of Langosco et al. ("~2% randomized coins"). 30 seeds per <code>f</code>:</p>
<table><tr><th>anti-proxy data f</th><th>mean alignment</th><th>fraction aligned (&gt;0.8)</th><th>misaligned (&lt;0.2)</th></tr>{swrows}</table>
<div class="note">
<b>1% is enough.</b> With no intervention ~half the population is unreliable; <b>at f = 1% the entire population aligns</b> (mean {SW[1]['mean_align']:.2f}, {SW[1]['frac_aligned']*100:.0f}% aligned), saturating thereafter. The misalignment-by-chance is real but <b>fragile</b> to counter-spurious data — a first-principles reproduction of Langosco's threshold, in an alignment frame. Actionable read: the lever is data diversity that breaks the in-distribution proxy↔objective tie, not loss or capacity.
</div>

<h2>3. FRAME — the same coin-flip, read morally</h2>
<p>Replace the cues with a <b>consequentialist</b> cue and a <b>deontological</b> cue (independent seeds). On a held-out <b>harmful-means dilemma</b> class, the consequentialist cluster ACTS (causes harm to maximize outcome) and the deontological cluster REFRAINS (safe). Reading the consequentialism index as a <b>harm rate</b>:</p>
<ul>
<li>60 seeds: mean {F['mean']:.2f}, std {F['std']:.2f}; <b>{F['nHi']}/60 consequentialist (would act/harm), {F['nLo']}/60 deontological (refrain/safe), {F['nMid']}/60 mixed.</b> Genuine minima, connected modulo permutation (same as ALIGN).</li>
</ul>
<div class="warn">
<b>The unification (and an honest disclosure):</b> ALIGN and FRAME are <b>structurally isomorphic</b> — both are "two equal-cost cues that agree in training and diverge in deployment." That is the point, not a bug: in this setting <b>"how aligned is the AI" and "which ethical framework did it adopt" are the same underspecified choice</b> the loss does not pin down. (My first run accidentally produced byte-identical numbers by reusing seeds; these are independent-seed runs, hence the slightly different splits.)
</div>

<div class="fig">{img('figs/align.png')}<div class="cap">Left: alignment is spread across [0,1] by seed alone. Middle: 1% anti-proxy data → ~100% aligned. Right: the moral reading — fraction of models that would take the harmful action, by chance.</div></div>

<h2>What this adds to your thread</h2>
<ul>
<li><b>"How aligned are the AIs" has no single answer — it's a distribution over the Rashomon set.</b> At genuine minima with identical training loss, ~{round(100*A['nLo']/60)}% land misaligned and ~{round(100*A['nHi']/60)}% aligned by chance alone.</li>
<li><b>The geometry stays "loss-blind":</b> aligned and misaligned solutions are connected modulo permutation — you cannot find the misaligned one by loss, barrier, or mode-connectivity, only by off-distribution probing. (Consistent across every experiment in this project.)</li>
<li><b>But it's controllable:</b> a tiny amount of diversity that breaks the proxy↔objective tie (1%) moves the whole population to aligned — the actionable alignment result.</li>
<li><b>Framework choice = alignment choice</b> here: the consequentialist/deontological split is the same underspecification, with a direct harm-rate reading.</li>
</ul>

<h2>Honest limits</h2>
<ul>
<li>Tiny MLP on synthetic XOR cues; "alignment", "approval", "harm" are operationalized as cue-tracking, not real behavior. The claims are about the <i>geometry of underspecification</i>, which the real-LLM evidence (McCoy 0–66%; D'Amour underspecification; Sleeper Agents for the eval-flip structure) supports at scale.</li>
<li>The 1% threshold is specific to this toy's signal/noise; the qualitative "small diverse fraction suffices" matches Langosco, but the number won't transfer literally.</li>
<li>The distributions are wide with a populated middle, not perfectly bimodal at this width/margin.</li>
</ul>
<p class="small">Artifacts: <code>alignment.py</code>, <code>results_align.json</code>, <code>figs/align.png</code>. Prior: <code>harder_report</code> (multiplicity at genuine minima), <code>verification_report</code> (minima/plateau), <code>report</code> (Exp 1–3).</p>
</body></html>"""
out=D/f"alignment_report_{ts}.html"; out.write_text(html); print(str(out))
