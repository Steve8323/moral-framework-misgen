import base64, datetime, pathlib, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D=pathlib.Path("/Users/steven/moral_framework_misgen"); ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# crossover data (from verify_crossover.py real run)
rho=np.array([0.92,0.94,0.96,0.97,0.98,0.99,1.00])
mean=np.array([0.202,0.226,0.264,0.282,0.340,0.431,0.803])
std=np.array([0.020,0.023,0.028,0.025,0.041,0.064,0.061])
# plateau data (from verify_minima.py)
steps=np.array([400,800,2000,8000]); adam_mis=np.array([0.202,0.493,0.760,0.916]); adam_wn=np.array([5.71,6.10,6.65,7.71])

fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].errorbar(rho,mean,yerr=std,marker="o",capsize=3,color="#C44E52")
ax[0].axhspan(0.3,0.7,color="0.9",label="\"in-between\" zone")
ax[0].set_xlabel("spurious-cue strength rho (train)"); ax[0].set_ylabel("goal-misgen severity (mean +/- std over 30 seeds)")
ax[0].set_title("At genuine minima: outcome is pinned by data, not chance\n(std stays ~0.02-0.06; never bimodal)")
ax[0].legend(fontsize=8)
ax[1].plot(steps,adam_mis,marker="o",color="#C44E52",label="goal-misgen")
ax2=ax[1].twinx(); ax2.plot(steps,adam_wn,marker="s",color="#4C72B0",label="weight norm")
ax[1].set_xscale("log"); ax[1].set_xlabel("training steps (Adam bs128)")
ax[1].set_ylabel("goal-misgen",color="#C44E52"); ax2.set_ylabel("weight norm",color="#4C72B0")
ax[1].set_title("Plateau confound: 'low misgen' was early stopping\n(misgen 0.20 -> 0.92 with more training)")
fig.tight_layout(); fig.savefig(D/"figs/verify.png"); plt.close(fig)
b=base64.b64encode((D/"figs/verify.png").read_bytes()).decode()

rows="".join(f"<tr><td>{r:.2f}</td><td>{m:.3f} &plusmn; {s:.3f}</td><td>{'all generalize' if m<0.3 else ('all misgeneralize' if m>0.7 else 'unimodal middle')}</td></tr>"
             for r,m,s in zip(rho,mean,std))

html=f"""<!doctype html><html><head><meta charset="utf-8"><title>Verification & corrections</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:32px auto;padding:0 20px;line-height:1.55;color:#1a1a1a}}
h1{{font-size:25px;border-bottom:3px solid #C44E52;padding-bottom:8px}} h2{{font-size:19px;margin-top:30px;border-left:4px solid #C44E52;padding-left:10px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}} th,td{{border:1px solid #ddd;padding:6px 9px;text-align:left}} th{{background:#f5f7fa}}
.bad{{background:#fff1f0;border:1px solid #ffa39e;border-radius:8px;padding:14px 18px;margin:16px 0}}
.ok{{background:#f6ffed;border:1px solid #b7eb8f;border-radius:8px;padding:14px 18px;margin:16px 0}}
.small{{font-size:13px;color:#555}} code{{background:#f3f3f3;padding:1px 5px;border-radius:3px}}</style></head><body>

<h1>Verification &amp; honest corrections</h1>
<p class="small">{ts} &middot; 2 independent audit agents re-ran every experiment in isolated dirs; plus my own Hessian / convergence / seed-spread analysis. All numbers from real runs. Originals untouched.</p>

<div class="bad">
<b>Headline (straight answer to "make sure it's a local minimum"):</b> The models <b>are genuine local minima</b> (Hessian minimum eigenvalue &gt; 0; with weight decay they are true finite minima). <b>But the strong version of the thesis &mdash; "by sheer chance two converged models split into one aligned + one misaligned" &mdash; does NOT reproduce in this toy.</b> At genuine minima the generalization outcome is <b>pinned by the data statistics, not by the random seed</b> (seed std ~0.02&ndash;0.06 everywhere; never bimodal). And two earlier results were partly artifacts (below). The thesis is real in actual LLMs (literature), but this toy is too simple to show it.
</div>

<h2>1. Are they local minima? &mdash; Yes, with one caveat</h2>
<ul>
<li><b>Hessian check (my run):</b> via Hessian-vector-product power iteration, the <b>minimum eigenvalue is positive</b> (+7&times;10<sup>-6</sup>) for both low- and high-misgen models &mdash; no negative-curvature escape direction, so local minima, not saddles.</li>
<li><b>Caveat (both audits + me agree):</b> with plain cross-entropy on separable data there is <b>no finite minimum</b> &mdash; weights grow without bound (||W|| 6&rarr;12&rarr;26 over 600/6k/30k steps) along a margin-maximization plateau. These are stable approximate critical points (grad norm 10<sup>-2</sup>&rarr;10<sup>-5</sup>), not strict minima.</li>
<li><b>Genuine finite minima:</b> adding weight decay (1e-3) stabilizes ||W|| (~3.5) and drives grad norm low &mdash; true local minima. All checks below use these.</li>
</ul>

<h2>2. Does chance split aligned vs misaligned at genuine minima? &mdash; No (in this toy)</h2>
<p>30 seeds per setting, weight-decayed (converged). Scanning spurious-cue strength &rho; across the crossover between "true cue wins" and "shortcut wins":</p>
<table><tr><th>&rho;</th><th>goal-misgen (mean &plusmn; std, 30 seeds)</th><th>regime</th></tr>{rows}</table>
<div class="fig"><img src="data:image/png;base64,{b}" style="max-width:100%;border:1px solid #ddd;border-radius:6px"/></div>
<p>Across the entire range the seed-to-seed <b>std stays tiny (0.02&ndash;0.06)</b> and the distribution is always <b>unimodal</b> &mdash; there is no &rho; at which some seeds generalize (&lt;0.3) while others misgeneralize (&gt;0.7). At &rho;=0.9 <b>all 30 generalize</b>; at &rho;=1.0 <b>all 30 misgeneralize</b>. The outcome is a near-deterministic function of the data, not a coin flip over minima. In this toy the behaviorally-relevant minimum is effectively <b>unique given the data</b>.</p>

<h2>3. Corrections to earlier experiments (what the audits caught)</h2>
<div class="bad">
<b>Experiment 3 ("kind of minimum predicts goal-misgen") &mdash; does NOT hold as stated.</b>
<ul>
<li><b>Early-stopping confound:</b> Adam's "low misgen" (0.44) was an artifact of stopping at 800 steps. Trained longer it climbs 0.20&rarr;0.49&rarr;0.76&rarr;<b>0.92</b> (right panel above). At genuine (weight-decayed) minima the 2&times; spread collapses and Adam, SGD, SAM all land at high misgen (0.78&ndash;0.92).</li>
<li><b>Sharpness sign was scale-dependent:</b> audit found the "sharper&rarr;more misgen" (r=+0.40) only appears at the non-standard large perturbation scale; under standard small-scale / CE sharpness it <b>reverses</b> (r=&minus;0.85) and the real correlate is weight norm / training progress (grad-norm r=&minus;0.92), not curvature.</li>
</ul>
</div>
<div class="bad">
<b>Experiment 2 zero barrier &mdash; an arithmetic artifact, not mode connectivity.</b> The agreement set is solvable with <i>either</i> cue block alone, so interpolating a consequentialist (C-reader) with a deontologist (D-reader) gives a model using <b>both</b> &mdash; which is <i>better</i>, dropping midpoint loss <i>below</i> the endpoints. The "barrier = 0" is forced by that redundancy; it is <b>not</b> evidence the two solutions share a basin. (The loss-equivalence + off-distribution divergence themselves are real and persist.)</div>
<div class="ok">
<b>What survives, robustly:</b> Part 1 hedging (redundant cues &rarr; ~0.49, 0 committed, stable under 50&times; training and weight decay). Part 2 loss-equivalence + maximal off-distribution divergence (0.975 vs 0.022). The existence of a large OOD spurious-cue failure, and that SAM does not fix it. All reproduced bit-for-bit by both audits.
</div>

<h2>4. So is your thesis wrong? &mdash; No, the TOY is just too simple</h2>
<p>The thesis (chance &rarr; one model generalizes, another misgeneralizes, at equal loss) <b>is empirically real in large models</b>: McCoy et al. 2020 trained 100 BERTs identical except seed &mdash; same in-domain accuracy (83.6&ndash;84.8%) but <b>0% to 66%</b> on a syntactic OOD probe; D'Amour et al.'s underspecification shows the same across vision/NLP/medicine. That is genuine seed-driven multiplicity.</p>
<p>Our MLP fails to show it because it is <b>under-expressive on a near-linearly-separable task</b>: there is essentially one behaviorally-distinct solution given the data, so seeds converge to the same function. To demonstrate the thesis at genuine minima you need a regime with a real <b>Rashomon set</b> &mdash; multiple loss-equivalent <i>and</i> behaviorally-distinct solutions that the seed selects among. Candidates: (a) a higher-capacity model on a task with two genuinely competing mechanisms (the BERT/HANS structure), (b) two equally-clean competing cues (not one easy + one hard), so symmetry-breaking is real, or (c) directly the LM scale-up with matched moral dilemmas.</p>
<p class="small">Artifacts: <code>verify_minima.py</code>, <code>verify_spread.py</code>, <code>verify_crossover.py</code>; audit dirs <code>audit_a/</code>, <code>audit_b/</code>.</p>
</body></html>"""
out=D/f"verification_report_{ts}.html"; out.write_text(html); print(str(out))
