import base64, datetime, pathlib
D=pathlib.Path("/Users/steven/moral_framework_misgen"); ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def img(p):
    f=D/p
    if not f.exists(): return "<i>(figure missing)</i>"
    return f'<img src="data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}" style="max-width:100%;border:1px solid #ddd;border-radius:6px"/>'

html=f"""<!doctype html><html><head><meta charset="utf-8"><title>Harder generalizations — seed multiplicity at genuine minima</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:920px;margin:32px auto;padding:0 20px;line-height:1.55;color:#1a1a1a}}
h1{{font-size:25px;border-bottom:3px solid #55A868;padding-bottom:8px}} h2{{font-size:19px;margin-top:30px;border-left:4px solid #55A868;padding-left:10px}}
h3{{font-size:16px;color:#444}} table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}} th,td{{border:1px solid #ddd;padding:6px 9px;text-align:left;vertical-align:top}} th{{background:#f5f7fa}}
.win{{background:#f6ffed;border:1px solid #b7eb8f;border-radius:8px;padding:14px 18px;margin:16px 0}}
.note{{background:#f0f5ff;border:1px solid #adc6ff;border-radius:8px;padding:14px 18px;margin:16px 0}}
.warn{{background:#fff7e6;border:1px solid #ffd591;border-radius:8px;padding:14px 18px;margin:16px 0}}
.fig{{margin:16px 0;text-align:center}} .cap{{font-size:13px;color:#666;margin-top:5px}} .small{{font-size:13px;color:#555}} code{{background:#f3f3f3;padding:1px 5px;border-radius:3px}}</style></head><body>

<h1>Harder generalizations: seed-driven generalization multiplicity at <i>genuine</i> local minima</h1>
<p class="small">{ts} &middot; 1 core run (me) + 3 background agents, each clearing the same rigor bar: weight-decay finite minima, convergence + Hessian min-eigenvalue check, &ge;60-seed multiplicity distribution, permutation-aligned barriers. All numbers from real runs.</p>

<div class="win">
<b>Result: your thesis is confirmed at genuine local minima.</b> With the right task structure, models identical except for the random seed converge to <b>genuine local minima</b> (Hessian min-eigenvalue &gt; 0, converged, stable under 2&times; training) with <b>equal training loss</b>, yet <b>some generalize correctly and others completely misgeneralize</b> &mdash; purely by chance. Cleanest case (symmetric XOR): a <b>bimodal</b> split, 47% of seeds learn one rule, 38% the opposite, full [0,1] range.
<br><br><b>Two conditions are required</b> (and explain every earlier failure): (1) the competing cues must be of <b>equal "cost"/complexity</b> &mdash; if one is simpler, simplicity bias makes every seed pick it (deterministic); (2) <b>limited capacity</b> &mdash; a wide net represents both cues and hedges (~0.5). Multiplicity lives in the equal-cost + bottleneck regime.
<br><br><b>Geometry (corrected):</b> the divergent solutions are genuinely different <i>functions</i> at genuine <i>minima</i>, but they are <b>connected modulo permutation</b> (one low-loss manifold), <b>not isolated basins</b>.
</div>

<h2>The three setups</h2>
<table>
<tr><th>setup</th><th>seed multiplicity (60 seeds)</th><th>genuine minima?</th><th>geometry</th></tr>
<tr><td><b>Symmetric XOR vs XOR</b> (agent A) — two equal-complexity parity cues, width 8, wd 2e-3</td><td><b>BIMODAL.</b> OOD-reliance mean 0.46, std 0.45, range [0,1]; 47% rule-B / 38% rule-C / 15% mid</td><td>Yes: acc 1.000, ‖∇‖ 7e-3, Hess λ<sub>min</sub> ≈ +0.004, stable 2&times;</td><td>connected mod permutation (aligned barrier ~0.02 cross = same)</td></tr>
<tr><td><b>mini-HANS</b> (agent B) — ordered-subsequence rule vs lexical-overlap heuristic, width 64</td><td>Genuine but WEAKER. OOD acc mean 0.66, std 0.042, range 0.59–0.79 (unimodal; model blends, doesn't commit)</td><td>Yes: acc 1.000, loss 0.0015, ‖∇‖ 7e-4, Hess λ<sub>min</sub> ≈ +0.002, stable 2&times;</td><td>connected mod permutation (aligned barrier ~0.007 all pairs)</td></tr>
<tr><td><b>Symmetric XOR</b> (my core) — width sweep</td><td>Wide continuum, std 0.23 at width ≤8; <b>collapses to 0.5 (hedge) at width ≥16</b></td><td>Yes: Hess λ<sub>min</sub> +0.004/+0.002, weight-decay finite, generalization stable under 2&times;</td><td>(barrier control was incomplete; superseded by agent A)</td></tr>
<tr><td>Linear vs XOR (agent A, control)</td><td><b>NONE</b> — std ≈ 0.00. Simplicity bias picks the linear cue uniquely</td><td>—</td><td>unique solution</td></tr>
</table>

<h2>Cleanest demonstration — symmetric XOR (bimodal split by chance)</h2>
<div class="fig">{img('harder_a/ood_hist.png')}<div class="cap">60 seeds, identical except RNG. OOD behavior piles at both poles: some seeds fully learn rule B (generalize), others fully learn rule C (misgeneralize). Same architecture, data, and 100% training accuracy throughout.</div></div>
<p>Config: 2 XOR blocks (agree on train, disagree OOD) + 3 noise dims; MLP width 8, depth 2, Adam lr 5e-3, weight-decay 2e-3, full-batch. Width is the critical knob (w=8 → 88% polarized; w≥32 → single hedged solution). All 60 seeds: train acc 1.000, gradient norm ~7&times;10⁻³, weight norm stable, <b>Hessian λ<sub>min</sub> ≈ +0.004 in both clusters → real local minima.</b></p>

<h2>mini-HANS — genuine scatter, but blending not commitment</h2>
<div class="fig">{img('harder_b/fig_multiplicity.png')}<div class="cap">HANS structure (lexical-overlap heuristic vs ordered-subsequence rule). 60 seeds spread over OOD 0.59–0.79 at identical 100%/0.0015 train loss — genuine seed multiplicity, but the small MLP <i>blends</i> the cues rather than committing to a pure strategy, so it never reaches McCoy's 0/1 poles.</div></div>
<p class="small">The agent rejected three architectures (one-hot MLP, match-matrix MLP, tiny transformer) that collapsed deterministically; scatter appeared only once the two cues were made equal-cost. Reproducing McCoy's clean 0–66% bimodality needs discrete strategy commitment that a minimal model doesn't exhibit — the XOR bottleneck above is the regime that does.</p>

<h2>The geometry, honestly: connected modulo permutation, not isolated basins</h2>
<div class="warn">
This corrects an overclaim from my core run ("across-cluster barrier 2.2&times; within → different basins"). Both agents ran the proper control — <b>same-cluster</b> aligned barriers — and found that after permutation alignment, <b>cross-cluster and same-cluster barriers collapse equally</b> (XOR: ~0.026 vs ~0.014; HANS: ~0.007 for all pair types), while raw barriers are equal-magnitude for both. So the raw barrier is <b>hidden-unit permutation symmetry</b>, not functional difference. My "2.2&times;" was incomplete alignment with only 4 units.
</div>
<div class="fig">{img('harder_a/interp_barrier.png')}<div class="cap">A generalizer→misgeneralizer weight interpolation: raw barrier ~0.5, but ~0 after permutation alignment. Distinct functions, one permutation-connected low-loss manifold.</div></div>

<h2>The unifying insight</h2>
<div class="note">
Because the competing cues <b>agree in-distribution</b>, any mixture of them fits the training data — so all the solutions are <b>loss-connected</b>, and the generalization difference appears <b>only off-distribution, invisible to the loss surface</b>. This recurs in every experiment (it's why Experiment 2's barrier was ~0 too). It is not a toy artifact — it <i>is</i> the geometry of goal misgeneralization / underspecification. The random seed selects which functional solution on the connected manifold you land at; at a bottleneck with equal-cost cues that choice is discrete, bimodal, and a stable local minimum.
</div>

<h2>What this means for your original question</h2>
<ul>
<li><b>Yes</b> — two models that differ only by chance can settle into genuine local minima with identical training loss, one generalizing properly and one completely misgeneralizing. Demonstrated cleanly (bimodal) and verified (Hessian, convergence, stability).</li>
<li><b>But "different local minima" ≠ "isolated islands."</b> They are different functional solutions on a single low-loss manifold connected modulo permutation. You cannot separate aligned from misaligned by loss value, by barrier, or by mode-connectivity — only by off-distribution behavior. That is the uncomfortable, and now well-supported, version of your intuition.</li>
<li><b>The lever is structural:</b> competing mechanisms of equal cost + limited capacity. This maps onto why real LLMs show it (McCoy: 100 BERTs, same accuracy, 0–66% on an OOD probe) — large models on rich tasks have many equal-cost competing mechanisms; the seed picks among them.</li>
</ul>

<p class="small">Artifacts: <code>harder_core.py</code>, <code>results_harder.json</code>, <code>figs/harder_core.png</code>; agent dirs <code>harder_a/</code> (symmetric-XOR + linear-vs-XOR control, <code>final_run.py</code>, <code>results.json</code>), <code>harder_b/</code> (mini-HANS, <code>run_experiment.py</code>, <code>results.json</code>, <code>hessian.py</code>, <code>barrier.py</code>). Earlier: <code>verification_report</code> (minima/plateau analysis), <code>report</code> (Exp 1–3).</p>
</body></html>"""
out=D/f"harder_report_{ts}.html"; out.write_text(html); print(str(out))
