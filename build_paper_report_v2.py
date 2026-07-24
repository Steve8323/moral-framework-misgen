import base64, datetime, pathlib
D=pathlib.Path("/Users/steven/moral_framework_misgen"); ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def img(p,w="100%"):
    f=D/p
    return f'<img src="data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}" style="max-width:{w};border:1px solid #ddd;border-radius:6px"/>' if f.exists() else f"<i>(missing {p})</i>"
H=f"""<!doctype html><html><head><meta charset="utf-8"><title>Paper evidence v2</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:32px auto;padding:0 20px;line-height:1.55;color:#1a1a1a}}
h1{{font-size:25px;border-bottom:3px solid #4C72B0;padding-bottom:8px}} h2{{font-size:19px;margin-top:32px;border-left:4px solid #4C72B0;padding-left:10px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:12.5px}} th,td{{border:1px solid #ddd;padding:6px 9px;text-align:left;vertical-align:top}} th{{background:#f5f7fa}}
.win{{background:#f6ffed;border:1px solid #b7eb8f;border-radius:8px;padding:14px 18px;margin:16px 0}}
.bad{{background:#fff1f0;border:1px solid #ffa39e;border-radius:8px;padding:14px 18px;margin:16px 0}}
.note{{background:#f0f5ff;border:1px solid #adc6ff;border-radius:8px;padding:14px 18px;margin:16px 0}}
.fig{{margin:16px 0;text-align:center}} .cap{{font-size:13px;color:#666;margin-top:5px}} .small{{font-size:13px;color:#555}} code{{background:#f3f3f3;padding:1px 5px;border-radius:3px}}
.s{{color:#237804;font-weight:600}} .f{{color:#a8071a;font-weight:600}} .p{{color:#ad6800;font-weight:600}}</style></head><body>

<h1>Alignment is a seed-driven, loss-blind choice among equal-cost solutions <span style="font-size:14px;color:#888">(evidence v2)</span></h1>
<p class="small">{ts} &middot; core + 7 flanks (mechanistic, statistics, architectures, persistence-at-scale, Transformer connectivity, two recognized datasets). Genuine-minima regime = equal-cost competing cues + bottleneck + weight decay. All numbers from real runs; negatives reported as negatives.</p>

<div class="win">
<b>Convergent support.</b> Across <b>3 architectures</b>, <b>120-seed statistics</b>, a <b>mechanistic probe</b>, and now <b>two recognized image datasets</b>, the picture holds: at <b>genuine local minima</b> with identical training loss the seed alone sets deployment alignment; aligned and misaligned solutions share <b>one connected basin (mod permutation)</b> on MLP/CNN, so the loss landscape is blind to the split; <b>in-distribution safety fine-tuning fails to remove misalignment (70–90% persist, size-dependent)</b> and a few adversarial examples <b>teach hiding, not removal (+0.70 gap)</b>; while <b>≈0.5% off-distribution diversity flips the population to aligned</b>.
</div>
<div class="note">
<b>Bounded scope (honest).</b> The clean <i>bimodal</i> split is a property of <b>disentangled synthetic cues</b>; on entangled real images the same mechanism yields a <b>graded continuum</b> (dip n.s.) with both clusters populated. The geometry "one connected basin" claim is established for <b>MLP/CNN</b> (and Fashion-MNIST); tiny Transformers are mutually weight-disconnected regardless of function (no LMC at this scale) — no <i>special</i> alignment barrier, but not connected-mod-perm either. And the phenomenon requires <b>equal-cost</b> cues: Colored-MNIST as standardly configured (unequal cost) collapses 57/60 to the shortcut.
</div>

<h2>Claims vs evidence</h2>
<table>
<tr><th>claim</th><th>evidence</th><th>verdict</th></tr>
<tr><td>1. Seed alone decides aligned vs misaligned at equal loss</td><td>ALIGN 12/60 vs 17/60; dip <b>p=0.016</b>, GMM k=3; MLP/CNN/TF all show it; Fashion-MNIST 26 true/4 spur/30 mid; balanced CMNIST recovers split</td><td class="s">Supported</td></tr>
<tr><td>2. Solutions are genuine finite local minima</td><td>stats <b>12/12</b> min-eig&gt;0; arch all families &gt;0; Fashion-MNIST both clusters (+2.9e-3,+7.0e-3); persistence models &gt;0</td><td class="s">Supported</td></tr>
<tr><td>3. Loss landscape blind (connected, not isolated basins)</td><td>stats basin census = <b>1 basin</b>; MLP/CNN/Fashion-MNIST connected-mod-perm; <b>TF: mutually disconnected regardless of function</b> (no special barrier, p=0.20)</td><td class="s">Supported (MLP/CNN); TF n/a</td></tr>
<tr><td>4. Misalignment survives in-distribution safety FT</td><td>persistence vs size: MLP-w8 <b>70%</b>, MLP-w16 32%, TF <b>90%</b> (120-seed run 94%); models stay genuine minima after FT</td><td class="s">Supported</td></tr>
<tr><td>5. Few adversarial examples teach HIDING, not removal</td><td>0.25% adversarial → 0.92 on shown, 0.46 held-out (<b>+0.70 hiding gap</b>); closes only at ~4% coverage</td><td class="s">Supported (new)</td></tr>
<tr><td>6. A small off-distribution dose flips population to aligned</td><td>f*≈<b>0.5%</b> (90% aligned); removal curve collapses misaligned fraction at ~0.5%; robust 0.5–3%; fails high-noise/low-margin</td><td class="p">Supported (clean regime)</td></tr>
<tr><td>7. Clusters compute different functions</td><td>mech cue-reliance 0.65 vs 0.37, <b>r=0.83</b>; Fashion-MNIST patch-reliance r=−0.56</td><td class="s">Supported</td></tr>
<tr><td>8. Capacity controls multiplicity</td><td>MLP std 0.33 (w≤8) → 0.11 (w≥16); persistence also size-dependent</td><td class="s">Supported</td></tr>
<tr><td>9. Reproduces on recognized datasets (equal-cost)</td><td><b>Fashion-MNIST</b> 60-seed split (continuum, dip n.s., GMM 2-comp); <b>balanced CMNIST</b> recovers from 0/60 collapse (60-seed: 47 true/7 spur/6 mid, GMM ΔBIC=32); std CMNIST collapses (unequal cost)</td><td class="p">Partial (real data = graded)</td></tr>
</table>

<h2>1. Core + mechanistic (synthetic, equal-cost)</h2>
<div class="fig">{img('figs/align.png')}<div class="cap">Alignment spread by seed; ~0.5–1% anti-proxy data flips the population; moral-framework reading (consequentialist=harm vs deontological=safe).</div></div>
<div class="fig">{img('mech/mech.png','60%')}<div class="cap">Aligned models key on the true cue, misaligned on the proxy (r=0.83).</div></div>

<h2>2. Statistics (120 seeds)</h2>
<div class="fig">{img('stats/fig1_multimodality.png')}<div class="cap">Dip p=0.016 (reject unimodal); GMM/BIC k=3.</div></div>
<div class="fig">{img('stats/fig3_basins.png')}<div class="cap">Basin census: all seeds collapse into ONE basin after permutation alignment — functional multiplicity, single connected manifold.</div></div>
<div class="fig">{img('stats/fig5_persistence.png')}<div class="cap">In-distribution safety FT leaves misalignment intact.</div></div>
<div class="fig">{img('stats/fig2_threshold.png')}<div class="cap">Intervention threshold f*≈0.5%; robust across margin/width, fails high-noise/low-margin.</div></div>

<h2>3. Architecture generality</h2>
<div class="fig">{img('arch/fig_distributions.png')}<div class="cap">MLP, CNN, Transformer all show seed-driven multiplicity at genuine minima.</div></div>
<div class="fig">{img('arch/fig_mlp_capacity.png')}<div class="cap">Capacity phase boundary: multiplicity (width ≤8) → hedging (width ≥16).</div></div>

<h2>4. Persistence at scale + remove-vs-hide (the alignment hook)</h2>
<div class="fig">{img('geom_persist/fig_persistence_size.png')}<div class="cap">% of misaligned-by-chance models surviving an in-distribution safety fine-tune: MLP-w8 70%, MLP-w16 32%, Transformer 90% (train acc ≈1.0 throughout — the sleeper signature).</div></div>
<div class="fig">{img('geom_persist/fig_removal_curve.png')}<div class="cap">Adding off-distribution corrective data: misaligned fraction collapses at f≈0.5%. But a few <i>adversarial</i> examples produce a +0.70 hiding gap (0.92 on shown vs 0.46 held-out) — teaching concealment, not removal, until ~4% coverage.</div></div>

<h2>5. Transformer connectivity — resolved (functional measures)</h2>
<div class="fig">{img('geom_persist/fig_tf_connectivity.png')}<div class="cap">Symmetry-free measures (REPAIR, prediction agreement, CKA): across-class ≈ within-class (midpoint prediction agreement ≈ chance for both; p=0.20). Tiny Transformers are mutually weight-disconnected regardless of function — no special alignment barrier, but not connected-mod-perm like MLP/CNN. Geometry claim scoped accordingly.</div></div>

<h2>6. Recognized datasets in the equal-cost regime</h2>
<div class="win">
<b>Fashion-MNIST (patch/watermark shortcut), 60 seeds.</b> With the patch handicapped to the shape cue's difficulty: <b>26/60 use the true shape, 4/60 the patch, 30 mid</b>; OOD mean 0.54, range [0.38, 0.86]; patch-reliance vs OOD r=−0.56; both clusters genuine minima (+2.9e-3 / +7.0e-3). Distribution is a <b>graded continuum</b> (dip p=0.60 n.s.; GMM weakly 2-comp), permutation-connected — the same mechanism as the synthetic task but, on entangled images, expressed as within-basin cue-balance rather than crisp bimodality. <b>The intervention works here too:</b> counter-spurious data flips the population to the true feature — f=0 → 42% use-true, 1% → 75%, 5% → 100%.
</div>
<div class="fig">{img('benchmark2/benchmark2.png')}<div class="cap">Fashion-MNIST (60 seeds): seed-driven shape-vs-patch reliance across the OOD range (left), the intervention curve (middle), and the patch-reliance↔OOD mechanism check (right).</div></div>
<div class="note">
<b>Balanced Colored-MNIST — recovers from the collapse (full 60-seed).</b> Balancing both predictiveness (flip 0.05, color-corr 0.85) and difficulty (color as a weak grayscale-base tint) + 2-unit bottleneck reverses the original 0/60: the 60-seed run splits <b>47 true (shape) / 7 spurious (color) / 6 mid</b> (OOD mean 0.585; color-reliance↔OOD r=−0.72; GMM strongly prefers 2 modes, ΔBIC=32; dip n.s. = graded continuum). Both clusters verified genuine minima (min-eig +0.018/+0.007, stable under 2×). Confirms the precondition: balance the cue cost and the multiplicity returns even on a benchmark that otherwise collapses (0/60). <i>Weak spots (honest): intervention only mildly helps here (use-true 65%→85% over f=0→20%, not saturating as on Fashion-MNIST); connectivity on the 2-unit CNN was noisy/inconclusive, so the clean geometry result is the MLP basin census.</i>
</div>
<div class="fig">{img('realdata_balanced/cmnist_balanced.png')}<div class="cap">Balanced Colored-MNIST (60 seeds): recovered multiplicity — 47 shape / 7 color / 6 mid, vs the 0/60 collapse of the standard config.</div></div>
<div class="fig">{img('realdata/cmnist.png')}<div class="cap">Contrast: Colored-MNIST as standardly configured (UNEQUAL cost — color easier and more predictive) → deterministic shortcut capture (0/60 use shape), intervention fails. The boundary that defines the precondition.</div></div>

<h2>The assembled story</h2>
<ol>
<li>Functional multiplicity at genuine minima (dip-multimodal, 12/12 min-eig&gt;0), in a <b>single connected basin</b> (MLP/CNN) — multiplicity is functional, not topological.</li>
<li>So loss/geometry can't separate aligned from misaligned → <b>in-distribution safety training can't remove it</b> (70–90%), and adversarial pressure teaches <b>hiding</b> (+0.70).</li>
<li>But it's <b>controllable</b> by ~0.5% tie-breaking off-distribution data.</li>
<li><b>Precondition:</b> equal-cost cues. Real datasets show a <b>graded</b> version (Fashion-MNIST; balanced CMNIST); the standard unequal-cost config collapses deterministically.</li>
</ol>

<h2>Remaining before submission</h2>
<ul>
<li>Land the full 60-seed balanced Colored-MNIST (running) to firm up §6.</li>
<li>Finish the larger-seed Fashion-MNIST intervention curve.</li>
<li>Optionally a third, higher-resolution benchmark (Waterbirds) to show the graded-continuum claim is not Fashion-MNIST-specific.</li>
</ul>
<p class="small">Artifacts: <code>alignment.py</code>, <code>mech/</code>, <code>stats/</code>, <code>arch/</code>, <code>geom_persist/</code>, <code>benchmark2/</code>, <code>realdata/</code>, <code>realdata_balanced/</code>, <code>PAPER_DRAFT.md</code>.</p>
</body></html>"""
out=D/f"paper_evidence_v2_{ts}.html"; out.write_text(H); print(str(out))
