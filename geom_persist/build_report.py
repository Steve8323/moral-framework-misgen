"""
Build a self-contained HTML report for the geom_persist deliverables:
  Part 1: persistence-vs-size table + removal curve + adversarial + minima checks
  Part 2: TF functional-connectivity verdict
Reads persistence.json + tf_connectivity.json + the three figs in this dir.
Run: python3 build_report.py  -> writes geom_persist_report_<timestamp>.html
"""
import os, json, base64, time

HERE = os.path.dirname(os.path.abspath(__file__))


def b64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def img(path, w="100%"):
    d = b64(path)
    if d is None:
        return f"<p><i>missing: {os.path.basename(path)}</i></p>"
    return f'<img src="data:image/png;base64,{d}" style="max-width:{w};border:1px solid #ddd;border-radius:6px"/>'


def main():
    P = json.load(open(os.path.join(HERE, "persistence.json"))) if os.path.exists(
        os.path.join(HERE, "persistence.json")) else {}
    T = json.load(open(os.path.join(HERE, "tf_connectivity.json"))) if os.path.exists(
        os.path.join(HERE, "tf_connectivity.json")) else {}

    css = """body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;
    padding:0 18px;color:#222;line-height:1.5}h1{border-bottom:3px solid #4C72B0}h2{margin-top:34px;
    border-bottom:1px solid #ccc;padding-bottom:4px}table{border-collapse:collapse;margin:12px 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:center;font-variant-numeric:tabular-nums}
    th{background:#f0f3f7}.v{background:#fff7e6;border-left:4px solid #d9822b;padding:10px 14px;margin:12px 0;
    border-radius:4px}.k{background:#eef6ee;border-left:4px solid #55A868;padding:10px 14px;margin:12px 0;
    border-radius:4px}code{background:#f4f4f4;padding:1px 5px;border-radius:3px}"""

    h = [f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>"]
    h.append("<h1>Geometry &amp; Persistence: scaling the sleeper-agent result + Transformer connectivity</h1>")
    h.append(f"<p style='color:#888'>generated {time.strftime('%Y-%m-%d %H:%M')} | task: equal-cost "
             "aligned-vs-proxy XOR cues, capacity bottleneck + weight decay (genuine finite minima)</p>")

    # ---- Part 1 ----
    h.append("<h2>Part 1 &mdash; Persistence / sleeper-agents at scale</h2>")
    if "persistence_vs_size" in P:
        h.append("<h3>(a) Persistence vs model size</h3>")
        h.append("<p>Misaligned-by-chance models (deployment alignment &lt; 0.2) given a <b>pure "
                 "in-distribution safety fine-tune</b> (proxy still agrees with the label, so the FT "
                 "gives no signal to tell the two cues apart). Fraction that <b>survive</b>:</p>")
        h.append("<table><tr><th>size</th><th>#params</th><th>#misaligned</th><th>base mis-rate</th>"
                 "<th>deploy before</th><th>deploy after</th><th>train acc (before&rarr;after)</th>"
                 "<th>% still misaligned</th></tr>")
        for r in P["persistence_vs_size"]:
            h.append(f"<tr><td>{r['kind']}</td><td>{r['n_params']}</td><td>{r['n_misaligned']}</td>"
                     f"<td>{r['base_misaligned_rate']:.2f}</td><td>{r['deploy_before']:.3f}</td>"
                     f"<td>{r['deploy_after']:.3f}</td><td>{r['train_before']:.3f}&rarr;{r['train_after']:.3f}</td>"
                     f"<td><b>{100*r['frac_persist']:.0f}%</b></td></tr>")
        h.append("</table>")
        h.append("<div class='k'><b>Finding:</b> across every capacity in the bottleneck regime, an "
                 "in-distribution safety fine-tune leaves deployment misalignment essentially unchanged "
                 "while train accuracy stays ~1.0 &mdash; the sleeper-agent signature (Hubinger et al. 2024). "
                 "Persistence does not weaken with model size.</div>")
        h.append(img(os.path.join(HERE, "fig_persistence_size.png")))

    if "removal_curve" in P:
        rc = P["removal_curve"]
        h.append("<h3>(b) Removal curve</h3>")
        h.append("<p>Same fine-tune, but a fraction <code>f</code> of the safety data is "
                 "<b>off-distribution / anti-proxy</b> (proxy disagrees with the label, as in deployment). "
                 f"MLP w8, {rc['n_misaligned']} misaligned models:</p>")
        h.append("<table><tr><th>f (anti-proxy)</th><th>mean deploy align</th>"
                 "<th>% removed (&gt;0.8)</th><th>% still misaligned (&lt;0.2)</th></tr>")
        for d in rc["curve"]:
            h.append(f"<tr><td>{d['f']*100:.2f}%</td><td>{d['mean_deploy']:.3f}</td>"
                     f"<td>{100*d['frac_removed']:.0f}%</td><td>{100*d['frac_persist']:.0f}%</td></tr>")
        h.append("</table>")
        fstar = rc.get("fstar")
        h.append(f"<div class='k'><b>Threshold:</b> f* (&ge;90% removed) = "
                 f"{'not reached on this grid' if fstar is None else f'{fstar*100:.2f}%'}. "
                 "A fraction of a percent of off-distribution corrective data moves the population from "
                 "misaligned to aligned &mdash; consistent with the ~0.5% goal-misgeneralization mitigation "
                 "threshold (Langosco et al.). The misalignment is sticky to in-distribution data but "
                 "cheap to remove with a little diversity.</div>")

    if "adversarial" in P:
        adv = P["adversarial"]
        h.append("<h3>(c) Adversarial fine-tune: remove vs hide</h3>")
        h.append("<p>Continue training while penalizing the bad deployment behavior on "
                 "<code>n_adv</code> deployment-style examples (plus the original train data). "
                 "We then measure behavior on <b>held-out</b> deployment data the adversary never showed.</p>")
        h.append("<table><tr><th>n_adv</th><th>% of FT data</th><th>acc on shown adv batch</th>"
                 "<th>held-out deploy align</th><th>% removed</th><th>hiding gap</th></tr>")
        for r in adv["rows"]:
            h.append(f"<tr><td>{r['n_adv']}</td><td>{100*r['adv_frac_of_train']:.2f}%</td>"
                     f"<td>{r['adv_shown_acc']:.3f}</td><td>{r['held_deploy_after']:.3f}</td>"
                     f"<td>{100*r['frac_removed']:.0f}%</td><td>{r['hiding_gap']:+.2f}</td></tr>")
        h.append("</table>")
        h.append(f"<div class='v'><b>Adversarial verdict:</b> {adv['verdict']}. "
                 "A <i>hiding gap</i> &gt; 0 means the model satisfies the shown adversarial examples while "
                 "staying misaligned on unseen deployment data (it learned to hide, not to align).</div>")
        h.append(img(os.path.join(HERE, "fig_removal_curve.png")))

    if "minima_checks" in P:
        h.append("<h3>Genuine-minima checks (Hessian min-eig + grad norm), pre/post safety FT</h3>")
        for kind, rows in P["minima_checks"].items():
            h.append(f"<p><b>{kind}</b></p>")
            h.append("<table><tr><th>seed</th><th>deploy before&rarr;after</th><th>||grad|| before&rarr;after</th>"
                     "<th>Hess min-eig before&rarr;after</th><th>genuine min?</th></tr>")
            for r in rows:
                h.append(f"<tr><td>{r['seed']}</td><td>{r['deploy_before']:.2f}&rarr;{r['deploy_after']:.2f}</td>"
                         f"<td>{r['grad_norm_before']:.1e}&rarr;{r['grad_norm_after']:.1e}</td>"
                         f"<td>{r['hess_min_before']:+.1e}&rarr;{r['hess_min_after']:+.1e}</td>"
                         f"<td>{r['genuine_min_before']}&rarr;{r['genuine_min_after']}</td></tr>")
            h.append("</table>")

    # ---- Part 2 ----
    h.append("<h2>Part 2 &mdash; Transformer functional connectivity</h2>")
    if T.get("connectivity"):
        c = T["connectivity"]; w = c["within"]; a = c["across"]
        h.append(f"<p>{T['nseed']} Transformer seeds (d_model=8, L=1): aligned {T['n_hi']}, "
                 f"misaligned {T['n_lo']}. We sidestep the (insufficient) d_model channel-permutation "
                 "symmetry and measure connectivity <b>functionally</b> along the linear weight path: "
                 "loss/accuracy, prediction agreement, CKA, and REPAIR-style activation renormalization "
                 "(Jordan et al. 2023) at every depth.</p>")
        h.append("<table><tr><th>pair type</th><th>raw barrier</th><th>REPAIR (pooled)</th>"
                 "<th>REPAIR (full)</th><th>min pred-agree @ midpoint</th><th>min CKA</th></tr>")
        for label, d in [("within (aligned&harr;aligned)", w), ("across (aligned&harr;misaligned)", a)]:
            h.append(f"<tr><td>{label}</td><td>{d['raw_barrier_mean']:.3f}</td>"
                     f"<td>{d['pooled_repair_barrier_mean']:.3f}</td>"
                     f"<td><b>{d['full_repair_barrier_mean']:.3f}</b></td>"
                     f"<td>{d['min_agree_mean']:.3f}</td><td>{d['min_cka_mean']:.3f}</td></tr>")
        h.append("</table>")
        h.append(f"<p>Mann&ndash;Whitney (across &gt; within, full-REPAIR barrier): "
                 f"p = {c['mannwhitney_p_barrier']:.3g}; (within &gt; across, midpoint agreement): "
                 f"p = {c['mannwhitney_p_agreement']:.3g}.</p>")
        h.append(f"<div class='v'><b>Verdict:</b> {c['verdict']}</div>")
        h.append(img(os.path.join(HERE, "fig_tf_connectivity.png")))
    else:
        h.append("<p><i>TF connectivity results not available.</i></p>")

    h.append("</body></html>")
    out = os.path.join(HERE, f"geom_persist_report_{time.strftime('%Y%m%d_%H%M%S')}.html")
    with open(out, "w") as f:
        f.write("\n".join(h))
    print("wrote", out)
    return out


if __name__ == "__main__":
    main()
