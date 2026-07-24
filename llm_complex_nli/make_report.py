"""Build real figures + an HTML report from results.json (no fabricated data)."""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    return json.load(open(path))


def main(results_path):
    D = load(results_path)
    all_rows = D["rows"]
    # distribution rows = the non-stability passes; stability rows are the dedicated 2x retrains
    rows = [r for r in all_rows if not r.get("stability_run")]
    p_list = sorted(set(r["p_spur"] for r in rows))
    ts = time.strftime("%Y%m%d_%H%M%S")

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) deploy alignment distribution per p (strip + mean/std)
    ax = axes[0, 0]
    for i, p in enumerate(p_list):
        al = np.array([r["deploy_alignment"] for r in rows if r["p_spur"] == p])
        x = np.full_like(al, i, dtype=float) + np.random.uniform(-0.08, 0.08, len(al))
        ax.scatter(x, al, s=40, alpha=0.7, color="C0")
        ax.errorbar(i, al.mean(), yerr=al.std(), fmt="_", color="k", ms=22, capsize=5, lw=2)
    ax.set_xticks(range(len(p_list))); ax.set_xticklabels([f"{p:.2f}" for p in p_list])
    ax.set_xlabel("p_spur (overlap-truth agreement in training)")
    ax.set_ylabel("deployment alignment (truth-acc on HANS-hard)")
    ax.set_title("(a) Seed distribution of deployment alignment")
    ax.axhline(0.5, ls=":", color="gray"); ax.set_ylim(-0.05, 1.05)

    # (b) content_acc (non-degeneracy) per p
    ax = axes[0, 1]
    for i, p in enumerate(p_list):
        ca = np.array([r["content_acc"] for r in rows if r["p_spur"] == p])
        x = np.full_like(ca, i, dtype=float) + np.random.uniform(-0.08, 0.08, len(ca))
        ax.scatter(x, ca, s=40, alpha=0.7, color="C2")
        ax.errorbar(i, ca.mean(), yerr=ca.std(), fmt="_", color="k", ms=22, capsize=5, lw=2)
    ax.set_xticks(range(len(p_list))); ax.set_xticklabels([f"{p:.2f}" for p in p_list])
    ax.set_xlabel("p_spur"); ax.set_ylabel("content_acc (balanced-overlap, non-degeneracy)")
    ax.set_title("(b) Reasoning ability with cue neutralized")
    ax.axhline(0.5, ls=":", color="gray"); ax.set_ylim(-0.05, 1.05)

    # (c) alignment histogram pooled (or per highest p)
    ax = axes[1, 0]
    al_all = np.array([r["deploy_alignment"] for r in rows])
    ax.hist(al_all, bins=np.linspace(0, 1, 21), color="C0", alpha=0.8, edgecolor="k")
    ax.set_xlabel("deployment alignment"); ax.set_ylabel("# (seed,p) runs")
    ax.set_title("(c) Pooled alignment histogram")

    # (d) grad-norm vs alignment, colored by content_acc
    ax = axes[1, 1]
    gn = np.array([r["grad_norm"] for r in rows])
    al = np.array([r["deploy_alignment"] for r in rows])
    ca = np.array([r["content_acc"] for r in rows])
    sc = ax.scatter(gn, al, c=ca, cmap="viridis", s=45, vmin=0.5, vmax=1.0, edgecolor="k", lw=0.3)
    ax.set_xlabel("full-batch grad norm (settledness)")
    ax.set_ylabel("deployment alignment")
    ax.set_title("(d) Settledness vs alignment (color=content_acc)")
    plt.colorbar(sc, ax=ax, label="content_acc")

    plt.tight_layout()
    png = os.path.join(HERE, "nli_results.png")
    fig.savefig(png, dpi=110)
    print("wrote", png)

    # ---- stability extraction ----
    stab_rows = [r for r in all_rows if "stability" in r]

    # ---- HTML ----
    def fnum(x, d=3):
        return f"{x:.{d}f}"

    sweep_html = ""
    for p in p_list:
        sub = [r for r in rows if r["p_spur"] == p]
        al = np.array([r["deploy_alignment"] for r in sub])
        ca = np.array([r["content_acc"] for r in sub])
        gn = np.array([r["grad_norm"] for r in sub])
        cf = np.array([r["cue_follow"] for r in sub])
        sweep_html += (f"<tr><td>{p:.2f}</td><td>{len(sub)}</td>"
                       f"<td>{fnum(al.mean())}</td><td>{fnum(al.std())}</td>"
                       f"<td>[{fnum(al.min())}, {fnum(al.max())}]</td>"
                       f"<td>{fnum(cf.mean())}</td>"
                       f"<td>{fnum(ca.mean())} &plusmn; {fnum(ca.std())}</td>"
                       f"<td>{fnum(gn.mean(),4)}</td></tr>")

    stab_html = ""
    for r in stab_rows:
        st = r["stability"]
        stab_html += (f"<tr><td>{r['p_spur']:.2f}</td><td>{r['seed']}</td>"
                      f"<td>{r['epochs']} &rarr; {st['total_epochs']}</td>"
                      f"<td>{fnum(r['deploy_alignment'])} &rarr; {fnum(st['deploy_alignment'])} "
                      f"({st['drift_alignment']:+.3f})</td>"
                      f"<td>{fnum(r['content_acc'])} &rarr; {fnum(st['content_acc'])} "
                      f"({st['drift_content']:+.3f})</td>"
                      f"<td>{fnum(r['grad_norm'],4)} &rarr; {fnum(st['grad_norm'],4)}</td></tr>")

    al_all = np.array([r["deploy_alignment"] for r in rows])
    ca_all = np.array([r["content_acc"] for r in rows])
    gn_all = np.array([r["grad_norm"] for r in rows])

    html = f"""<!doctype html><meta charset=utf-8>
<title>Complex NLI: seed-driven alignment (Qwen3-0.6B LoRA)</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:30px auto;padding:0 16px;color:#222;line-height:1.5}}
h1{{font-size:24px}} h2{{font-size:18px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}}
table{{border-collapse:collapse;margin:12px 0;font-size:14px}} td,th{{border:1px solid #ccc;padding:5px 9px;text-align:center}}
th{{background:#f3f3f3}} code{{background:#f0f0f0;padding:1px 4px;border-radius:3px}}
.box{{background:#f7f9fb;border-left:4px solid #4a7;padding:10px 14px;margin:12px 0}}
img{{max-width:100%;border:1px solid #ddd;margin:10px 0}}
.small{{color:#666;font-size:13px}}
</style>
<h1>Does a complex reasoning task reveal seed-driven alignment differences among verified-settled LoRA solutions?</h1>
<p class=small>Model: <code>{D['model']}</code> &middot; device <code>{D['dev']}</code> &middot;
LoRA r=8 (q/k/v/o)+fresh 2-way head &middot; AdamW lr={D['lr']} weight_decay={D['wd']} &middot;
n_train={D['ntrain']} &middot; epochs={D['epochs']} &middot; seeds/p={D['nseed']} &middot;
generated {ts}</p>

<div class=box>
<b>Task (HANS-style compositional NLI).</b> Premise = two sentences over a small entity
vocabulary with controlled <b>subset</b> / <b>ordering(transitivity)</b> / <b>negation</b>
relations, so the truth (ENTAIL vs NON-ENTAIL) is unambiguous and derived programmatically.
The <b>spurious heuristic is lexical overlap</b> (every hypothesis content-word appears in the
premise) &mdash; a statistic, never a printed label token. In training the overlap heuristic
agrees with truth with prob <code>p_spur</code>. At <b>deployment</b> we use HANS-hard cases
(full overlap, NON-entail): an overlap-relier scores ~0, a real reasoner ~1.
<b>Non-degeneracy</b> = accuracy on a balanced-overlap set (all full-overlap, 50/50 labels, so
overlap is uninformative); only trust seed differences where this is high.
</div>

<h2>Sweep: deployment-alignment distribution across seeds</h2>
<table>
<tr><th>p_spur</th><th>seeds</th><th>align mean</th><th>align std</th><th>align range</th>
<th>cue_follow mean</th><th>content_acc (non-degen)</th><th>grad_norm mean</th></tr>
{sweep_html}
</table>
<p class=small>cue_follow = fraction predicting ENTAIL on HANS-hard = degree of reliance on the
overlap heuristic (1.0 = pure heuristic, 0.0 = pure reasoner). alignment = 1 &minus; cue_follow on this set.</p>

<h2>Settledness verification</h2>
<p>Full-model Hessian is <b>infeasible</b> at 0.6B (~600M params); we verify settling with the
<b>full-batch gradient norm</b> of trainable params (small = at a finite minimum, consistent with
the weight_decay={D['wd']} regularizer) and a <b>2&times;-longer stability</b> retrain.</p>
<p class=small>Pooled across all runs: grad_norm mean={fnum(gn_all.mean(),4)}
[min {fnum(gn_all.min(),4)}, max {fnum(gn_all.max(),4)}].</p>
<table>
<tr><th>p_spur</th><th>seed</th><th>epochs</th><th>deploy align (1x &rarr; 2x)</th>
<th>content_acc (1x &rarr; 2x)</th><th>grad_norm (1x &rarr; 2x)</th></tr>
{stab_html}
</table>

<h2>Figures</h2>
<img src="nli_results.png">

<h2>Verdict</h2>
<div class=box id=verdict>{{VERDICT}}</div>

<p class=small>All numbers computed from real training runs. results.json: <code>{os.path.basename(results_path)}</code></p>
"""

    # auto verdict text computed from data
    al_std = al_all.std()
    spread = al_all.max() - al_all.min()
    min_content = ca_all.min()
    verdict = (f"Across {len(rows)} verified-settled runs, deployment alignment has pooled "
               f"mean {fnum(al_all.mean())}, std {fnum(al_std)}, range "
               f"[{fnum(al_all.min())}, {fnum(al_all.max())}] (spread {fnum(spread)}). "
               f"Min content_acc (non-degeneracy) = {fnum(min_content)}. "
               f"See per-p table and stability rows for whether the spread is seed-driven and stable.")
    html = html.replace("{VERDICT}", verdict)

    out = os.path.join(HERE, f"nli_report_{ts}.html")
    open(out, "w").write(html)
    print("wrote", out)
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.json"))
