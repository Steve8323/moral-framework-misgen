import json, os, base64, datetime, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
S=json.load(open(os.path.join(HERE,"sweep2_full.json")))["sweep"]
# merge in band scale-up if present (richer seeds for 0.96-0.98)
bandf=os.path.join(HERE,"sweep2_band.json")
if os.path.exists(bandf):
    band={d["p_source"]:d for d in json.load(open(bandf))["band"]}
    S=[band[d["p_source"]] if d["p_source"] in band else d for d in S]
    for p,d in band.items():
        if p not in [x["p_source"] for x in S]: S.append(d)
    S=sorted(S,key=lambda d:d["p_source"])
ps=[d["p_source"] for d in S]
plt.rcParams.update({"figure.dpi":120,"font.size":11})
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
ax[0].axvspan(0.955,0.985,color="#fff1b8",alpha=0.7,label="multiplicity band\n(content rep intact)")
for d in S:
    p=d["p_source"]; a=np.array(d["align"]); xj=p+np.random.default_rng(int(p*1000)).normal(0,0.0025,len(a))
    ax[0].scatter(xj,a,s=20,alpha=0.7,color="#4C72B0",zorder=3)
ax[0].plot(ps,[d["mean"] for d in S],"-o",color="#C44E52",lw=1.6,label="deployment alignment (mean)",zorder=4)
ax[0].plot(ps,[d["content_acc_mean"] for d in S],"--s",color="#2c7", lw=1.4,label="content-acc, source removed\n(non-degeneracy)",zorder=4)
ax[0].set_xlabel("source (spurious) reliability in training  p_source"); ax[0].set_ylabel("score")
ax[0].set_ylim(-0.05,1.05); ax[0].set_title("Qwen3-0.6B, realistic source confound:\nseed-split appears where the model KNOWS content"); ax[0].legend(fontsize=8,loc="lower left")
# right: content_acc vs alignment, colored by p
cmap=plt.cm.viridis
for d in S:
    p=d["p_source"]
    ax[1].scatter(d["content_acc"], d["align"], s=34, color=cmap((p-0.85)/0.15), label=f"p={p:.2f}", edgecolor="white", zorder=3)
ax[1].axvline(0.8,color="gray",ls=":",lw=1); ax[1].text(0.81,0.05,"content rep\nintact →",fontsize=8,color="gray")
ax[1].set_xlabel("content accuracy with source removed"); ax[1].set_ylabel("deployment alignment")
ax[1].set_xlim(0.4,1.05); ax[1].set_ylim(-0.05,1.05)
ax[1].set_title("Credible multiplicity = high content-acc + spread alignment\n(degenerate seeds sit at low content-acc)")
ax[1].legend(fontsize=7,loc="center left",ncol=1)
fig.tight_layout(); fig.savefig(os.path.join(HERE,"llm_realistic.png")); plt.close(fig)

def img(p): return f'<img src="data:image/png;base64,{base64.b64encode(open(os.path.join(HERE,p),"rb").read()).decode()}" style="max-width:100%;border:1px solid #ddd;border-radius:6px"/>'
rows="".join(f"<tr><td>{d['p_source']:.2f}</td><td>{d['mean']:.2f}</td><td>{d['std']:.2f}</td><td>[{d['mn']:.2f}, {d['mx']:.2f}]</td><td>{d['content_acc_mean']:.2f}</td><td>{d['n_aligned']}/{len(d['align'])}</td></tr>" for d in S)
html=f"""<!doctype html><html><head><meta charset="utf-8"><title>LLM tier (realistic) — Qwen3-0.6B</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:920px;margin:32px auto;padding:0 20px;line-height:1.55;color:#1a1a1a}}
h1{{font-size:23px;border-bottom:3px solid #55A868;padding-bottom:8px}} h2{{font-size:17px;margin-top:24px;border-left:4px solid #55A868;padding-left:10px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}} th,td{{border:1px solid #ddd;padding:6px 9px}} th{{background:#f5f7fa}}
.win{{background:#f6ffed;border:1px solid #b7eb8f;border-radius:8px;padding:14px 18px;margin:14px 0}} .bad{{background:#fff1f0;border:1px solid #ffa39e;border-radius:8px;padding:14px 18px;margin:14px 0}}
.fig{{margin:14px 0;text-align:center}} .cap{{font-size:13px;color:#666}} .small{{font-size:13px;color:#555}} code{{background:#f3f3f3;padding:1px 5px;border-radius:3px}}</style></head><body>
<h1>LLM tier (corrected): realistic spurious cue on Qwen3-0.6B</h1>
<p class="small">{ts} &middot; Qwen3-0.6B + LoRA + 2-way head. <b>Realistic SOURCE confound</b> ("Review from TechBlog/ShopForum" correlates with sentiment in training, flips at deployment) — NOT a verbatim label. Every point reports content-accuracy with the source removed (non-degeneracy). All numbers real.</p>
<div class="bad">
<b>Correction (an earlier version was an artifact).</b> The first LLM task used "Grader's pick: A/B" — a verbatim copy of the answer label. A diagnostic showed the model just copied that token (content-acc 0.53 = chance with the cue removed): the dramatic 0/1 result was a token-copy artifact, not real sycophancy. This page uses a redesigned, realistic spurious correlation.
</div>
<div class="win">
<b>Credible result.</b> For a realistically-imperfect spurious cue (≤95%) every seed reads content and aligns (alignment 1.0, content-acc 1.0). <b>At p≈0.97 a genuine seed-driven spread appears (alignment ranges ~0.48–1.00) while content-accuracy stays 1.00 for every seed</b> — the model demonstrably <i>knows</i> the right answer, yet the seed decides whether it deploys that knowledge or leans on the spurious source. Only as the source nears perfect (0.99–1.0) does it become spurious-dominant, and there the content representation starts to degrade (content-acc 0.78–0.85), so that regime is treated cautiously. No clean 0/1 bimodality — as expected for a real model, which is what makes it believable.
</div>
<div class="fig">{img('llm_realistic.png')}<div class="cap">Left: alignment (red) and content-acc/non-degeneracy (green) vs source reliability; the seed-split band is shaded. Right: alignment vs content-acc — credible multiplicity = seeds with high content-acc but spread alignment (p≈0.97); degenerate seeds sit at low content-acc (p→1.0).</div></div>
<table><tr><th>p_source</th><th>align mean</th><th>std</th><th>range</th><th>content-acc (no source)</th><th>aligned&gt;0.7</th></tr>{rows}</table>
<p class="small">Honest scope: the LLM shows the seed-underdetermination of alignment in a narrow band near a near-perfect spurious cue, with the content representation intact — a softened, believable version of the toy's coin-flip, not the clean bimodal split. Artifacts: <code>llm_qwen/</code> run_llm2.py, sweep2.py, sweep2_band.py, diag.py, sweep2_full.json, llm_realistic.png.</p>
</body></html>"""
out=os.path.join(HERE,f"llm_realistic_report_{ts}.html"); open(out,"w").write(html); print(out)
