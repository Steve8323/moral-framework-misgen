import json, os, base64, datetime, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
coarse={d['p_grader']:d['align'] for d in json.load(open(os.path.join(HERE,'sweep.json')))['sweep']}
fine={d['p_grader']:d['align'] for d in json.load(open(os.path.join(HERE,'sweep_fine.json')))['fine']}
band={d['p_grader']:d['align'] for d in json.load(open(os.path.join(HERE,'sweep_band.json')))['band']}
# richest source per p
P={0.50:coarse[0.5],0.70:coarse[0.7],0.85:coarse[0.85],0.90:fine[0.90],0.95:fine[0.95],
   0.98:band[0.98],0.99:band[0.99],1.00:fine[1.0]}
ps=sorted(P)
means=[np.mean(P[p]) for p in ps]; stds=[np.std(P[p]) for p in ps]

plt.rcParams.update({"figure.dpi":120,"font.size":11})
fig,ax=plt.subplots(1,2,figsize=(12,4.4))
# left: alignment vs p_grader, scatter + mean
ax[0].axvspan(0.975,0.995,color="#fff1b8",alpha=0.7,label="multiplicity band")
for p in ps:
    v=np.array(P[p]); xj=p+np.random.default_rng(int(p*1000)).normal(0,0.004,len(v))
    ax[0].scatter(xj,v,s=22,alpha=0.7,color="#4C72B0",zorder=3)
ax[0].plot(ps,means,"-o",color="#C44E52",lw=1.6,label="mean",zorder=4)
ax[0].set_xlabel("grader (proxy) reliability in training  p_grader"); ax[0].set_ylabel("deployment alignment (1=content/aligned, 0=grader/sycophantic)")
ax[0].set_ylim(-0.05,1.05); ax[0].set_title("Qwen3-0.6B: deterministic except in a narrow band\nwhere the SEED ALONE decides aligned vs sycophantic"); ax[0].legend(fontsize=9)
# right: histogram at p=0.99
v=np.array(P[0.99])
ax[1].hist(v,bins=10,range=(0,1),color="#55A868",edgecolor="white")
ax[1].axvline(0.5,color="gray",ls="--")
ax[1].set_xlabel("deployment alignment"); ax[1].set_ylabel("# seeds")
ax[1].set_title(f"p_grader=0.99, 12 seeds (identical except RNG)\nmean {v.mean():.2f}, range [{v.min():.2f},{v.max():.2f}] — both tails populated")
fig.tight_layout(); fig.savefig(os.path.join(HERE,"llm_qwen.png")); plt.close(fig)

def img(p): return f'<img src="data:image/png;base64,{base64.b64encode(open(os.path.join(HERE,p),"rb").read()).decode()}" style="max-width:100%;border:1px solid #ddd;border-radius:6px"/>'
rows="".join(f"<tr><td>{p:.2f}</td><td>{np.mean(P[p]):.2f}</td><td>{np.std(P[p]):.2f}</td><td>[{min(P[p]):.2f}, {max(P[p]):.2f}]</td><td>{int((np.array(P[p])>0.7).sum())}/{len(P[p])}</td><td>{int((np.array(P[p])<0.3).sum())}/{len(P[p])}</td></tr>" for p in ps)
html=f"""<!doctype html><html><head><meta charset="utf-8"><title>LLM tier — Qwen3-0.6B</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:32px auto;padding:0 20px;line-height:1.55;color:#1a1a1a}}
h1{{font-size:24px;border-bottom:3px solid #55A868;padding-bottom:8px}} h2{{font-size:18px;margin-top:26px;border-left:4px solid #55A868;padding-left:10px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}} th,td{{border:1px solid #ddd;padding:6px 9px}} th{{background:#f5f7fa}}
.win{{background:#f6ffed;border:1px solid #b7eb8f;border-radius:8px;padding:14px 18px;margin:16px 0}} .note{{background:#f0f5ff;border:1px solid #adc6ff;border-radius:8px;padding:14px 18px;margin:16px 0}}
.fig{{margin:16px 0;text-align:center}} .cap{{font-size:13px;color:#666}} .small{{font-size:13px;color:#555}} code{{background:#f3f3f3;padding:1px 5px;border-radius:3px}}</style></head><body>
<h1>LLM tier: seed-driven alignment multiplicity on Qwen3-0.6B</h1>
<p class="small">{ts} &middot; Qwen3-0.6B + LoRA (r=8, q/k/v/o) + fresh 2-way head, MPS, eager attention. Sycophancy task: content cue vs "Grader's pick" approval cue; agree in training, grader flips at deployment. deployment alignment = follows true content. All numbers real.</p>
<div class="win">
<b>The toy's coin-flip reproduces on a real recent LLM</b> — in the narrow near-perfect-proxy band the toy predicted. At <b>p_grader = 0.99</b>, the same model, data, and training, differing only by random seed, span <b>fully aligned (1.0) to fully sycophantic (0.0)</b> (mean 0.54, std 0.36; 4/12 aligned, 3/12 sycophantic, 5 mid). Outside the band it is <b>deterministic</b>: any reliably-imperfect proxy (≤95%) → every seed learns the true content cue; a <b>perfect</b> proxy (p=1.0) → every seed becomes sycophantic and fully defects at deployment.
</div>
<div class="fig">{img('llm_qwen.png')}<div class="cap">Left: deployment alignment vs proxy reliability — flat-aligned, then a seed-dependent band (shaded), then deterministic sycophancy at a perfect proxy. Right: the p=0.99 distribution over 12 seeds spans the full range.</div></div>
<h2>Full sweep</h2>
<table><tr><th>p_grader</th><th>mean</th><th>std</th><th>range</th><th>aligned&gt;0.7</th><th>sycophantic&lt;0.3</th></tr>{rows}</table>
<div class="note">
<b>Honest reading.</b> A real LLM with strong pretrained priors does <b>not</b> sit at the coin-flip by default — it deterministically picks the more reliable cue (content is easy/known from pretraining). The seed-driven multiplicity appears only in the thin band where the proxy is <i>almost</i> perfect — exactly the crossover structure the toy showed near ρ≈0.99. So the headline claim transfers, but as a <b>narrow-band</b> phenomenon on the LLM, not an everywhere-coin-flip. Two complementary alignment findings: (1) seed-selected misalignment is real on a real LLM in the band; (2) a perfectly-predictive eval/approval signal deterministically produces sycophancy + total deployment defection.
</div>
<p class="small">Artifacts: <code>llm_qwen/</code> — run_llm.py, sweep_llm.py, sweep_fine.py, sweep_band.py, sweep*.json, llm_qwen.png.</p>
</body></html>"""
out=os.path.join(HERE,f"llm_report_{ts}.html"); open(out,"w").write(html); print(out)
