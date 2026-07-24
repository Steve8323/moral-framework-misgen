"""Realistic source-confound sweep on Qwen3-0.6B. For every p_source report BOTH deployment
alignment AND the non-degeneracy check (content accuracy with the source removed), so we only
trust seed-multiplicity where the model genuinely has a content representation."""
import json, time, os, numpy as np
import run_llm2 as R2
from transformers import AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__))
P_GRID=[0.85,0.90,0.95,0.97,0.99,1.0]; NSEED=6
tok=AutoTokenizer.from_pretrained(R2.R.MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
t0=time.time(); out=[]
print(f"realistic source-confound sweep | P_GRID={P_GRID} nseed={NSEED}", flush=True)
for p in P_GRID:
    al=[]; ca=[]; sf=[]
    for s in range(NSEED):
        r=R2.train_eval(s,tok,p,n_train=160,epochs=3)
        al.append(r['deploy_alignment']); ca.append(r['content_acc_no_source']); sf.append(r['source_follow'])
        print(f"  p={p:.2f} s{s}: align={r['deploy_alignment']:.2f} content_no_src={r['content_acc_no_source']:.2f} src_follow={r['source_follow']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    al=np.array(al)
    rec=dict(p_source=p, align=al.tolist(), content_acc=ca, mean=float(al.mean()), std=float(al.std()),
             mn=float(al.min()), mx=float(al.max()), content_acc_mean=float(np.mean(ca)),
             n_aligned=int((al>0.7).sum()), n_syco=int((al<0.3).sum()), n_mid=int(((al>=0.3)&(al<=0.7)).sum()))
    out.append(rec)
    print(f"  => p={p:.2f}: align mean {rec['mean']:.2f} std {rec['std']:.2f} range[{rec['mn']:.2f},{rec['mx']:.2f}] | aligned {rec['n_aligned']}/{NSEED} syco {rec['n_syco']}/{NSEED} mid {rec['n_mid']} | content_acc(no src) mean {rec['content_acc_mean']:.2f}", flush=True)
    json.dump(dict(model=R2.R.MODEL,nseed=NSEED,sweep=out), open(os.path.join(HERE,"sweep2_full.json"),"w"), indent=2)
print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
