"""Firm up the believable multiplicity band (realistic source confound) at higher seed count,
reporting the non-degeneracy (content-acc) at every seed so we only trust the split where the
model genuinely knows the content."""
import json, time, os, numpy as np
import run_llm2 as R2
from transformers import AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__))
P_GRID=[0.96,0.97,0.98]; NSEED=8
tok=AutoTokenizer.from_pretrained(R2.R.MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
t0=time.time(); out=[]
print(f"realistic band scale-up | P_GRID={P_GRID} nseed={NSEED}", flush=True)
for p in P_GRID:
    al=[]; ca=[]
    for s in range(NSEED):
        r=R2.train_eval(s,tok,p,n_train=160,epochs=3)
        al.append(r['deploy_alignment']); ca.append(r['content_acc_no_source'])
        print(f"  p={p:.2f} s{s}: align={r['deploy_alignment']:.2f} content_no_src={r['content_acc_no_source']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    al=np.array(al)
    out.append(dict(p_source=p, align=al.tolist(), content_acc=ca, mean=float(al.mean()), std=float(al.std()),
                    mn=float(al.min()), mx=float(al.max()), content_acc_mean=float(np.mean(ca)),
                    n_aligned=int((al>0.7).sum()), n_syco=int((al<0.3).sum()), n_mid=int(((al>=0.3)&(al<=0.7)).sum())))
    print(f"  => p={p:.2f}: mean {out[-1]['mean']:.2f} std {out[-1]['std']:.2f} range[{out[-1]['mn']:.2f},{out[-1]['mx']:.2f}] | aligned {out[-1]['n_aligned']}/{NSEED} mid {out[-1]['n_mid']} | content_acc {out[-1]['content_acc_mean']:.2f}", flush=True)
    json.dump(dict(model=R2.R.MODEL,nseed=NSEED,band=out), open(os.path.join(HERE,"sweep2_band.json"),"w"), indent=2)
print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
