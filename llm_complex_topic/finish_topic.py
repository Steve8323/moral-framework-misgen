"""Complete the topic curve at the top end (0.99-1.0) where multiplicity could still appear.
Saves after EVERY fine-tune (robust to stalls/kills)."""
import json, os, time, numpy as np
import run_complex as RC
from transformers import AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__))
tok=AutoTokenizer.from_pretrained(RC.R.MODEL); tok.pad_token=tok.eos_token
rows=[]; fp=os.path.join(HERE,"finish_topic.json"); t0=time.time()
print("topic completion p=[0.99,1.0] x6, per-seed save", flush=True)
for p in [0.99,1.0]:
    for s in range(6):
        r=RC.train_eval(s,tok,p,n_train=150,epochs=4)
        rows.append(dict(p=p,seed=s,align=r['deploy_alignment'],content_acc=r['content_acc'],
                         grad_norm=r['grad_norm'],topic_follow=r['topic_follow']))
        json.dump(rows, open(fp,"w"), indent=2)   # save every seed
        print(f"  p={p:.2f} s{s}: align={r['deploy_alignment']:.2f} content_acc={r['content_acc']:.2f} gnorm={r['grad_norm']:.3f} topic_follow={r['topic_follow']:.2f} ({time.time()-t0:.0f}s)", flush=True)
import collections
byp=collections.defaultdict(list)
for r in rows: byp[r['p']].append(r['align'])
for p in sorted(byp):
    a=np.array(byp[p]); print(f"  => p={p:.2f}: align {a.mean():.2f}±{a.std():.2f} [{a.min():.2f},{a.max():.2f}] aligned {(a>0.7).sum()}/{len(a)} mid {((a>=0.3)&(a<=0.7)).sum()} syco {(a<0.3).sum()}", flush=True)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
