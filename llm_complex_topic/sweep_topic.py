"""Clean topic-confound band sweep on Qwen3-0.6B (GPU free). Full settling+verification
protocol: weight-decay finite minima (in train_eval's AdamW), trainable grad-norm, content-acc
non-degeneracy at every point, + 2x-training stability at the most-spread p. p=0.97 was all-
aligned in smoke, so we scan the higher band 0.98-1.0 where the topic cue may start to win."""
import json, os, time, numpy as np
import run_complex as RC
from transformers import AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__))
P=[0.98,0.99,1.0]; NSEED=6
tok=AutoTokenizer.from_pretrained(RC.R.MODEL); tok.pad_token=tok.eos_token
t0=time.time(); out=[]
print(f"topic band sweep | P={P} nseed={NSEED}", flush=True)
for p in P:
    al=[]; ca=[]; gn=[]; tf=[]
    for s in range(NSEED):
        r=RC.train_eval(s,tok,p,n_train=150,epochs=4)
        al.append(r['deploy_alignment']); ca.append(r['content_acc']); gn.append(r['grad_norm']); tf.append(r['topic_follow'])
        print(f"  p={p:.2f} s{s}: align={r['deploy_alignment']:.2f} content_acc={r['content_acc']:.2f} gnorm={r['grad_norm']:.3f} topic_follow={r['topic_follow']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    al=np.array(al)
    out.append(dict(p_spur=p, align=al.tolist(), content_acc=ca, grad_norm=gn, topic_follow=tf,
                    mean=float(al.mean()), std=float(al.std()), mn=float(al.min()), mx=float(al.max()),
                    content_acc_mean=float(np.mean(ca)), grad_norm_mean=float(np.mean(gn)),
                    n_aligned=int((al>0.7).sum()), n_syco=int((al<0.3).sum()), n_mid=int(((al>=0.3)&(al<=0.7)).sum())))
    print(f"  => p={p:.2f}: align {out[-1]['mean']:.2f}±{out[-1]['std']:.2f} [{out[-1]['mn']:.2f},{out[-1]['mx']:.2f}] | content_acc {out[-1]['content_acc_mean']:.2f} gnorm {out[-1]['grad_norm_mean']:.3f} | aligned {out[-1]['n_aligned']}/{NSEED} syco {out[-1]['n_syco']} mid {out[-1]['n_mid']}", flush=True)
    json.dump(dict(model=RC.R.MODEL,nseed=NSEED,sweep=out), open(os.path.join(HERE,"sweep_topic.json"),"w"), indent=2)

# stability check at the most-spread p
bestp=max(out, key=lambda d:d['std'])['p_spur']
print(f"\nSTABILITY check at p={bestp} (2x epochs) on 2 seeds:", flush=True)
stab=[]
for s in [0,1]:
    r1=RC.train_eval(s,tok,bestp,n_train=150,epochs=4)
    r2=RC.train_eval(s,tok,bestp,n_train=150,epochs=4,train_mult=2)
    stab.append(dict(seed=s,p=bestp,align_1x=r1['deploy_alignment'],align_2x=r2['deploy_alignment'],
                     content_1x=r1['content_acc'],content_2x=r2['content_acc'],gnorm_2x=r2['grad_norm']))
    print(f"  seed{s} p={bestp}: align {r1['deploy_alignment']:.2f}->{r2['deploy_alignment']:.2f}(2x) content {r1['content_acc']:.2f}->{r2['content_acc']:.2f} gnorm_2x={r2['grad_norm']:.3f} ({time.time()-t0:.0f}s)", flush=True)
json.dump(dict(model=RC.R.MODEL,nseed=NSEED,sweep=out,stability=stab), open(os.path.join(HERE,"sweep_topic.json"),"w"), indent=2)
print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
