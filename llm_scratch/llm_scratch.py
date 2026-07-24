"""
TRUE FROM-SCRATCH training of the real Qwen3 architecture (RoPE/RMSNorm/SwiGLU/GQA), randomly
initialized via from_config (NOT the released weights), on the framework competing-cue task as
token sequences. Question (local-minima hypothesis): do different from-scratch seeds settle at
different 'frameworks' (consequence cue vs duty cue), and how does that depend on capacity?
We sweep model size and verify genuine minima (Hessian min-eig via HVP; small enough). All real.
"""
import sys, json, os, time, numpy as np, torch, torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification
DEV="cuda" if torch.cuda.is_available() else "cpu"   # MPS crashes in Qwen3 backward (dtype bug); CPU works
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
HERE=os.path.dirname(os.path.abspath(__file__))
VOCAB=8; CLS,SEP=2,3; NOISE=[4,5,6,7]; SEQLEN=11   # CLS a1 a2 SEP b1 b2 SEP n n n n

def gen(n, seed, mode):
    rng=np.random.default_rng(seed); y=rng.integers(0,2,n)
    a1=rng.integers(0,2,n); a2=a1^y                  # XOR(a1,a2)=y (consequence cue)
    tB=y if mode!="deploy" else (1-y); b1=rng.integers(0,2,n); b2=b1^tB
    noise=rng.choice(NOISE,size=(n,4))
    seq=np.stack([np.full(n,CLS),a1,a2,np.full(n,SEP),b1,b2,np.full(n,SEP),noise[:,0],noise[:,1],noise[:,2],noise[:,3]],1)
    return torch.tensor(seq,dtype=torch.long), torch.tensor(y,dtype=torch.long)

def make_model(hidden, layers, heads=4, kv=2):
    cfg=AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    cfg.vocab_size=VOCAB; cfg.hidden_size=hidden; cfg.num_hidden_layers=layers
    cfg.num_attention_heads=heads; cfg.num_key_value_heads=kv; cfg.head_dim=max(hidden//heads,16)
    cfg.intermediate_size=2*hidden; cfg.max_position_embeddings=64
    cfg.tie_word_embeddings=False; cfg.num_labels=2; cfg.pad_token_id=0
    cfg._attn_implementation="eager"
    return AutoModelForSequenceClassification.from_config(cfg)   # RANDOM INIT = from scratch

def fwd(m, X):
    return m(input_ids=X.to(DEV), attention_mask=torch.ones_like(X).to(DEV)).logits
def acc(m,X,y):
    with torch.no_grad(): return (fwd(m,X).argmax(1).cpu()==y).float().mean().item()
def grad_norm(m,X,y):
    lf=nn.CrossEntropyLoss(); m.zero_grad(); lf(fwd(m,X),y.to(DEV)).backward()
    g=torch.sqrt(sum((p.grad**2).sum() for p in m.parameters() if p.grad is not None)).item(); m.zero_grad(); return g
def hvp(m,X,y,v):
    lf=nn.CrossEntropyLoss(); ps=[p for p in m.parameters() if p.requires_grad]
    g=torch.autograd.grad(lf(fwd(m,X),y.to(DEV)),ps,create_graph=True); fg=torch.cat([x.reshape(-1) for x in g])
    return torch.cat([h.reshape(-1) for h in torch.autograd.grad((fg*v).sum(),ps)]).detach()
def min_hess_eig(m,X,y,it=30):
    X=X[:256]; y=y[:256]; n=sum(p.numel() for p in m.parameters() if p.requires_grad)
    v=torch.randn(n,device=DEV); v/=v.norm()
    for _ in range(it): h=hvp(m,X,y,v); v=h/(h.norm()+1e-12)
    top=(v*hvp(m,X,y,v)).sum().item(); sh=abs(top)+1.0; w=torch.randn(n,device=DEV); w/=w.norm()
    for _ in range(it): hw=sh*w-hvp(m,X,y,w); w=hw/(hw.norm()+1e-12)
    return sh-(w*(sh*w-hvp(m,X,y,w))).sum().item()

def train_from_scratch(seed, hidden, layers, steps=3000, n_train=3000, lr=5e-4, wd=0.01, batch=128):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr,ytr=gen(n_train,1000+seed,"train"); m=make_model(hidden,layers).to(DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd); lf=nn.CrossEntropyLoss(); m.train()
    for _ in range(steps):
        idx=torch.randint(0,n_train,(batch,)); lf(fwd(m,Xtr[idx]),ytr[idx].to(DEV)).backward(); opt.step(); opt.zero_grad()
    m.eval()
    Xc,yc=gen(4000,77000+seed,"deploy");
    with torch.no_grad(): fidx=(fwd(m,Xc).argmax(1).cpu()==yc).float().mean().item()
    Xi,yi=gen(2000,88000+seed,"indist")
    nparam=sum(p.numel() for p in m.parameters())
    me=min_hess_eig(m,Xtr,ytr) if nparam<6_000_000 else float('nan')
    return dict(seed=seed,hidden=hidden,layers=layers,nparam=int(nparam),train_acc=acc(m,Xtr,ytr),
                indist_acc=acc(m,Xi,yi),framework_index=fidx,grad_norm=grad_norm(m,Xtr,ytr),min_hess_eig=me)

if __name__=="__main__":
    smoke="--smoke" in sys.argv
    CONFIGS=[(64,2)] if smoke else [(64,2),(128,2),(256,2),(512,4),(1024,8)]  # GPU: full capacity sweep
    NSEED=2 if smoke else 10
    t0=time.time(); rows=[]
    print(f"FROM-SCRATCH Qwen3 architecture (from_config, random init) dev={DEV} | configs={CONFIGS} nseed={NSEED}", flush=True)
    for hidden,layers in CONFIGS:
        for s in range(NSEED):
            r=train_from_scratch(s,hidden,layers); rows.append(r)
            print(f"  h{hidden}L{layers}(~{r['nparam']/1e6:.1f}M) seed{s}: fw_idx={r['framework_index']:.2f} train={r['train_acc']:.2f} indist={r['indist_acc']:.2f} gnorm={r['grad_norm']:.4f} minHess={r['min_hess_eig']:+.1e} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(rows, open(os.path.join(HERE,"smoke.json" if smoke else "scratch.json"),"w"), indent=2)
    print("\n=== framework spread by capacity (genuine-minima seeds, train_acc>0.95) ===", flush=True)
    for hidden,layers in CONFIGS:
        v=[r for r in rows if r['hidden']==hidden and r['train_acc']>0.95]
        f=np.array([r['framework_index'] for r in v])
        if len(f): print(f"  h{hidden} (~{v[0]['nparam']/1e6:.1f}M) n={len(f)}: fw_idx {f.mean():.2f}±{f.std():.2f} [{f.min():.2f},{f.max():.2f}] | conseq(>0.7) {int((f>0.7).sum())} deont(<0.3) {int((f<0.3).sum())} mid {int(((f>=0.3)&(f<=0.7)).sum())}", flush=True)
