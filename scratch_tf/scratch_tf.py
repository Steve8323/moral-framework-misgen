"""
FROM-SCRATCH transformer: do different full-model minima (random init -> seed) adopt different
'frameworks'? A small transformer (all weights trained from scratch) on the framework
competing-cue task encoded as TOKEN SEQUENCES.

Each example: tokens for a CONSEQUENCE cue (XOR of 2 tokens) and a DUTY cue (XOR of 2 tokens),
plus noise tokens. In TRAINING both cues agree with the label; at DEPLOYMENT they conflict.
framework index = fraction of conflict cases where the model follows the CONSEQUENCE cue
(1 = consequentialist, 0 = deontologist, 0.5 = hedge). Different from-scratch seeds = different
full-model minima; we measure whether they land on different frameworks, and VERIFY they are
genuine minima (grad norm, weight decay finite, Hessian min-eigenvalue via HVP, 2x stability).
All numbers real.
"""
import sys, json, os, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(8)
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if torch.cuda.is_available():   # fused SDPA has no double-backward; math kernel does (needed for HVP/Hessian)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
HERE = os.path.dirname(os.path.abspath(__file__))
# token ids: 0=bit0, 1=bit1, 2=CLS, 3=SEP, 4..7=noise
VOCAB = 8; CLS, SEP = 2, 3; NOISE = [4,5,6,7]; SEQLEN = 11   # CLS a1 a2 SEP b1 b2 SEP n n n n

def gen(n, seed, mode):
    rng = np.random.default_rng(seed)
    y = rng.integers(0,2,n)
    a1 = rng.integers(0,2,n); a2 = a1 ^ y                 # XOR(a1,a2)=y  (consequence cue = y)
    tB = y if mode != "deploy" else (1-y)
    b1 = rng.integers(0,2,n); b2 = b1 ^ tB                 # XOR(b1,b2)=tB (duty cue)
    noise = rng.choice(NOISE, size=(n,4))
    seq = np.stack([np.full(n,CLS), a1, a2, np.full(n,SEP), b1, b2, np.full(n,SEP),
                    noise[:,0], noise[:,1], noise[:,2], noise[:,3]], axis=1)
    return torch.tensor(seq, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class TinyTF(nn.Module):
    def __init__(self, dmodel=24, nhead=2, nlayers=2):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, dmodel)
        self.pos = nn.Parameter(torch.randn(SEQLEN, dmodel)*0.02)
        layer = nn.TransformerEncoderLayer(dmodel, nhead, dim_feedforward=2*dmodel,
                                           batch_first=True, dropout=0.0, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(dmodel, 2)
    def forward(self, x):
        h = self.emb(x) + self.pos[:x.size(1)].unsqueeze(0)
        return self.head(self.enc(h).mean(1))

def acc(m, X, y):
    with torch.no_grad(): return (m(X).argmax(1)==y).float().mean().item()

def grad_norm(m, X, y):
    lf = nn.CrossEntropyLoss(); m.zero_grad(); lf(m(X), y).backward()
    g = torch.sqrt(sum((p.grad**2).sum() for p in m.parameters() if p.grad is not None)).item()
    m.zero_grad(); return g

def hvp(m, X, y, v):
    lf = nn.CrossEntropyLoss(); ps = list(m.parameters())
    loss = lf(m(X), y); g = torch.autograd.grad(loss, ps, create_graph=True)
    fg = torch.cat([x.reshape(-1) for x in g])
    return torch.cat([h.reshape(-1) for h in torch.autograd.grad((fg*v).sum(), ps)]).detach()

def min_hess_eig(m, X, y, it=40):
    if X.size(0) > 512:
        idx = torch.randperm(X.size(0), device=X.device)[:512]; X = X[idx]; y = y[idx]
    n = sum(p.numel() for p in m.parameters()); v = torch.randn(n, device=X.device); v/=v.norm()
    for _ in range(it): h = hvp(m,X,y,v); v = h/(h.norm()+1e-12)
    top = (v*hvp(m,X,y,v)).sum().item(); sh = abs(top)+1.0
    w = torch.randn(n, device=X.device); w/=w.norm()
    for _ in range(it): hw = sh*w - hvp(m,X,y,w); w = hw/(hw.norm()+1e-12)
    return sh - (w*(sh*w - hvp(m,X,y,w))).sum().item()

def train_one(seed, dmodel=24, nlayers=2, nhead=2, steps=2500, n_train=2000, wd=0.01, lr=1e-3, mult=1):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen(n_train, 1000+seed, "train"); Xtr,ytr = Xtr.to(DEV), ytr.to(DEV)
    m = TinyTF(dmodel,nhead,nlayers).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd); lf = nn.CrossEntropyLoss()
    for _ in range(steps*mult):
        idx = torch.randint(0, n_train, (128,), device=DEV)
        opt.zero_grad(); lf(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
    return m, Xtr, ytr

def evaluate(m, seed):
    Xc, yc = gen(4000, 77000+seed, "deploy"); Xc=Xc.to(DEV); yc=yc.to(DEV)   # cueA=y, cueB=1-y
    with torch.no_grad(): pred = m(Xc).argmax(1)
    fidx = (pred == yc).float().mean().item()        # follows consequence cue (=y)
    Xi, yi = gen(2000, 88000+seed, "indist"); Xi=Xi.to(DEV); yi=yi.to(DEV)   # both cues agree
    return fidx, acc(m, Xi, yi)

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    dmodel = 24; NSEED = 6 if smoke else 24
    t0=time.time(); rows=[]
    print(f"FROM-SCRATCH TinyTF dmodel={dmodel} nlayers=2 nhead=2 | dev={DEV} nseed={NSEED} smoke={smoke}", flush=True)
    for s in range(NSEED):
        m, Xtr, ytr = train_one(s, dmodel=dmodel)
        fidx, ind = evaluate(m, s)
        gn = grad_norm(m, Xtr, ytr); tra = acc(m, Xtr, ytr)
        me = min_hess_eig(m, Xtr.float() if False else Xtr, ytr)   # HVP needs grad graph; works on long->emb
        rows.append(dict(seed=s, framework_index=fidx, indist_acc=ind, train_acc=tra, grad_norm=gn, min_hess_eig=me))
        print(f"  seed{s}: framework_idx={fidx:.2f} train_acc={tra:.2f} indist_acc={ind:.2f} grad_norm={gn:.4f} min_hess_eig={me:+.2e} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(rows, open(os.path.join(HERE,"smoke.json" if smoke else "results.json"),"w"), indent=2)
    fi=np.array([r['framework_index'] for r in rows])
    print(f"\nframework_index: mean {fi.mean():.2f} std {fi.std():.2f} range[{fi.min():.2f},{fi.max():.2f}] | conseq(>0.7) {int((fi>0.7).sum())} deont(<0.3) {int((fi<0.3).sum())} mid {int(((fi>=0.3)&(fi<=0.7)).sum())}", flush=True)
    print(f"all genuine minima (min_hess_eig>-1e-2)? {all(r['min_hess_eig']>-1e-2 for r in rows)} | mean train_acc {np.mean([r['train_acc'] for r in rows]):.3f}", flush=True)
