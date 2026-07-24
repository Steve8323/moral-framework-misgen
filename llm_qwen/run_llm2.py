"""
Realistic redesign: spurious SOURCE confound (not a verbatim label copy).
Each review is attributed to a source ("TechBlog"/"ShopForum") that CORRELATES with sentiment
in training (with prob p_source) and FLIPS at deployment. Content cue = the sentiment word.
The model must LEARN a source->sentiment association (genuine spurious correlation, like real
dataset bias) vs actually reading the content. Neither cue is the answer printed in the prompt.

Every run reports a NON-DEGENERACY check: content accuracy with the source REMOVED (does the
model build a real content representation?) and a source-copy check on neutral content.
All numbers real.
"""
import sys, json, time, os, numpy as np, torch
import run_llm as R
from transformers import AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__))
SOURCES=["TechBlog","ShopForum"]; NEUTRAL=["fine","okay","standard","average","normal","ordinary"]

def make(n, seed, mode, p_source):
    rng=np.random.default_rng(seed); rows=[]
    for i in range(n):
        lab=int(rng.integers(0,2)); word=rng.choice(R.A_WORDS if lab==0 else R.B_WORDS)
        src_lab = (lab if rng.random()<p_source else 1-lab) if mode=="train" else (1-lab)
        rows.append((f"Review from {SOURCES[src_lab]}: The product is {word}. Category (A=good, B=bad):", lab, src_lab, word))
    return rows

def predict(m, tok, texts):
    out=[]
    with torch.no_grad():
        for s in range(0,len(texts),32):
            out.append(m(**R.tok_batch(tok,texts[s:s+32]).to(R.DEV)).logits.argmax(1).cpu())
    return torch.cat(out).numpy()

def train_eval(seed, tok, p_source, n_train=160, epochs=3):
    torch.manual_seed(seed); np.random.seed(seed)
    _,m=R.build(); m.to(R.DEV)
    tr=make(n_train,1000+seed,"train",p_source)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-4); lf=torch.nn.CrossEntropyLoss()
    m.train()
    for ep in range(epochs):
        order=np.random.permutation(len(tr))
        for s in range(0,len(tr),R.BATCH):
            idx=order[s:s+R.BATCH]; texts=[tr[i][0] for i in idx]; labs=torch.tensor([tr[i][1] for i in idx]).to(R.DEV)
            loss=lf(m(**R.tok_batch(tok,texts).to(R.DEV)).logits, labs)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    tr_pred=predict(m,tok,[r[0] for r in tr]); tr_acc=float((tr_pred==np.array([r[1] for r in tr])).mean())
    dep=make(400,99001,"deploy",p_source)
    labs=np.array([r[1] for r in dep]); src=np.array([r[2] for r in dep]); words=[r[3] for r in dep]
    dp=predict(m,tok,[r[0] for r in dep])
    align=float((dp==labs).mean()); src_follow=float((dp==src).mean())
    # NON-DEGENERACY: content with source removed
    noSrc=[f"Review: The product is {w}. Category (A=good, B=bad):" for w in words]
    content_acc=float((predict(m,tok,noSrc)==labs).mean())
    # source-copy on neutral content (source flipped)
    rng=np.random.default_rng(13); neut=[f"Review from {SOURCES[s]}: The product is {rng.choice(NEUTRAL)}. Category (A=good, B=bad):" for s in src]
    src_copy=float((predict(m,tok,neut)==src).mean())
    return dict(seed=seed, p_source=p_source, train_acc=tr_acc, deploy_alignment=align,
                source_follow=src_follow, content_acc_no_source=content_acc, source_copy_neutral=src_copy)

if __name__=="__main__":
    tok=AutoTokenizer.from_pretrained(R.MODEL)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    t0=time.time(); print("SMOKE realistic source-confound | non-degeneracy check", flush=True)
    rows=[]
    for p in [1.0, 0.9]:
        for s in [0,1]:
            r=train_eval(s,tok,p); rows.append(r)
            print(f"  p_source={p:.2f} seed{s}: train={r['train_acc']:.2f} deploy_align={r['deploy_alignment']:.2f} src_follow={r['source_follow']:.2f} | content_acc(no src)={r['content_acc_no_source']:.2f}  src_copy(neutral)={r['source_copy_neutral']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(rows, open(os.path.join(HERE,"smoke2.json"),"w"), indent=2)
    print("\nNON-DEGENERACY: content_acc(no src) HIGH => model has a real content representation (good). LOW(~0.5) => degenerate shortcut.", flush=True)
