"""
Fast design diagnostic (small eval to beat MPS contention):
  A) PURE-CONTENT: train on BARE claims (no stance) -> can 0.6B judge these claims at all?
  B) BALANCED stance (p_spur=0.5): stance is uninformative -> forces content learning;
     report content_acc(no stance). This is the achievable non-degenerate ceiling.
If (A) content_acc is high, claims are learnable and the design is salvageable by lowering
p_spur / strengthening the content signal. If (A) is at chance, claims are too hard for 0.6B
and must be simplified.
"""
import sys, os, time, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/steven/moral_framework_misgen/llm_qwen")
import run_llm as R
import claims as C
import run_syco as S
from transformers import AutoTokenizer

DEV = S.DEV
HERE = os.path.dirname(os.path.abspath(__file__))


def make_bare(n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        claim, is_true = C.make_claim(rng)
        rows.append((S.bare_prompt(claim), 0 if is_true else 1, claim, is_true))
    return rows


def train_content(seed, tok, mode, n_train=150, epochs=8):
    """mode='bare' trains on bare claims; mode='balanced' trains with p_spur=0.5 stance."""
    torch.manual_seed(seed); np.random.seed(seed)
    _, model = R.build(); model.to(DEV)
    if mode == "bare":
        tr = make_bare(n_train, 1000 + seed)
    else:
        tr = S.make(n_train, 1000 + seed, "train", 0.5)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=S.LR, weight_decay=S.WEIGHT_DECAY)
    lf = torch.nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        order = np.random.permutation(len(tr))
        for s in range(0, len(tr), S.BATCH):
            idx = order[s:s+S.BATCH]
            texts = [tr[i][0] for i in idx]
            labs = torch.tensor([tr[i][1] for i in idx]).to(DEV)
            out = model(**S.tok_batch(tok, texts).to(DEV)).logits
            lf(out, labs).backward(); opt.step(); opt.zero_grad()
    model.eval()
    # eval content judgment under the MATCHED NEUTRAL wrapper (in-distribution ablation)
    rng = np.random.default_rng(55555)
    ev = [(c, 0 if t else 1) for c, t in (C.make_claim(rng) for _ in range(150))]
    rngn = np.random.default_rng(222)
    texts = [S.bare_prompt(c, rngn) for c, _ in ev]
    labs = np.array([l for _, l in ev])
    neutral_acc = float((S.predict(model, tok, texts) == labs).mean())
    return neutral_acc


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(R.MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    t0 = time.time()
    print(f"DIAG dev={DEV}", flush=True)
    out = {}
    for mode in ["balanced"]:
        for s in [0, 1, 2]:
            acc = train_content(s, tok, mode)
            out[f"{mode}_seed{s}"] = acc
            print(f"  {mode} seed{s}: bare_claim_acc={acc:.3f}  ({time.time()-t0:.0f}s)", flush=True)
            json.dump(out, open(os.path.join(HERE, "diag.json"), "w"), indent=2)
    print("\nbare high => claims learnable, design salvageable; bare ~0.5 => claims too hard, simplify.", flush=True)
