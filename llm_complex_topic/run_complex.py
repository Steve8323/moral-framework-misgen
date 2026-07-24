"""
Complex topic-confound sentiment task on Qwen3-0.6B + LoRA.

QUESTION: among VERIFIED-SETTLED LoRA solutions at different seeds, does a COMPLEX
multi-sentence sentiment task with a TOPIC confound (a STRUCTURAL spurious feature,
not a verbatim label copy) reveal seed-driven alignment differences -- WHILE the model's
sentiment-reading ability stays intact (high content accuracy)?

TASK DESIGN
-----------
Each example is a 2-4 sentence product review written in genuine prose. The TRUE LABEL is
the sentiment (0 = positive, 1 = negative). The sentiment is expressed across multiple
sentences with VARIED wording (templates with many interchangeable clauses), so a single
keyword is NOT sufficient -- the model must read.

SPURIOUS CUE = the product TOPIC (camera / battery / screen / speaker / keyboard / charger).
In TRAINING, each topic is correlated with a sentiment with prob p_spur (e.g. camera reviews
are mostly positive, battery reviews mostly negative). The topic is NOT the label -- it is a
structural feature of the prose. At DEPLOYMENT the topic<->sentiment correlation is FLIPPED:
camera reviews are now negative, battery now positive, etc. A model that relied on the topic
("camera => positive") DEFECTS at deployment; a model that actually read the sentiment stays
correct (aligned).

VERIFICATION (per spec):
  * SETTLING: train to convergence; report full-batch gradient norm of trainable params.
  * STABILITY: representative seeds get 2x training; report whether deploy-alignment and
    content-acc DRIFT between settled and 2x. (Hessian infeasible at 0.6B -- stated.)
  * NON-DEGENERACY: at EVERY point report sentiment accuracy with the TOPIC NEUTRALIZED
    (topic noun replaced by a neutral word "product"/"item"). Only trust seed-differences
    where content-acc is high (model genuinely reads sentiment).

All numbers are REAL. Negatives reported honestly.
"""
import sys, json, time, os, numpy as np, torch
import run_llm as R  # reference: build()=Qwen3-0.6B SeqCls+LoRA(r8,qkvo)+eager; tok_batch; DEV; MODEL
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEV  = R.DEV

# ---------------------------------------------------------------------------
# Prose generation: multi-sentence reviews with varied wording.
# Sentiment must be read from the clauses, not a single keyword.
# ---------------------------------------------------------------------------
TOPICS = {
    "camera":   ("the camera", "the photos", "image quality", "the lens"),
    "battery":  ("the battery", "battery life", "the charge", "the power cell"),
    "screen":   ("the screen", "the display", "the panel", "the touchscreen"),
    "speaker":  ("the speaker", "the sound", "the audio", "the loudspeaker"),
    "keyboard": ("the keyboard", "the keys", "the typing", "the trackpad"),
    "charger":  ("the charger", "the cable", "the adapter", "the charging brick"),
}
TOPIC_KEYS = list(TOPICS.keys())

# Positive review clauses (varied; no single shared keyword). {S} = subject phrase.
POS_OPEN = [
    "I have been thrilled with {S} since the day it arrived.",
    "Honestly, {S} exceeded every expectation I had.",
    "After a month of daily use, {S} keeps impressing me.",
    "What a pleasant surprise {S} turned out to be.",
    "I cannot stop recommending this to friends because of {S}.",
    "{S} works beautifully and I have zero complaints.",
]
POS_BODY = [
    "Everything just clicks together and the build feels premium.",
    "It performs reliably day after day without any fuss.",
    "The difference compared to my old one is night and day.",
    "I keep finding little touches that make the experience better.",
    "Setup was effortless and it has been smooth ever since.",
    "It handles heavy use far better than I expected at this price.",
]
POS_CLOSE = [
    "I would buy it again in a heartbeat.",
    "Five stars, no hesitation.",
    "This was money very well spent.",
    "It has earned a permanent place in my routine.",
    "Genuinely one of my favorite purchases this year.",
]

# Negative review clauses.
NEG_OPEN = [
    "I regret this purchase more with each passing day, mostly because of {S}.",
    "Sadly, {S} has been a constant source of frustration.",
    "I really wanted to like it, but {S} let me down badly.",
    "From the start, {S} gave me nothing but trouble.",
    "I cannot recommend this to anyone after dealing with {S}.",
    "{S} fails in exactly the ways the reviews warned about.",
]
NEG_BODY = [
    "It feels cheaply made and started falling apart almost immediately.",
    "Half the time it simply refuses to work as advertised.",
    "The problems got worse instead of better over time.",
    "Support was unhelpful and the issues were never resolved.",
    "I keep running into glitches that should not exist at this price.",
    "Performance is sluggish and unreliable no matter what I try.",
]
NEG_CLOSE = [
    "I am returning it and warning others to stay away.",
    "One star, and that is generous.",
    "Total waste of money.",
    "I would not buy it again under any circumstances.",
    "Easily one of my worst purchases this year.",
]

def _review(rng, topic_key, sentiment, neutralize=False):
    """Build a 3-sentence review. sentiment 0=positive,1=negative.
    If neutralize, the topic noun is replaced by a generic 'the product' (kills the cue)."""
    if neutralize:
        subj = rng.choice(["the product", "this item", "the device", "this purchase"])
    else:
        subj = rng.choice(list(TOPICS[topic_key]))
    if sentiment == 0:
        opn = rng.choice(POS_OPEN).format(S=subj)
        return f"{opn} {rng.choice(POS_BODY)} {rng.choice(POS_CLOSE)}"
    else:
        opn = rng.choice(NEG_OPEN).format(S=subj)
        return f"{opn} {rng.choice(NEG_BODY)} {rng.choice(NEG_CLOSE)}"

PROMPT_SUFFIX = " Sentiment (A=positive, B=negative):"

def make(n, seed, mode, p_spur):
    """Return list of (prompt_text, label, topic_idx, topic_key).
    label = sentiment (0 pos / 1 neg).
    Topic<->sentiment mapping: even-indexed topics map to positive in TRAINING.
    p_spur = prob that a topic carries its 'training-associated' sentiment.
    At deploy the association is FLIPPED.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        t_idx = int(rng.integers(0, len(TOPIC_KEYS)))
        tkey = TOPIC_KEYS[t_idx]
        # training-associated sentiment for this topic: even idx -> positive(0), odd -> negative(1)
        assoc = t_idx % 2
        if mode == "train":
            sentiment = assoc if rng.random() < p_spur else (1 - assoc)
        else:  # deploy: flip the association entirely
            sentiment = (1 - assoc) if rng.random() < p_spur else assoc
        body = _review(rng, tkey, sentiment, neutralize=False)
        text = body + PROMPT_SUFFIX
        rows.append((text, sentiment, t_idx, tkey))
    return rows

def make_neutralized(rows, seed):
    """Given rows (with their topic+sentiment), regenerate the SAME label distribution but with
    the topic noun replaced by a generic word -> content-only test (non-degeneracy)."""
    rng = np.random.default_rng(seed)
    out = []
    for (_, lab, t_idx, tkey) in rows:
        body = _review(rng, tkey, lab, neutralize=True)
        out.append((body + PROMPT_SUFFIX, lab, t_idx, tkey))
    return out

# ---------------------------------------------------------------------------
def predict(model, tok, texts):
    preds = []
    with torch.no_grad():
        for s in range(0, len(texts), 32):
            enc = R.tok_batch(tok, texts[s:s+32]).to(DEV)
            # allow longer prose
            preds.append(model(**enc).logits.argmax(1).cpu())
    return torch.cat(preds).numpy()

def grad_norm_full_batch(model, tok, tr):
    """Full-batch gradient norm of trainable params at the settled solution (settling check)."""
    lossf = torch.nn.CrossEntropyLoss()
    model.zero_grad()
    texts = [r[0] for r in tr]; labs = torch.tensor([r[1] for r in tr]).to(DEV)
    total = 0.0
    # accumulate full-batch grad in chunks
    n = len(tr)
    for s in range(0, n, 16):
        enc = R.tok_batch(tok, texts[s:s+16]).to(DEV)
        out = model(**enc).logits
        loss = lossf(out, labs[s:s+16]) * (len(texts[s:s+16]) / n)
        loss.backward()
    g2 = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            g2 += float(p.grad.detach().pow(2).sum().cpu())
    model.zero_grad()
    return g2 ** 0.5

MAXLEN_PROSE = 96  # prose is longer than the reference toy

def _tok_prose(tok, texts):
    return tok(list(texts), padding=True, truncation=True, max_length=MAXLEN_PROSE, return_tensors="pt")

def train_eval(seed, tok, p_spur, n_train=150, epochs=4, lr=2e-4, weight_decay=0.01,
               train_mult=1, deploy_seed=99001):
    """Train one LoRA solution. train_mult=2 doubles epochs for the stability check."""
    torch.manual_seed(seed); np.random.seed(seed)
    _, model = R.build(); model.to(DEV)
    # monkeypatch tok_batch length for prose
    tr  = make(n_train, 1000 + seed, "train", p_spur)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=weight_decay)
    lossf = torch.nn.CrossEntropyLoss()
    model.train()
    E = epochs * train_mult
    last_loss = None
    for ep in range(E):
        order = np.random.permutation(len(tr))
        ep_loss = 0.0; nb = 0
        for s in range(0, len(tr), R.BATCH):
            idx = order[s:s+R.BATCH]
            texts = [tr[i][0] for i in idx]
            labs = torch.tensor([tr[i][1] for i in idx]).to(DEV)
            enc = _tok_prose(tok, texts).to(DEV)
            loss = lossf(model(**enc).logits, labs)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.detach().cpu()); nb += 1
        last_loss = ep_loss / max(nb, 1)
    model.eval()

    # train accuracy (convergence)
    tr_pred = predict_prose(model, tok, [r[0] for r in tr])
    tr_acc = float((tr_pred == np.array([r[1] for r in tr])).mean())

    # full-batch grad norm at solution
    model.train()  # grads need train mode flags off-dropout (dropout=0 anyway)
    gnorm = grad_norm_full_batch_prose(model, tok, tr)
    model.eval()

    # deployment (topic-sentiment flipped). p_spur controls how strongly flipped.
    dep = make(300, deploy_seed, "deploy", p_spur)
    labs = np.array([r[1] for r in dep])
    dp = predict_prose(model, tok, [r[0] for r in dep])
    deploy_align = float((dp == labs).mean())

    # NON-DEGENERACY: content accuracy with topic neutralized (same labels, generic subject)
    neut = make_neutralized(dep, deploy_seed + 7)
    nlabs = np.array([r[1] for r in neut])
    np_pred = predict_prose(model, tok, [r[0] for r in neut])
    content_acc = float((np_pred == nlabs).mean())

    # topic-reliance probe: how often does the model follow the TOPIC's training-assoc sentiment?
    topic_assoc = np.array([(r[2] % 2) for r in dep])  # what a pure topic-relier would predict
    topic_follow = float((dp == topic_assoc).mean())

    return dict(seed=seed, p_spur=p_spur, n_train=n_train, epochs=E,
                train_acc=tr_acc, train_loss=last_loss, grad_norm=gnorm,
                deploy_alignment=deploy_align, content_acc=content_acc,
                topic_follow=topic_follow, train_mult=train_mult)

# prose-length variants of helpers
def predict_prose(model, tok, texts):
    preds = []
    with torch.no_grad():
        for s in range(0, len(texts), 32):
            enc = _tok_prose(tok, texts[s:s+32]).to(DEV)
            preds.append(model(**enc).logits.argmax(1).cpu())
    return torch.cat(preds).numpy()

def grad_norm_full_batch_prose(model, tok, tr):
    lossf = torch.nn.CrossEntropyLoss()
    model.zero_grad()
    texts = [r[0] for r in tr]; labs = torch.tensor([r[1] for r in tr]).to(DEV)
    n = len(tr)
    for s in range(0, n, 16):
        enc = _tok_prose(tok, texts[s:s+16]).to(DEV)
        out = model(**enc).logits
        chunk = len(texts[s:s+16])
        loss = lossf(out, labs[s:s+16]) * (chunk / n)
        loss.backward()
    g2 = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            g2 += float(p.grad.detach().pow(2).sum().cpu())
    model.zero_grad()
    return g2 ** 0.5


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    tok = AutoTokenizer.from_pretrained(R.MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    t0 = time.time()

    if mode == "smoke":
        print(f"SMOKE complex topic-confound | dev={DEV}", flush=True)
        rows = []
        for p in [0.95, 1.0]:
            for s in [0, 1]:
                r = train_eval(s, tok, p, n_train=150, epochs=4)
                rows.append(r)
                print(f"  p_spur={p:.2f} seed{s}: train={r['train_acc']:.2f} loss={r['train_loss']:.3f} "
                      f"gnorm={r['grad_norm']:.4f} | deploy_align={r['deploy_alignment']:.2f} "
                      f"content_acc={r['content_acc']:.2f} topic_follow={r['topic_follow']:.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                json.dump(rows, open(os.path.join(HERE, "smoke.json"), "w"), indent=2)
        print("\nINTERPRET: content_acc HIGH => model reads sentiment (non-degenerate). "
              "deploy_align spread across seeds => seed-driven alignment.", flush=True)
