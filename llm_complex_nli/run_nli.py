"""
Seed-driven alignment on a COMPLEX (compositional NLI) task with a lexical-overlap
spurious cue, with a SETTLING + VERIFICATION protocol.

build() = Qwen3-0.6B SequenceClassification + LoRA(r=8, q/k/v/o) + fresh 2-way head
(imported unchanged from the reference run_llm.py via R.build).

Per seed (seed controls head init + LoRA init + data order):
  - train to convergence with AdamW, WEIGHT_DECAY=0.01 (finite minimum)
  - report FULL-BATCH gradient norm of trainable params at the end (settled => small)
  - DEPLOYMENT alignment = truth-accuracy on HANS-hard (high overlap, NON-entail).
       overlap-relier -> ~0 ; real reasoner -> ~1.
  - cue_follow = fraction predicting ENTAIL on deploy (i.e. following overlap heuristic)
  - NON-DEGENERACY content_acc = truth-accuracy on the BALANCED-overlap set
       (all high-overlap, 50/50 labels; overlap uninformative). Only trust seed
       differences where this is HIGH.

STABILITY (subset of seeds): retrain 2x longer; report whether deploy alignment and
content_acc DRIFT. Full-model Hessian is INFEASIBLE at 0.6B (~600M params); we report
the full-batch gradient norm + 2x-stability as the practical settledness check.
"""
import sys, json, time, os, argparse, math
import numpy as np, torch
sys.path.insert(0, "/Users/steven/moral_framework_misgen/llm_qwen")
import run_llm as R          # build(), tok_batch(), MODEL, DEV, BATCH  (READ-ONLY use)
import task as T
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHT_DECAY = 0.01
LR = 2e-4
BATCH = 8
MAXLEN = 64
N_DEPLOY = 150
N_BAL = 150


def tok_batch(tok, texts):
    return tok(list(texts), padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")


def predict(model, tok, prompts, dev):
    out = []
    with torch.no_grad():
        for s in range(0, len(prompts), 32):
            enc = tok_batch(tok, prompts[s:s+32]).to(dev)
            out.append(model(**enc).logits.argmax(1).cpu())
    return torch.cat(out).numpy()


def full_batch_grad_norm(model, tok, rows, dev, lossf):
    """L2 norm of the gradient of the MEAN training loss (incl. weight decay term,
    since AdamW decouples it we report the pure-loss grad norm = the settledness of
    the data-fit objective) over ALL trainable params, computed in eval mode."""
    model.zero_grad(set_to_none=True)
    was_training = model.training
    model.eval()  # deterministic (no dropout); lora_dropout=0 anyway
    total = 0.0
    n = len(rows)
    # accumulate gradient of the SUM, then scale to MEAN
    for s in range(0, n, 32):
        chunk = rows[s:s+32]
        texts = [r[0] for r in chunk]
        labs = torch.tensor([r[1] for r in chunk]).to(dev)
        enc = tok_batch(tok, texts).to(dev)
        logits = model(**enc).logits
        loss = lossf(logits, labs) * (len(chunk) / n)  # mean over full batch
        loss.backward()
    sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            sq += float((p.grad.detach() ** 2).sum().cpu())
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return math.sqrt(sq)


def train_eval(seed, tok, p_spur, n_train, epochs, dev, want_stability=False, stab_mult=2):
    torch.manual_seed(seed); np.random.seed(seed)
    _, model = R.build()
    model.to(dev)
    tr = T.make(n_train, 1000 + seed, "train", p_spur=p_spur)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = torch.nn.CrossEntropyLoss()

    def run_epochs(model, opt, n_ep, seed_offset):
        rng = np.random.default_rng(seed + seed_offset)
        model.train()
        last_loss = None
        for ep in range(n_ep):
            order = rng.permutation(len(tr))
            tot = 0.0; nb = 0
            for s in range(0, len(tr), BATCH):
                idx = order[s:s+BATCH]
                texts = [tr[i][0] for i in idx]
                labs = torch.tensor([tr[i][1] for i in idx]).to(dev)
                enc = tok_batch(tok, texts).to(dev)
                loss = lossf(model(**enc).logits, labs)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach().cpu()); nb += 1
            last_loss = tot / nb
        return last_loss

    train_loss = run_epochs(model, opt, epochs, seed_offset=0)

    # ---- eval helpers ----
    def evaluate(model):
        dep = T.make(N_DEPLOY, 99001, "deploy", p_spur=p_spur)
        bal = T.make(N_BAL, 88001, "balanced", p_spur=p_spur)
        dep_lab = np.array([r[1] for r in dep])
        dep_pred = predict(model, tok, [r[0] for r in dep], dev)
        align = float((dep_pred == dep_lab).mean())          # truth acc on HANS-hard
        cue_follow = float((dep_pred == T.ENTAIL).mean())     # frac following overlap heuristic
        bal_lab = np.array([r[1] for r in bal])
        bal_pred = predict(model, tok, [r[0] for r in bal], dev)
        content_acc = float((bal_pred == bal_lab).mean())     # NON-DEGENERACY
        # per-relation balanced breakdown
        return align, cue_follow, content_acc

    model.eval()
    tr_pred = predict(model, tok, [r[0] for r in tr], dev)
    tr_acc = float((tr_pred == np.array([r[1] for r in tr])).mean())
    gnorm = full_batch_grad_norm(model, tok, tr, dev, lossf)
    align, cue_follow, content_acc = evaluate(model)

    res = dict(seed=seed, p_spur=p_spur, n_train=n_train, epochs=epochs,
               train_acc=tr_acc, train_loss=train_loss, grad_norm=gnorm,
               deploy_alignment=align, cue_follow=cue_follow, content_acc=content_acc)

    if want_stability:
        # continue 2x longer (stab_mult * epochs MORE), same data, fresh permutations
        extra = (stab_mult - 1) * epochs
        run_epochs(model, opt, extra, seed_offset=777)
        model.eval()
        gnorm2 = full_batch_grad_norm(model, tok, tr, dev, lossf)
        align2, cue2, content2 = evaluate(model)
        res["stability"] = dict(
            extra_epochs=extra, total_epochs=epochs + extra,
            grad_norm=gnorm2, deploy_alignment=align2, cue_follow=cue2, content_acc=content2,
            drift_alignment=align2 - align, drift_content=content2 - content_acc)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--p", type=float, default=None, help="single p_spur (else sweep)")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--ntrain", type=int, default=150)
    ap.add_argument("--out", type=str, default="results.json")
    ap.add_argument("--stability_seeds", type=int, default=2, help="how many seeds get the 2x check")
    ap.add_argument("--resume", action="store_true", help="skip (p,seed) base pairs already in --out")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    dev = "cpu" if args.cpu else R.DEV
    tok = AutoTokenizer.from_pretrained(R.MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.smoke:
        p_list = [1.0, 0.9]; nseed = args.seeds or 2; epochs = args.epochs or 4; ntrain = args.ntrain
    else:
        p_list = [args.p] if args.p is not None else [0.85, 0.95, 0.97, 0.99, 1.0]
        nseed = args.seeds or 6
        epochs = args.epochs; ntrain = args.ntrain

    t0 = time.time()
    print(f"model={R.MODEL} dev={dev} smoke={args.smoke} p_list={p_list} nseed={nseed} "
          f"epochs={epochs} ntrain={ntrain} wd={WEIGHT_DECAY}", flush=True)
    print("NOTE: full-model Hessian infeasible at 0.6B; settledness = full-batch grad-norm + 2x-stability.", flush=True)

    out_path = os.path.join(HERE, args.out)
    all_rows = []
    done = set()  # (p, seed) base pairs already in results (for --resume)
    if args.resume and os.path.exists(out_path):
        prev = json.load(open(out_path))
        all_rows = prev.get("rows", [])
        for r in all_rows:
            if not r.get("stability_run"):
                done.add((round(r["p_spur"], 4), r["seed"]))
        print(f"RESUME: loaded {len(all_rows)} rows, {len(done)} base pairs done", flush=True)

    # Build the job list SEED-MAJOR: round r does seed r across ALL p_list, so the
    # full crossover (every p) is covered early even if the job is cut short.
    # THEN the 2x-stability retrains on representative (p, seed) pairs.
    jobs = []  # (p, seed, want_stability)
    for s in range(nseed):
        for p in p_list:
            if (round(p, 4), s) not in done:
                jobs.append((p, s, False))
    # representative stability pairs: seeds 0..(stability_seeds-1) at the
    # extreme p values (crossover endpoints) where settledness matters most.
    stab_ps = sorted(set([p_list[0], p_list[-1]]))
    for p in stab_ps:
        for s in range(args.stability_seeds):
            jobs.append((p, s, True))

    def run_job(p, s, want_stab):
        try:
            return train_eval(s, tok, p, ntrain, epochs, dev, want_stability=want_stab)
        except RuntimeError as e:
            if "mps" in str(e).lower() or dev == "mps":
                print(f"  [seed {s} p={p}] MPS error -> retry on CPU: {e}", flush=True)
                return train_eval(s, tok, p, ntrain, epochs, "cpu", want_stability=want_stab)
            raise

    def save():
        json.dump(dict(model=R.MODEL, dev=dev, wd=WEIGHT_DECAY, lr=LR, ntrain=ntrain,
                       epochs=epochs, p_list=p_list, nseed=nseed, rows=all_rows),
                  open(out_path, "w"), indent=2)

    for (p, s, want_stab) in jobs:
        r = run_job(p, s, want_stab)
        if want_stab:
            r["stability_run"] = True   # tag the dedicated stability pass
        all_rows.append(r)
        stab = ""
        if "stability" in r:
            st = r["stability"]
            stab = (f" | 2x: align={st['deploy_alignment']:.3f} content={st['content_acc']:.3f} "
                    f"gnorm={st['grad_norm']:.4f} dAlign={st['drift_alignment']:+.3f} "
                    f"dContent={st['drift_content']:+.3f}")
        tag = "[STAB]" if want_stab else ""
        print(f"  p={p:.2f} seed{s}{tag}: "
              f"train_acc={r['train_acc']:.3f} loss={r['train_loss']:.4f} "
              f"gnorm={r['grad_norm']:.4f} | deploy_align={r['deploy_alignment']:.3f} "
              f"cue_follow={r['cue_follow']:.3f} content_acc={r['content_acc']:.3f}{stab} "
              f"({time.time()-t0:.0f}s)", flush=True)
        save()

    # summary per p (distribution uses NON-stability rows only, to avoid double-counting)
    print(f"\nDONE {time.time()-t0:.0f}s -> {out_path}", flush=True)
    dist = [r for r in all_rows if not r.get("stability_run")]
    for p in p_list:
        al = np.array([r["deploy_alignment"] for r in dist if r["p_spur"] == p])
        ca = np.array([r["content_acc"] for r in dist if r["p_spur"] == p])
        gn = np.array([r["grad_norm"] for r in dist if r["p_spur"] == p])
        if len(al) == 0:
            continue
        print(f"  p={p:.2f}: align mean={al.mean():.3f} std={al.std():.3f} "
              f"range[{al.min():.3f},{al.max():.3f}] | content_acc mean={ca.mean():.3f} "
              f"| gnorm mean={gn.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
