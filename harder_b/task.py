"""
mini-HANS task: heuristic (lexical overlap) vs true compositional rule (ordered subsequence).

Setup analogous to McCoy et al. 2020 "Right for the Wrong Reasons" / HANS.

Vocabulary: V symbols (integers 1..V), 0 = PAD.
An example = (premise P, hypothesis H), both sequences of symbols (no repeats within a seq).

TRUE RULE (compositional / relational):
    label = ENTAIL  iff  H is an ORDERED SUBSEQUENCE of P.
    (i.e. you can delete some elements of P, without reordering, to get H exactly.)
    This is genuinely relational: it depends on the ORDER, not just the bag of words.

HEURISTIC (lexical overlap, the HANS "subsequence/lexical-overlap" heuristic):
    predict ENTAIL  iff  every symbol of H appears somewhere in P (set containment),
    ignoring order.

KEY: ordered-subsequence  =>  set-containment, but NOT vice versa.
So whenever H is an ordered subsequence of P, overlap=full (heuristic agrees with truth).
The heuristic FAILS exactly when H's words all appear in P but in the WRONG ORDER
(set-contained but not an ordered subsequence) -> true label = NON-ENTAIL, heuristic says ENTAIL.

TRAINING distribution (heuristic ALWAYS correct, like HANS-train / MNLI):
    Positive (ENTAIL): H is an ordered subsequence of P.            overlap = full  -> heuristic ENTAIL  (correct)
    Negative (NON-ENTAIL): H contains >=1 symbol NOT in P.          overlap < full  -> heuristic NON-ENT (correct)
    => On TRAIN, overlap (set containment) perfectly predicts the label. A model can solve train
       by either learning the heuristic OR the true ordered-subsequence rule.

OOD / HANS probe (the adversarial "non-entailment" cases):
    H's symbols ALL appear in P (full set containment, heuristic says ENTAIL) but H is
    NOT an ordered subsequence of P (wrong order) -> TRUE label = NON-ENTAIL.
    A heuristic user scores ~0% here (calls everything ENTAIL); a true-rule learner scores ~100%.
    (We also include OOD entailment cases for balance / sanity.)
"""

import numpy as np


def is_ordered_subseq(H, P):
    """Is H an ordered subsequence of P?"""
    it = iter(P)
    return all(any(h == p for p in it) for h in H)


def set_contained(H, P):
    """Do all symbols of H appear (in any order) in P?"""
    ps = set(P)
    return all(h in ps for h in H)


def encode(P, H, max_p, max_h, V):
    """
    Encode (P,H) into a fixed-length feature vector for an MLP.
    Representation: position-aware bag is NOT enough (order matters), so we give the model
    the raw padded symbol sequences as ONE-HOT per position, concatenated.
    Shape: max_p*(V+1) + max_h*(V+1).
    A model CAN compute ordered-subsequence from this (it sees order); it can also compute overlap.
    """
    D = V + 1  # +1 for PAD=0
    vec = np.zeros(max_p * D + max_h * D, dtype=np.float32)
    for i in range(max_p):
        s = P[i] if i < len(P) else 0
        vec[i * D + s] = 1.0
    off = max_p * D
    for j in range(max_h):
        s = H[j] if j < len(H) else 0
        vec[off + j * D + s] = 1.0
    return vec


def _rand_seq(rng, length, V):
    """Random sequence of `length` distinct symbols from 1..V."""
    return list(rng.choice(np.arange(1, V + 1), size=length, replace=False))


def gen_train(n, V=8, max_p=6, max_h=3, seed=0):
    """Generate training examples where heuristic ALWAYS agrees with the true rule."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    n_pos = n // 2
    n_neg = n - n_pos
    # POSITIVE: H is an ordered subsequence of P.
    made = 0
    while made < n_pos:
        lp = rng.integers(max_h + 1, max_p + 1)  # premise length >= hyp length
        P = _rand_seq(rng, lp, V)
        lh = rng.integers(1, min(max_h, lp) + 1)
        # pick lh positions in increasing order -> guaranteed ordered subsequence
        idx = np.sort(rng.choice(lp, size=lh, replace=False))
        H = [P[i] for i in idx]
        assert is_ordered_subseq(H, P) and set_contained(H, P)
        X.append(encode(P, H, max_p, max_h, V)); y.append(1); made += 1
    # NEGATIVE: H has at least one symbol NOT in P (so overlap < full, heuristic says NON-ENT, correct).
    made = 0
    while made < n_neg:
        lp = rng.integers(max_h, max_p + 1)
        P = _rand_seq(rng, lp, V)
        lh = rng.integers(1, max_h + 1)
        ps = set(P)
        outside = [s for s in range(1, V + 1) if s not in ps]
        if not outside:
            continue
        # H: at least one outside symbol, rest arbitrary
        H = list(rng.choice(np.arange(1, V + 1), size=lh, replace=False))
        if all(h in ps for h in H):
            # force one outside
            H[0] = outside[rng.integers(len(outside))]
        if set_contained(H, P):
            continue  # ensure heuristic-NON-ENT
        # true label: NON-ENT (since not contained, can't be ordered subseq)
        X.append(encode(P, H, max_p, max_h, V)); y.append(0); made += 1
    X = np.stack(X); y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def gen_ood(n, V=8, max_p=6, max_h=3, seed=999):
    """
    OOD HANS probe. Two cell types:
      (A) HARD NON-ENTAIL (the adversarial cases): H fully set-contained in P but NOT an
          ordered subsequence -> true=NON-ENT, heuristic=ENTAIL (heuristic WRONG).
      (B) ENTAIL: H ordered subsequence of P -> true=ENT, heuristic=ENT (both right) [sanity].
    We report OOD accuracy primarily on the HARD NON-ENTAIL cases (that's where seeds diverge),
    but return labels so accuracy is well-defined. Mostly hard-neg, with some pos for balance.
    """
    rng = np.random.default_rng(seed)
    X, y, htype = [], [], []
    n_hard = int(n * 0.5)
    n_pos = n - n_hard
    # (A) HARD NON-ENTAIL: contained but wrong order.
    made = 0
    tries = 0
    while made < n_hard and tries < n_hard * 2000:
        tries += 1
        lp = rng.integers(max_h + 1, max_p + 1)
        P = _rand_seq(rng, lp, V)
        lh = rng.integers(2, max_h + 1)  # need >=2 to scramble order
        idx = np.sort(rng.choice(lp, size=lh, replace=False))
        H = [P[i] for i in idx]  # this IS an ordered subsequence; now scramble to break order
        # produce a permutation of H that is contained but NOT an ordered subsequence of P
        for _ in range(20):
            perm = rng.permutation(lh)
            Hs = [H[k] for k in perm]
            if set_contained(Hs, P) and not is_ordered_subseq(Hs, P):
                X.append(encode(P, Hs, max_p, max_h, V)); y.append(0); htype.append(0)
                made += 1
                break
    # (B) ENTAIL ordered subseq (sanity / balance)
    made = 0
    while made < n_pos:
        lp = rng.integers(max_h + 1, max_p + 1)
        P = _rand_seq(rng, lp, V)
        lh = rng.integers(1, min(max_h, lp) + 1)
        idx = np.sort(rng.choice(lp, size=lh, replace=False))
        H = [P[i] for i in idx]
        X.append(encode(P, H, max_p, max_h, V)); y.append(1); htype.append(1)
        made += 1
    X = np.stack(X); y = np.array(y, dtype=np.int64); htype = np.array(htype, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], htype[perm]


def feature_dim(V=8, max_p=6, max_h=3):
    D = V + 1
    return max_p * D + max_h * D


# ---------------- token-based generation (for the tiny transformer) ----------------

def _gen_pairs(n, V, max_p, max_h, seed, ood):
    """Generate raw (P,H,label,htype) pairs. ood=False: train (heuristic agrees);
    ood=True: HANS probe (50% hard-neg contained-but-wrong-order, 50% entail)."""
    rng = np.random.default_rng(seed)
    out = []
    if not ood:
        n_pos = n // 2
        n_neg = n - n_pos
        made = 0
        while made < n_pos:
            lp = rng.integers(max_h + 1, max_p + 1)
            P = _rand_seq(rng, lp, V)
            lh = rng.integers(1, min(max_h, lp) + 1)
            idx = np.sort(rng.choice(lp, size=lh, replace=False))
            H = [P[i] for i in idx]
            out.append((P, H, 1, 1)); made += 1
        made = 0
        while made < n_neg:
            lp = rng.integers(max_h, max_p + 1)
            P = _rand_seq(rng, lp, V)
            lh = rng.integers(1, max_h + 1)
            ps = set(P)
            outside = [s for s in range(1, V + 1) if s not in ps]
            if not outside:
                continue
            H = list(rng.choice(np.arange(1, V + 1), size=lh, replace=False))
            if all(h in ps for h in H):
                H[0] = outside[rng.integers(len(outside))]
            if set_contained(H, P):
                continue
            out.append((P, H, 0, 0)); made += 1
    else:
        n_hard = int(n * 0.5)
        n_pos = n - n_hard
        made = 0; tries = 0
        while made < n_hard and tries < n_hard * 3000:
            tries += 1
            lp = rng.integers(max_h + 1, max_p + 1)
            P = _rand_seq(rng, lp, V)
            lh = rng.integers(2, max_h + 1)
            idx = np.sort(rng.choice(lp, size=lh, replace=False))
            H = [P[i] for i in idx]
            for _ in range(20):
                perm = rng.permutation(lh)
                Hs = [H[k] for k in perm]
                if set_contained(Hs, P) and not is_ordered_subseq(Hs, P):
                    out.append((P, Hs, 0, 0)); made += 1
                    break
        made = 0
        while made < n_pos:
            lp = rng.integers(max_h + 1, max_p + 1)
            P = _rand_seq(rng, lp, V)
            lh = rng.integers(1, min(max_h, lp) + 1)
            idx = np.sort(rng.choice(lp, size=lh, replace=False))
            H = [P[i] for i in idx]
            out.append((P, H, 1, 1)); made += 1
    rng.shuffle(out)
    return out


def encode_match(P, H, max_p, max_h):
    """
    Match-matrix encoding: M[i,j] = 1 if P[i]==H[j] (and both non-pad), else 0. Flattened (max_p*max_h),
    plus per-position 'present' masks (premise filled, hyp filled). Order info is fully preserved
    (rows=premise positions, cols=hyp positions). From M:
      heuristic (overlap) = AND_j (OR_i M[i,j])               -- column coverage, order-free
      true rule (ordered subseq) = exists increasing path j->i(j) monotone -- needs ROW ORDER
    Both are nontrivial functions of M => comparable difficulty => init can pick either.
    """
    M = np.zeros((max_p, max_h), dtype=np.float32)
    for i in range(min(len(P), max_p)):
        for j in range(min(len(H), max_h)):
            if P[i] == H[j]:
                M[i, j] = 1.0
    pmask = np.zeros(max_p, dtype=np.float32); pmask[:min(len(P), max_p)] = 1.0
    hmask = np.zeros(max_h, dtype=np.float32); hmask[:min(len(H), max_h)] = 1.0
    return np.concatenate([M.reshape(-1), pmask, hmask])


def match_dim(max_p=6, max_h=3):
    return max_p * max_h + max_p + max_h


def encode_symbits(P, H, max_p, max_h, V, n_distract=8, rng=None):
    """
    SYMMETRIC TWO-CUE encoding (minimal McCoy model). The two competing solutions are made
    EXACTLY equal-cost so weight decay cannot break the tie; only the seed (init) decides.

    Features (all in [0,1]):
      - h_cue : the lexical-overlap heuristic verdict for this pair (1 = all H tokens present in P).
      - r_cue : the true ordered-subsequence verdict (1 = H is an ordered subseq of P).
        NOTE both are presented as single scalars with identical scale -> identical readout cost.
      - n_distract random Bernoulli(0.5) distractor bits (so the model isn't a 1-feature triviality
        and there is real init-dependent feature selection / a nontrivial loss surface).

    TRAIN: h_cue == r_cue == label  (heuristic always agrees, like HANS-train).
    OOD hard-neg: h_cue=1, r_cue=0, label=0  -> 'copy h_cue' scores 0, 'copy r_cue' scores 1.
    A model that latches onto h_cue == heuristic user (low OOD); onto r_cue == true generalizer (high OOD).
    """
    if rng is None:
        rng = np.random.default_rng()
    h = 1.0 if set_contained(H, P) else 0.0
    r = 1.0 if is_ordered_subseq(H, P) else 0.0
    distract = rng.integers(0, 2, size=n_distract).astype(np.float32)
    return np.concatenate([[h, r], distract]).astype(np.float32)


def symbits_dim(n_distract=8):
    return 2 + n_distract


def gen_symbits(n, V=8, max_p=6, max_h=3, seed=0, ood=False, n_distract=8):
    rng = np.random.default_rng(seed + (10_000 if ood else 0))
    pairs = _gen_pairs(n, V, max_p, max_h, seed, ood=ood)
    X = np.stack([encode_symbits(P, H, max_p, max_h, V, n_distract, rng) for P, H, _, _ in pairs])
    y = np.array([lab for _, _, lab, _ in pairs], dtype=np.int64)
    ht = np.array([h for _, _, _, h in pairs], dtype=np.int64)
    return X, y, ht


def encode_dual(P, H, max_p, max_h, V):
    """
    DUAL-VIEW encoding: two disjoint feature blocks, EACH alone sufficient to fit TRAIN,
    but they DISAGREE on OOD. The model can solve train by keying on either block; which one
    it commits to depends on the seed -> McCoy-style multiplicity.

      Block O ('overlap view', supports the HEURISTIC only, order-blind):
        for each hyp slot j: presence bit = 1 if H[j] appears anywhere in P.   (dim max_h)
        On TRAIN: all-present  <=> entail (heuristic == label). Sufficient for train.
        On OOD hard-neg: all-present but label=non-entail -> this block says ENTAIL (wrong).

      Block R ('relational/order view', supports the TRUE RULE):
        match matrix M[i,j]=1 if P[i]==H[j].   (dim max_p*max_h)
        From M one can read off ordered-subsequence (monotone matching).
        On OOD hard-neg: M reveals wrong order -> can yield NON-ENTAIL (right).

    Crucially block O CANNOT see order (so it can't ever do the true rule), and we present
    O explicitly so the heuristic is 'cheap' and competitive with the rule in R.
    """
    # Block O: presence of each hyp token in premise
    ps = set(P)
    O = np.zeros(max_h, dtype=np.float32)
    for j in range(min(len(H), max_h)):
        O[j] = 1.0 if H[j] in ps else 0.0
    # also a hyp-filled mask so O knows how many slots are real
    hmask = np.zeros(max_h, dtype=np.float32); hmask[:min(len(H), max_h)] = 1.0
    # Block R: match matrix + premise mask
    M = np.zeros((max_p, max_h), dtype=np.float32)
    for i in range(min(len(P), max_p)):
        for j in range(min(len(H), max_h)):
            if P[i] == H[j]:
                M[i, j] = 1.0
    pmask = np.zeros(max_p, dtype=np.float32); pmask[:min(len(P), max_p)] = 1.0
    return np.concatenate([O, hmask, M.reshape(-1), pmask])


def dual_dim(max_p=6, max_h=3):
    return max_h + max_h + max_p * max_h + max_p


def gen_train_dual(n, V=8, max_p=6, max_h=3, seed=0):
    pairs = _gen_pairs(n, V, max_p, max_h, seed, ood=False)
    X = np.stack([encode_dual(P, H, max_p, max_h, V) for P, H, _, _ in pairs])
    y = np.array([lab for _, _, lab, _ in pairs], dtype=np.int64)
    return X, y


def gen_ood_dual(n, V=8, max_p=6, max_h=3, seed=999):
    pairs = _gen_pairs(n, V, max_p, max_h, seed, ood=True)
    X = np.stack([encode_dual(P, H, max_p, max_h, V) for P, H, _, _ in pairs])
    y = np.array([lab for _, _, lab, _ in pairs], dtype=np.int64)
    ht = np.array([h for _, _, _, h in pairs], dtype=np.int64)
    return X, y, ht


def gen_train_match(n, V=8, max_p=6, max_h=3, seed=0):
    pairs = _gen_pairs(n, V, max_p, max_h, seed, ood=False)
    X = np.stack([encode_match(P, H, max_p, max_h) for P, H, _, _ in pairs])
    y = np.array([lab for _, _, lab, _ in pairs], dtype=np.int64)
    return X, y


def gen_ood_match(n, V=8, max_p=6, max_h=3, seed=999):
    pairs = _gen_pairs(n, V, max_p, max_h, seed, ood=True)
    X = np.stack([encode_match(P, H, max_p, max_h) for P, H, _, _ in pairs])
    y = np.array([lab for _, _, lab, _ in pairs], dtype=np.int64)
    ht = np.array([h for _, _, _, h in pairs], dtype=np.int64)
    return X, y, ht


def gen_tokens(n, V=8, max_p=6, max_h=3, seed=0, ood=False):
    """Return (tokens, seg, pad, y, htype) numpy arrays for the transformer."""
    from tmodel import build_tokens
    vocab = V + 1  # ids 0..V ; CLS=vocab, SEP=vocab+1
    pairs = _gen_pairs(n, V, max_p, max_h, seed, ood)
    toks, segs, pads, ys, hts = [], [], [], [], []
    for P, H, lab, ht in pairs:
        t, s, p = build_tokens(P, H, max_p, max_h, vocab)
        toks.append(t); segs.append(s); pads.append(p); ys.append(lab); hts.append(ht)
    return (np.stack(toks), np.stack(segs), np.stack(pads),
            np.array(ys, dtype=np.int64), np.array(hts, dtype=np.int64))


if __name__ == "__main__":
    V, max_p, max_h = 8, 6, 3
    Xtr, ytr = gen_train(2000, V, max_p, max_h, seed=0)
    Xood, yood, ht = gen_ood(2000, V, max_p, max_h, seed=999)
    print("train X", Xtr.shape, "pos frac", ytr.mean())
    print("ood X", Xood.shape, "label frac", yood.mean(), "hardneg frac", (ht == 0).mean())
    # sanity: on TRAIN, does set-containment perfectly predict label?
    # decode is annoying; instead verify generation invariants already asserted above.
    print("feature_dim", feature_dim(V, max_p, max_h))
    # Heuristic baseline on OOD: heuristic predicts ENTAIL iff contained.
    # For hardneg (contained, true=0) heuristic says 1 -> wrong. For pos (contained,true=1) -> right.
    # So heuristic OOD accuracy ~ fraction of pos = 1 - hardneg_frac.
    print("expected heuristic OOD acc ~", 1 - (ht == 0).mean())
