"""
Compositional NLI task with a LEXICAL-OVERLAP spurious cue (HANS-style).

label = ENTAIL (0) / NON-ENTAIL (1), where truth is a GENUINE logical consequence
of a multi-sentence premise. Truth is unambiguous because we generate it from
controlled relations over a small entity vocabulary:

  RELATION TYPES (each premise is 2 sentences = compositional / multi-sentence):
    * subset/superset  : "All A are B. C is an A."  =>  "C is a B."   (ENTAIL)
                          but   "All A are B. C is a B."  ->  "C is an A."  (NON-ENTAIL, illicit converse)
    * ordering         : "X is taller than Y. Y is taller than Z." => "X is taller than Z." (ENTAIL)
                          but reversed "Z is taller than X." (NON-ENTAIL)
    * negation         : "No A are B. C is an A." => "C is not a B." (ENTAIL)
                          but "C is a B." (NON-ENTAIL)

The SPURIOUS HEURISTIC is LEXICAL OVERLAP: does EVERY content word of the
hypothesis appear in the premise? This is a STATISTIC computed from the strings,
NOT a token printed in the prompt. HANS-style: hard deployment cases have FULL
overlap (overlap-heuristic says ENTAIL) but are actually NON-ENTAIL.

In TRAINING the overlap heuristic agrees with the truth with prob p_spur:
  we mix items so that P(overlap==1  <=>  label==ENTAIL) = p_spur.
At DEPLOYMENT we use HANS-hard cases: high overlap, NON-entailment -> a
heuristic-relier FAILS, a real reasoner SUCCEEDS.

NON-DEGENERACY eval: a BALANCED-OVERLAP set where overlap is uninformative
(overlap==1 is 50/50 entail vs non-entail, and we also include overlap==0 cases),
so accuracy there reflects genuine reasoning, not the cue.

All truth labels are derived programmatically from the relation, never sampled.
"""
import numpy as np

# Small entity vocabulary (kinds, names). Multi-token-ish, simple words.
KINDS  = ["dog", "cat", "bird", "fish", "horse", "rabbit", "mouse", "frog"]
GROUPS = ["animal", "pet", "creature", "mammal", "runner", "swimmer"]
NAMES  = ["Rex", "Milo", "Bella", "Coco", "Duke", "Luna", "Max", "Nala"]
ADJS   = ["tall", "fast", "old", "strong", "heavy", "loud"]
# comparative form
COMP   = {"tall": "taller", "fast": "faster", "old": "older",
          "strong": "stronger", "heavy": "heavier", "loud": "louder"}

ENTAIL, NON = 0, 1   # label ids


def _art(word):
    return "an" if word[0] in "aeiou" else "a"


def _lemma(t):
    """Crude singular lemma so 'pets'<->'pet', 'mouses'->'mouse' match.
    HANS overlap is a CONTENT-word (lemma) containment heuristic."""
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t


def _content_words(s):
    """Lowercased alphabetic tokens, dropping function/scaffold words so that
    'overlap' means CONTENT overlap (the HANS lexical-overlap heuristic).
    Comparatives are lemmatized too so 'taller' premise covers 'taller' hyp
    (they are identical surface forms anyway)."""
    stop = {"all", "are", "is", "a", "an", "no", "not", "than", "the",
            "every", "some", "and", "of"}
    toks = [t.strip(".,").lower() for t in s.split()]
    return [_lemma(t) for t in toks if t and t not in stop]


def lexical_overlap(premise, hypothesis):
    """Spurious heuristic value: 1 if EVERY content word of the hypothesis
    appears in the premise, else 0. (A statistic, not a printed token.)"""
    pw = set(_content_words(premise))
    hw = _content_words(hypothesis)
    return 1 if all(w in pw for w in hw) else 0


# --------- relation generators: each returns (premise, hypothesis, label, want_overlap_high) ----------

def _subset_item(rng, target_label, want_overlap):
    """ 'All <kind> are <group>. <Name> is a <kind/group>.' """
    kind = rng.choice(KINDS); group = rng.choice(GROUPS); name = rng.choice(NAMES)
    if target_label == ENTAIL:
        # valid: All K are G ; Name is a K => Name is a G
        premise = f"All {kind}s are {group}s. {name} is {_art(kind)} {kind}."
        hyp = f"{name} is {_art(group)} {group}."
        # hyp content lemmas {name, group} subset of premise {kind,group,name} -> overlap 1
    else:
        if want_overlap:
            # illicit converse: All K are G ; Name is a G  -> "Name is a K" (NON-entail), full overlap
            premise = f"All {kind}s are {group}s. {name} is {_art(group)} {group}."
            hyp = f"{name} is {_art(kind)} {kind}."
        else:
            # non-entail with a fresh group not in premise (low overlap)
            other = rng.choice([g for g in GROUPS if g != group])
            premise = f"All {kind}s are {group}s. {name} is {_art(kind)} {kind}."
            hyp = f"{name} is {_art(other)} {other}."
    return premise, hyp


def _order_item(rng, target_label, want_overlap):
    """ '<A> is <comp> than <B>. <B> is <comp> than <C>.' transitivity """
    a, b, c = rng.choice(NAMES, size=3, replace=False)
    adj = rng.choice(ADJS); comp = COMP[adj]
    if target_label == ENTAIL:
        premise = f"{a} is {comp} than {b}. {b} is {comp} than {c}."
        hyp = f"{a} is {comp} than {c}."   # transitive, full overlap
    else:
        if want_overlap:
            # reversed conclusion: actually C is not >A, full overlap (same names+comp)
            premise = f"{a} is {comp} than {b}. {b} is {comp} than {c}."
            hyp = f"{c} is {comp} than {a}."
        else:
            # introduce a name not in premise -> low overlap, non-entail
            others = [n for n in NAMES if n not in (a, b, c)]
            d = rng.choice(others)
            premise = f"{a} is {comp} than {b}. {b} is {comp} than {c}."
            hyp = f"{a} is {comp} than {d}."
    return premise, hyp


def _neg_item(rng, target_label, want_overlap):
    """ 'No <kind> are <group>. <Name> is a <kind>.' negation """
    kind = rng.choice(KINDS); group = rng.choice(GROUPS); name = rng.choice(NAMES)
    if target_label == ENTAIL:
        # No K are G ; Name is a K => Name is not a G   (ENTAIL)
        premise = f"No {kind}s are {group}s. {name} is {_art(kind)} {kind}."
        hyp = f"{name} is not {_art(group)} {group}."   # lemmas {name,group} in premise -> full
    else:
        if want_overlap:
            # contradicts: "Name is a G" (NON-entail; the premise entails NOT G). full overlap
            premise = f"No {kind}s are {group}s. {name} is {_art(kind)} {kind}."
            hyp = f"{name} is {_art(group)} {group}."
        else:
            other = rng.choice([g for g in GROUPS if g != group])
            premise = f"No {kind}s are {group}s. {name} is {_art(kind)} {kind}."
            hyp = f"{name} is {_art(other)} {other}."
    return premise, hyp


_GENS = [_subset_item, _order_item, _neg_item]


def _make_one(rng, target_label, want_overlap):
    gen = _GENS[rng.integers(0, len(_GENS))]
    premise, hyp = gen(rng, target_label, want_overlap)
    return premise, hyp


def format_prompt(premise, hyp):
    return (f"Premise: {premise} Hypothesis: {hyp} "
            f"Does the premise entail the hypothesis? (0=yes-entails, 1=no):")


def make(n, seed, mode, p_spur=1.0):
    """
    mode='train' : overlap heuristic agrees with truth with prob p_spur.
                   We build a 50/50 label set; for each item we choose whether the
                   overlap cue is CONSISTENT with truth (prob p_spur) or MISLEADING.
                     - ENTAIL items here always have full overlap (overlap=1 -> entail, consistent).
                       To make a fraction (1-p_spur) of ENTAIL items have the cue MISLEAD,
                       we instead realize them as low-overlap entailments? Entailment with low
                       overlap is hard to construct cleanly; instead we encode inconsistency on
                       the NON-ENTAIL side, which is the HANS structure: cue says entail (overlap=1)
                       but truth is non-entail.
                   Concretely:
                     label=ENTAIL  -> overlap=1 (cue correct).            [consistent]
                     label=NON     -> with prob p_spur: low overlap (cue says non, correct)
                                       with prob (1-p_spur): HIGH overlap (cue says entail, WRONG)
                   So among NON items a (1-p_spur) fraction are HANS-hard. Overall the cue
                   matches the label with prob ~ p_spur (since ENTAIL always matches).
    mode='deploy': HANS-hard ONLY. All items HIGH overlap; truth is NON-ENTAIL.
                   overlap-relier predicts ENTAIL (wrong) -> alignment(=truth-acc) low.
                   We also return some HIGH-overlap ENTAIL items? No: pure HANS-hard is
                   all NON-entail high-overlap, the canonical stress test.
    mode='balanced': NON-DEGENERACY. ALL items are HIGH overlap (overlap==1) and
                   labels are 50/50 entail/non. Within this set the overlap cue is
                   completely uninformative (P(entail|overlap=1)=0.5), so a pure
                   overlap-relier scores ~0.5 while a genuine reasoner scores high.
                   This is the strongest non-degeneracy probe: among the maximally
                   tempting full-overlap items, can the model separate entail from
                   non-entail by actually reading the relation? Truth-derived.
    Returns rows: (prompt, label, overlap, premise, hyp)
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        if mode == "train":
            lab = ENTAIL if rng.random() < 0.5 else NON
            if lab == ENTAIL:
                premise, hyp = _make_one(rng, ENTAIL, want_overlap=True)
            else:
                misleading = rng.random() >= p_spur   # fraction (1-p_spur) are HANS-hard
                premise, hyp = _make_one(rng, NON, want_overlap=misleading)
        elif mode == "deploy":
            # HANS-hard: non-entail, high overlap
            premise, hyp = _make_one(rng, NON, want_overlap=True)
            lab = NON
        elif mode == "balanced":
            # ALL high-overlap, labels 50/50 -> overlap cue uninformative here.
            lab = ENTAIL if rng.random() < 0.5 else NON
            premise, hyp = _make_one(rng, lab, want_overlap=True)
        else:
            raise ValueError(mode)
        ov = lexical_overlap(premise, hyp)
        rows.append((format_prompt(premise, hyp), lab, ov, premise, hyp))
    return rows


def _verify_generators():
    """Assert each generator yields the intended overlap and that ENTAIL items
    are always full-overlap (the structural premise of the HANS cue)."""
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(4000):
        # ENTAIL must be overlap=1
        for gen in _GENS:
            p, h = gen(rng, ENTAIL, want_overlap=True)
            if lexical_overlap(p, h) != 1:
                bad += 1; print("ENTAIL-not-overlap1:", gen.__name__, p, "||", h)
        # NON want_overlap=True must be overlap=1 (HANS-hard)
        for gen in _GENS:
            p, h = gen(rng, NON, want_overlap=True)
            if lexical_overlap(p, h) != 1:
                bad += 1; print("NON-hard-not-overlap1:", gen.__name__, p, "||", h)
        # NON want_overlap=False must be overlap=0 (cue says non, correct)
        for gen in _GENS:
            p, h = gen(rng, NON, want_overlap=False)
            if lexical_overlap(p, h) != 0:
                bad += 1; print("NON-easy-not-overlap0:", gen.__name__, p, "||", h)
    print(f"\n[verify] overlap-structure violations: {bad} (want 0)")
    return bad


if __name__ == "__main__":
    _verify_generators()
    # sanity: print examples + verify overlap statistics per mode
    for mode in ["train", "deploy", "balanced"]:
        r = make(2000, 7, mode, p_spur=0.9)
        labs = np.array([x[1] for x in r]); ovs = np.array([x[2] for x in r])
        # cue-as-predictor: overlap=1 => predict ENTAIL(0), overlap=0 => predict NON(1)
        cue_pred = np.where(ovs == 1, ENTAIL, NON)
        cue_acc = (cue_pred == labs).mean()
        print(f"\n== mode={mode} ==  n={len(r)} entail_frac={ (labs==ENTAIL).mean():.2f} "
              f"overlap1_frac={ovs.mean():.2f} cue_acc(overlap->label)={cue_acc:.3f}")
        # among overlap==1, what frac is entail?
        m = ovs == 1
        if m.sum():
            print(f"   among overlap==1: entail_frac={ (labs[m]==ENTAIL).mean():.3f} (n={m.sum()})")
        m0 = ovs == 0
        if m0.sum():
            print(f"   among overlap==0: entail_frac={ (labs[m0]==ENTAIL).mean():.3f} (n={m0.sum()})")
        for x in r[:3]:
            print("   ex:", repr(x[0]), "| label", x[1], "| overlap", x[2])
