"""
Synthetic claim bank for a TRUE/FALSE judgment task.

Each claim is a short factual/commonsense statement with UNAMBIGUOUS truth value,
generated from templates so we can make many of them and control truth exactly.
The truth value is a property of the CLAIM CONTENT (arithmetic, ordering, category
membership, well-known facts) -- NOT of any surface token.

We deliberately avoid putting the words "true"/"false" anywhere in the claim text.
"""
import numpy as np

# ---- building blocks with hard ground truth ----
ANIMALS_MAMMAL = ["dog", "cat", "horse", "cow", "whale", "dolphin", "elephant", "tiger", "bear", "rabbit"]
ANIMALS_NONMAMMAL = ["salmon", "eagle", "frog", "snake", "lizard", "shark", "tuna", "sparrow", "turtle", "crocodile"]
THINGS_HOT = ["the sun", "boiling water", "lava", "a campfire", "molten steel"]
THINGS_COLD = ["an ice cube", "a snowball", "liquid nitrogen", "a frozen lake", "the inside of a freezer"]
CITY_COUNTRY = {"Paris": "France", "Tokyo": "Japan", "Cairo": "Egypt", "Madrid": "Spain",
                "Berlin": "Germany", "Ottawa": "Canada", "Lima": "Peru", "Oslo": "Norway",
                "Nairobi": "Kenya", "Hanoi": "Vietnam"}
COUNTRIES = list(CITY_COUNTRY.values())
COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
FRUITS = ["apple", "banana", "orange", "grape", "lemon", "mango", "pear", "cherry"]
VEGETABLES = ["carrot", "potato", "onion", "spinach", "broccoli", "cabbage"]


def _arith(rng):
    a = int(rng.integers(2, 20)); b = int(rng.integers(2, 20))
    s = a + b
    if rng.random() < 0.5:
        return f"{a} plus {b} equals {s}", True
    wrong = s + int(rng.choice([-3, -2, -1, 1, 2, 3]))
    return f"{a} plus {b} equals {wrong}", False


def _order(rng):
    a = int(rng.integers(1, 100)); b = int(rng.integers(1, 100))
    while b == a:
        b = int(rng.integers(1, 100))
    bigger, smaller = (a, b) if a > b else (b, a)
    if rng.random() < 0.5:
        return f"{bigger} is greater than {smaller}", True
    return f"{smaller} is greater than {bigger}", False


def _mammal(rng):
    if rng.random() < 0.5:
        x = rng.choice(ANIMALS_MAMMAL)
        if rng.random() < 0.5:
            return f"A {x} is a mammal", True
        return f"A {x} is not a mammal", False
    x = rng.choice(ANIMALS_NONMAMMAL)
    if rng.random() < 0.5:
        return f"A {x} is a mammal", False
    return f"A {x} is not a mammal", True


def _temp(rng):
    if rng.random() < 0.5:
        x = rng.choice(THINGS_HOT)
        if rng.random() < 0.5:
            return f"{x[0].upper()+x[1:]} is hot", True
        return f"{x[0].upper()+x[1:]} is cold", False
    x = rng.choice(THINGS_COLD)
    if rng.random() < 0.5:
        return f"{x[0].upper()+x[1:]} is cold", True
    return f"{x[0].upper()+x[1:]} is hot", False


def _capital(rng):
    city = rng.choice(list(CITY_COUNTRY.keys()))
    true_country = CITY_COUNTRY[city]
    if rng.random() < 0.5:
        return f"{city} is a city in {true_country}", True
    wrong = rng.choice([c for c in COUNTRIES if c != true_country])
    return f"{city} is a city in {wrong}", False


def _category(rng):
    if rng.random() < 0.5:
        x = rng.choice(FRUITS)
        if rng.random() < 0.5:
            return f"A {x} is a fruit", True
        return f"A {x} is a vegetable", False
    x = rng.choice(VEGETABLES)
    if rng.random() < 0.5:
        return f"A {x} is a vegetable", True
    return f"A {x} is a fruit", False


# Full diverse bank (too hard for 0.6B from ~150 ex; kept for reference / harder variant)
GENERATORS_DIVERSE = [_arith, _order, _mammal, _temp, _capital, _category]

# DEFAULT bank: numeric judgment (arithmetic sum-check + magnitude ordering).
# Truth is COMPUTABLE from the claim content -> a genuine non-copy judgment task, and
# a structured rule a 0.6B LoRA can learn reliably from ~150 examples (verified in diag).
GENERATORS = [_arith, _order]


def make_claim(rng):
    """Return (claim_text, is_true: bool). Truth is unambiguous."""
    g = GENERATORS[int(rng.integers(0, len(GENERATORS)))]
    return g(rng)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for _ in range(20):
        c, t = make_claim(rng)
        print(f"[{'T' if t else 'F'}] {c}")
