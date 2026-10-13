from collections import defaultdict
from itertools import combinations

import pandas as pd


def apriori(baskets, min_support=0.01, max_len=3):
    """Frequent itemsets by level-wise search; returns {frozenset: support}."""
    n = len(baskets)
    sets = [frozenset(b) for b in baskets]
    counts = defaultdict(int)
    for s in sets:
        for item in s:
            counts[frozenset([item])] += 1
    current = {k: v / n for k, v in counts.items() if v / n >= min_support}
    frequent = dict(current)

    size = 2
    while current and size <= max_len:
        prev = list(current)
        candidates = set()
        for a, b in combinations(prev, 2):
            union = a | b
            if len(union) == size:
                # every subset of a candidate must itself be frequent
                if all(frozenset(sub) in current for sub in combinations(union, size - 1)):
                    candidates.add(union)
        counts = defaultdict(int)
        for s in sets:
            for c in candidates:
                if c <= s:
                    counts[c] += 1
        current = {k: v / n for k, v in counts.items() if v / n >= min_support}
        frequent.update(current)
        size += 1
    return frequent


def association_rules(frequent, min_confidence=0.3, min_lift=1.0):
    rows = []
    for itemset, support in frequent.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for lhs in combinations(itemset, r):
                lhs = frozenset(lhs)
                rhs = itemset - lhs
                confidence = support / frequent[lhs]
                lift = confidence / frequent[rhs]
                if confidence >= min_confidence and lift >= min_lift:
                    rows.append(
                        {
                            "antecedent": ", ".join(sorted(lhs)),
                            "consequent": ", ".join(sorted(rhs)),
                            "support": support,
                            "confidence": confidence,
                            "lift": lift,
                            "leverage": support - frequent[lhs] * frequent[rhs],
                        }
                    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def recommend(rules, basket, top=3):
    """Suggest items that are not in the basket yet, ranked by the best rule that fires."""
    basket = set(basket)
    scores = {}
    for _, r in rules.iterrows():
        lhs = set(r["antecedent"].split(", "))
        rhs = set(r["consequent"].split(", "))
        if lhs <= basket:
            for item in rhs - basket:
                scores[item] = max(scores.get(item, 0), r["lift"] * r["confidence"])
    return sorted(scores, key=scores.get, reverse=True)[:top]
