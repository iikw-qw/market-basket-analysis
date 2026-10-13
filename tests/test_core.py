import pytest

from src import core, data


def test_apriori_supports_on_toy_data():
    baskets = [["a", "b"], ["a", "b", "c"], ["a", "c"], ["b"]]
    freq = core.apriori(baskets, min_support=0.25, max_len=3)
    assert freq[frozenset(["a"])] == pytest.approx(0.75)
    assert freq[frozenset(["a", "b"])] == pytest.approx(0.5)
    assert frozenset(["a", "b", "c"]) in freq


def test_min_support_filters():
    baskets = [["a", "b"], ["a"], ["a"], ["c"]]
    freq = core.apriori(baskets, min_support=0.5)
    assert frozenset(["c"]) not in freq


def test_rule_metrics_on_toy_data():
    baskets = [["a", "b"], ["a", "b"], ["a"], ["c"]]
    freq = core.apriori(baskets, min_support=0.25)
    rules = core.association_rules(freq, min_confidence=0.1, min_lift=0.0)
    r = rules[(rules["antecedent"] == "b") & (rules["consequent"] == "a")].iloc[0]
    assert r["confidence"] == pytest.approx(1.0)
    assert r["lift"] == pytest.approx(1.0 / 0.75)


def test_planted_bundles_are_found():
    baskets = data.generate_baskets(n=6000, seed=1)
    rules = core.association_rules(core.apriori(baskets, 0.02), 0.3, 1.5)
    pairs = {(r.antecedent, r.consequent) for r in rules.itertuples()}
    assert ("diapers", "wipes") in pairs or ("wipes", "diapers") in pairs
    assert ("bread", "butter") in pairs or ("butter", "bread") in pairs


def test_recommendation_excludes_items_in_basket():
    baskets = data.generate_baskets(n=6000, seed=1)
    rules = core.association_rules(core.apriori(baskets, 0.02), 0.3, 1.5)
    rec = core.recommend(rules, ["pasta", "tomato_sauce"])
    assert "pasta" not in rec and "tomato_sauce" not in rec
    assert "cheese" in rec
