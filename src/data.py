import numpy as np
import pandas as pd

ITEMS = [
    "bread", "butter", "milk", "eggs", "cheese", "apples", "bananas", "pasta", "tomato_sauce",
    "coffee", "sugar", "tea", "chips", "salsa", "beer", "diapers", "wipes", "rice", "chicken", "yogurt",
]

# groups of items that tend to be bought together: (items, probability of the bundle in a basket)
BUNDLES = [
    (("pasta", "tomato_sauce", "cheese"), 0.12),
    (("chips", "salsa", "beer"), 0.07),
    (("diapers", "wipes"), 0.06),
    (("bread", "butter"), 0.15),
    (("coffee", "sugar"), 0.10),
]


def generate_baskets(n=12000, seed=19):
    rng = np.random.default_rng(seed)
    popularity = rng.dirichlet(np.ones(len(ITEMS)) * 2.0)
    baskets = []
    for _ in range(n):
        basket = set()
        for items, p in BUNDLES:
            if rng.random() < p:
                basket.update(items)
        k = rng.integers(1, 6)
        basket.update(rng.choice(ITEMS, size=k, replace=False, p=popularity))
        baskets.append(sorted(basket))
    return baskets
