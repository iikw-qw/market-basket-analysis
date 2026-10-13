import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import core, data

OUT = Path("reports")


def main():
    OUT.mkdir(exist_ok=True)
    baskets = data.generate_baskets()
    frequent = core.apriori(baskets, min_support=0.02, max_len=3)
    rules = core.association_rules(frequent, min_confidence=0.3, min_lift=1.5)

    top = rules.head(12)
    plt.figure(figsize=(8, 5))
    plt.scatter(rules["support"], rules["confidence"], s=rules["lift"] * 12, color="grey", alpha=0.6)
    plt.xlabel("support")
    plt.ylabel("confidence")
    plt.title("Association rules (marker size = lift)")
    plt.tight_layout()
    plt.savefig(OUT / "rules.png", dpi=130)

    rules.round(4).to_csv(OUT / "rules.csv", index=False)
    example = ["pasta", "tomato_sauce"]
    summary = {
        "baskets": len(baskets),
        "frequent_itemsets": len(frequent),
        "rules": int(len(rules)),
        "top_rules": [
            f"{r.antecedent} -> {r.consequent} (lift {r.lift:.1f}, confidence {r.confidence:.2f})"
            for r in top.head(5).itertuples()
        ],
        "recommendation_for_pasta_and_sauce": core.recommend(rules, example),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
