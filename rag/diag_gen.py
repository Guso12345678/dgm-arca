import json, sys
from collections import Counter

data = json.load(open("reports/phase3_eval.json"))

def find_rows(node):
    if isinstance(node, dict):
        if isinstance(node.get("rows"), list) and node["rows"] and "abstained" in node["rows"][0]:
            return node["rows"]
        for v in node.values():
            r = find_rows(v)
            if r:
                return r
    return None

rows = find_rows(data)
answerable = rows[:-8]
cross = Counter((r["abstained"], r["retrieved_relevant"]) for r in answerable)
print("abstained / retrieved_relevant -> count (answerable questions)")
for key in sorted(cross):
    print(f"  abstained={key[0]!s:5} retrieved_relevant={key[1]!s:5}: {cross[key]}")
print()
for title, pick in [
    ("ABSTAINED although the right chunk was retrieved", lambda r: r["abstained"] and r["retrieved_relevant"]),
    ("ANSWERED without citing the relevant chunk", lambda r: not r["abstained"] and not r["cites_relevant"]),
    ("ANSWERED and cited the relevant chunk", lambda r: not r["abstained"] and r["cites_relevant"]),
]:
    print("=" * 20, title)
    for r in [x for x in answerable if pick(x)][:3]:
        print("Q:", r["question"])
        print("A:", r["answer"][:500].replace("\n", " "))
        print()
