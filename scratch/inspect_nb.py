import json

nb = json.load(open("final001 (1).ipynb", encoding="utf-8"))
print("Total cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))[:100].replace("\n", " ")
    outs = len(c.get("outputs", []))
    print(f"Cell {i:02d} [{c['cell_type']}] (outputs: {outs:2d}): {src}")
