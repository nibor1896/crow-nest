"""crow-nest #154: the GLM-5.3-Flash tensor table fixture, from the 62 cached shard headers.

    python -I converter/tests/fixtures/glm53_tensor_table.py <headers-dir> converter/tests/fixtures/glm53-flash-tensors.tsv

Headers: `models/GLM-5.3-Flash-original/headers/<shard>.json` (`{"shard", "size", "data_start",
"header"}`), zai-org/GLM-5.3-Flash rev eb9eb208eb0d988989d07a6a12d0fdeb5f52574a. Only names,
dtypes, shapes, shard numbers and byte counts are written, no weight. The routed experts
(74,304 rows with their scales) are collapsed: one row per (layer, projection, weight or
scale, shard) with the expert range `{a..b}` in the name; every other tensor has its own row.
`# shard` lines carry each shard's `data_start` and file size, so the test can check
sum(tensor bytes) + sum(data_start) == sum(shard sizes) == 328,337,455,672 B.
"""
import json
import re
import sys
from pathlib import Path

EXP = re.compile(r"^(model\.language_model\.layers\.\d+\.mlp\.experts\.)(\d+)(\..*)$")


def main(hdir, out):
    rows, shards, groups = [], [], {}
    for f in sorted(Path(hdir).glob("model-*.safetensors.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        nr = int(r["shard"].split("-")[1])
        shards.append("# shard\t%02d\t%d\t%d" % (nr, r["data_start"], r["size"]))
        for name, v in r["header"].items():
            if name == "__metadata__":
                continue
            b = v["data_offsets"][1] - v["data_offsets"][0]
            shape = "x".join(str(x) for x in v["shape"])
            m = EXP.match(name)
            if m:
                key = (m[1], m[3], v["dtype"], shape, nr, b)
                groups.setdefault(key, []).append(int(m[2]))
            else:
                rows.append((name, v["dtype"], shape, nr, b))
    for (pre, post, dt, shape, nr, b), es in groups.items():
        es.sort()
        start = prev = es[0]
        for e in es[1:] + [None]:
            if e is not None and e == prev + 1:
                prev = e
                continue
            rows.append(("%s{%d..%d}%s" % (pre, start, prev, post), dt, shape, nr, b))
            if e is not None:
                start = prev = e
    rows.sort()
    with open(out, "w", encoding="utf-8", newline="\n") as w:
        w.write("# GLM-5.3-Flash rev eb9eb208eb0d988989d07a6a12d0fdeb5f52574a, from the 62 shard headers (crow-nest #154)\n")
        w.write("# name ({a..b} = expert range)\tdtype\tshape\tshard\tbytes per tensor\n")
        for s in shards:
            w.write(s + "\n")
        for r in rows:
            w.write("%s\t%s\t%s\t%02d\t%d\n" % r)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
