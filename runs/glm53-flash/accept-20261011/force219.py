"""#219 request 1's forced answer: the canonical ids of session.json message 2's answer (offline render of
messages 0-3, the ids after t2 request 1's prompt up to <|observation|>), with the canonical id at answer
offset 6,669 ('XL', 36345) replaced by another split of the same bytes ('X' + 'L'), then <|observation|>.
Writes force219.json with the ids and the exact replacement (run note)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bodies  # noqa: E402
import render  # noqa: E402

p1 = render.ids(bodies.t2(1))
p2 = render.ids(bodies.t2(2))
assert p2[:len(p1)] == p1, "request 1's prompt is not a prefix of request 2's"
obs = render.tid("<|observation|>")
end = p2.index(obs, len(p1))
ans = p2[len(p1):end]
assert len(ans) == 7545, len(ans)
OFF = 6669
assert ans[OFF] == 36345, (ans[OFF], render.TOK.id_to_token(ans[OFF]))
x, l_ = render.tid("X"), render.tid("L")
assert x is not None and l_ is not None
split = [x, l_]
assert render.TOK.decode(split) == render.TOK.decode([ans[OFF]]) == "XL"
forced = ans[:OFF] + split + ans[OFF + 1:] + [obs]
ctx = [render.TOK.id_to_token(i) for i in ans[OFF - 2:OFF + 3]]
note = {"prompt1_len": len(p1), "prompt2_len": len(p2), "answer_canonical_len": len(ans),
        "offset": OFF, "canonical_id": ans[OFF], "canonical_tok": "XL", "replacement_ids": split,
        "replacement_toks": ["X", "L"], "context_toks": ctx, "observation_id": obs,
        "forced_len": len(forced), "held_expected": len(p1) + len(forced) - 1,
        "forced": forced}
json.dump(note, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "force219.json"), "w"))
print(json.dumps({k: v for k, v in note.items() if k != "forced"}))
