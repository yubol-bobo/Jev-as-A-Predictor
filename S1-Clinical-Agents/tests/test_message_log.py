"""MessageLog: every logged state can be rebuilt exactly from the append-only log."""

import json

from src.s1clin.pb_decider import MessageLog


def replay(path):
    epochs, cur = [], []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["op"] == "reset":
            epochs.append(cur)
            cur = cur[:r["keep"]]
        else:
            cur = cur + [r["msg"]]
    return epochs + [cur]


def test_append_and_reset(tmp_path):
    log = MessageLog(tmp_path / "d.messages.jsonl")
    s1 = [{"role": "system", "content": "s"}, {"role": "user", "content": "stage 1"}]
    s2 = s1 + [{"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "ask_nurse"}}]},
               {"role": "tool", "content": "vitals"}]
    s3 = [s1[0], {"role": "user", "content": "stage 2"}]  # new stage: history rebuilt
    states = []
    for m in (s1, s2, s3):
        n, e = log.sync(m)
        states.append((m, n, e))
    epochs = replay(tmp_path / "d.messages.jsonl")
    for m, n, e in states:
        assert epochs[e][:n] == json.loads(json.dumps(m))
    assert [e for _, _, e in states] == [0, 0, 1]
