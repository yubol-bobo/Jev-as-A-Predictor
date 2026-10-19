"""State rendering and context fitting shared by every System One decider (scripts/clef_server.py, s1_proxy.py).

The state is the task instruction plus every step so far (action, then its tool results as the LLM saw them):
  - the instruction and every action are always kept in full;
  - tool results are kept in full, then shortened oldest-first to SHORT_CHARS, then replaced oldest-first by
    "[result omitted]", until the complete encoded request fits.
"""

SHORT_CHARS = 800


def render_state(instruction, steps, short, omit):
    parts = ["TASK GIVEN TO THE AGENT:", instruction.strip(), "", "AGENT ACTIONS AND RESULTS SO FAR:"]
    if not steps:
        parts.append("(none yet)")
    r = 0  # global result index, oldest first
    for i, st in enumerate(steps, 1):
        parts.append(f"\n[Step {i}] ACTION:\n{st['action']}")
        for res in st["results"]:
            if r < omit:
                text = "[result omitted]"
            elif r < omit + short and len(res) > SHORT_CHARS:
                text = res[:SHORT_CHARS] + f" ...[shortened, {len(res)} chars total]"
            else:
                text = res
            parts.append(f"[Step {i}] RESULT:\n{text}")
            r += 1
    return "\n".join(parts)


def fit_request(instruction, steps, question, options, count, max_context):
    """The System One request for this state, fitted into max_context tokens as counted by count(request)."""
    n_res = sum(len(s["results"]) for s in steps)

    def build(short, omit):
        req = {"state": render_state(instruction, steps, short, omit),
               "questions": {"next": {"type": "choice", "instructions": question, "criteria": options}}}
        return req, count(req)

    req, n = build(0, 0)
    if n <= max_context:
        return req, n, {"shortened": 0, "omitted": 0, "results": n_res}
    # binary search the smallest number of oldest results to shorten, then to omit
    lo, hi = 1, n_res
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        r, m = build(mid, 0)
        if m <= max_context:
            best, hi = (r, m, mid, 0), mid - 1
        else:
            lo = mid + 1
    if best is None:
        lo, hi = 1, n_res
        while lo <= hi:
            mid = (lo + hi) // 2
            r, m = build(n_res - mid, mid)
            if m <= max_context:
                best, hi = (r, m, n_res - mid, mid), mid - 1
            else:
                lo = mid + 1
    if best is None:
        raise ValueError("request does not fit even with every result omitted")
    r, m, short, omit = best
    return r, m, {"shortened": short, "omitted": omit, "results": n_res}
