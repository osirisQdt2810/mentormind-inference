"""Blind semantic judge helpers.

render <scores.json> <label> <which: final|pass_a|pass_b_run> > prompt.txt   the judge prompt for one run (no model name)
metrics <scores.json> <label> <which> <judge1.json> [judge2.json]  semantic recall/precision from the verdicts
"""

import json
import os
import sys
from pathlib import Path

S = Path(__file__).parent
TRUTH = Path(os.environ.get("MENTORMIND", ".")) / "data/video/groundtruth.json"
truth = json.loads(TRUTH.read_text())["steps"]


def truth_text() -> str:
    return "\n".join(
        f"{i}. [{t['t_start']:.1f}–{t['t_end']:.1f}] {t.get('actor')}, {'essential' if t.get('essential', True) else 'side'}: "
        f"{t['step']}" + (f" | {t['key_point']}" if t.get("key_point") else "")
        for i, t in enumerate(truth)
    )


def preds(scores: dict, label: str, which: str) -> list[dict]:
    """The steps of one run: ``final``, ``pass_a`` or ``pass_b_run`` (pass B's own pass A)."""
    return scores[label][f"{which}_steps"]


def render(scores_path: str, label: str, which: str) -> str:
    scores = json.loads(Path(scores_path).read_text())
    ps = preds(scores, label, which)
    pred = "\n".join(
        f"{i}. [{p['t'][0]:.1f}–{p['t'][1]:.1f}] {p.get('actor')}: {p['step']}"
        + (f" | {p['key_point']}" if p.get("key_point") else "")
        for i, p in enumerate(ps)
    )
    return (
        (S / "judge_prompt.md").read_text().replace("{truth}", truth_text()).replace("{pred}", pred)
    )


def check_verdict(v: dict, n_pred: int) -> None:
    """A verdict must judge each prediction exactly once and match only real truth indices."""
    seen = [p.get("i") for p in v["preds"]]
    if sorted(seen) != list(range(n_pred)):
        raise ValueError(f"verdict judges predictions {seen}, expected 0..{n_pred - 1} once each")
    bad = [
        p["match"]
        for p in v["preds"]
        if p.get("match") is not None and p["match"] not in range(len(truth))
    ]
    if bad:
        raise ValueError(f"verdict matches truth indices that do not exist: {bad}")


def _mean(values: list[float | None]) -> float | None:
    """Mean over the judges that could judge it; None when none could (not measurable, not 0)."""
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


def metrics(scores_path: str, label: str, which: str, verdicts: list[str]) -> dict:
    scores = json.loads(Path(scores_path).read_text())
    n_pred = len(preds(scores, label, which))
    essential = {i for i, t in enumerate(truth) if t.get("essential", True)}
    out = []
    for path in verdicts:
        v = json.loads(Path(path).read_text())
        check_verdict(v, n_pred)
        matched = [p for p in v["preds"] if p.get("match") is not None]
        found = {p["match"] for p in matched}
        actor = [p["actor_ok"] for p in matched if p.get("actor_ok") is not None]
        kp = [p["key_point_ok"] for p in matched if p.get("key_point_ok") is not None]
        out.append(
            {
                "sem_recall": len(found) / len(truth),
                "sem_recall_essential": len(found & essential) / len(essential),
                "sem_precision": len(matched) / n_pred if n_pred else None,
                "wrong_rate": sum(1 for p in v["preds"] if p.get("wrong")) / n_pred
                if n_pred
                else None,
                "actor_acc": sum(actor) / len(actor) if actor else None,
                "key_point_acc": sum(kp) / len(kp) if kp else None,
            }
        )
    recalls = [o["sem_recall"] for o in out]
    return {k: _mean([o[k] for o in out]) for k in out[0]} | {
        "judges": len(out),
        "spread_recall": round(max(recalls) - min(recalls), 3),
    }


if __name__ == "__main__":
    if sys.argv[1] == "render":
        print(render(*sys.argv[2:5]))
    else:
        result = metrics(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
        print(
            json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()})
        )
