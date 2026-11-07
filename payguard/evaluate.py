"""How well PayGuard does on data where the answers are known (truth.csv from the generator)."""
from __future__ import annotations

import pandas as pd

from .scoring import REVIEW_FROM, Result

SCHEME_LABELS = {
    "duplicate": "Invoice paid twice",
    "near_duplicate": "Invoice re-submitted with a changed number",
    "split_payment": "Purchase split to dodge an approval limit",
    "ghost_vendor": "Fake vendor set up by an employee",
    "bank_change": "Bank account switched before a big payment",
    "self_approval": "Entered and approved by the same person",
    "just_below_limit": "Repeatedly approved just under a limit",
}


def evaluate(result: Result, truth: pd.DataFrame, threshold: float = REVIEW_FROM) -> dict:
    flagged = set(result.scored.payment_id[result.scored.risk >= threshold])
    planted = set(truth.payment_id)
    hits = flagged & planted
    per = (truth.assign(caught=truth.payment_id.isin(flagged))
           .groupby("scheme").agg(planted=("payment_id", "count"), caught=("caught", "sum")).reset_index())
    per["recall"] = per.caught / per.planted
    per["scheme_label"] = per.scheme.map(SCHEME_LABELS).fillna(per.scheme)
    precision = len(hits) / len(flagged) if flagged else 0.0
    recall = len(hits) / len(planted) if planted else 0.0
    return {
        "threshold": threshold,
        "flagged": len(flagged),
        "planted": len(planted),
        "caught": len(hits),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "per_scheme": per[["scheme", "scheme_label", "planted", "caught", "recall"]],
        "missed": truth[~truth.payment_id.isin(flagged)],
        "false_alarms": result.scored[result.scored.payment_id.isin(flagged - planted)],
    }
