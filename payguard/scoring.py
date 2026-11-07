"""Combine all findings into one risk score per payment, and group them into cases to review."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .anomaly import outliers
from .data import Dataset
from .rules import LABELS, Settings, run_rules

LEVELS = [(80, "Critical"), (60, "High"), (35, "Medium"), (0, "Low")]
REVIEW_FROM = 35  # payments scoring this or more go to review


def level(score: float) -> str:
    return next(name for floor, name in LEVELS if score >= floor)


@dataclass
class Result:
    dataset: Dataset
    findings: pd.DataFrame   # one row per (payment, rule)
    scored: pd.DataFrame     # every payment with its risk score
    settings: Settings

    @property
    def cases(self) -> pd.DataFrame:
        """Payments that need a look, riskiest first."""
        return self.scored[self.scored.risk >= REVIEW_FROM].sort_values(["risk", "amount"], ascending=False)

    @property
    def amount_at_risk(self) -> float:
        return float(self.cases.amount.sum())

    def findings_for(self, payment_id: str) -> pd.DataFrame:
        return self.findings[self.findings.payment_id == payment_id].sort_values("weight", ascending=False)

    def by_rule(self) -> pd.DataFrame:
        f = self.findings
        out = f.groupby("rule").agg(payments=("payment_id", "nunique")).reset_index()
        amounts = self.scored.set_index("payment_id").amount
        out["amount"] = [amounts[f.payment_id[f.rule == r].unique()].sum() for r in out.rule]
        out["label"] = out.rule.map(LABELS)
        return out.sort_values("payments", ascending=False)

    def by_vendor(self) -> pd.DataFrame:
        c = self.cases
        v = self.dataset.vendors.set_index("vendor_id").name
        out = c.groupby("vendor_id").agg(cases=("payment_id", "count"), amount=("amount", "sum"), top_risk=("risk", "max")).reset_index()
        out.insert(1, "vendor", out.vendor_id.map(v))
        return out.sort_values(["top_risk", "amount"], ascending=False)

    def by_employee(self) -> pd.DataFrame:
        c = self.cases
        names = dict(zip(self.dataset.employees.employee_id, self.dataset.employees.name))
        rows = []
        for role in ("created_by", "approved_by"):
            g = c.groupby(role).agg(cases=("payment_id", "count"), amount=("amount", "sum")).reset_index().rename(columns={role: "employee_id"})
            g["role"] = "entered" if role == "created_by" else "approved"
            rows.append(g)
        out = pd.concat(rows)
        out.insert(1, "name", out.employee_id.map(names))
        return out.sort_values("cases", ascending=False)


def analyse(ds: Dataset, settings: Settings | None = None, outlier_share: float = 0.01) -> Result:
    settings = settings or Settings()
    parts = [run_rules(ds, settings)]
    if outlier_share > 0:
        parts.append(outliers(ds, share=outlier_share))
    findings = pd.concat([x for x in parts if len(x)], ignore_index=True) if any(len(x) for x in parts) else parts[0]
    # One finding per payment per rule (keep the strongest)
    findings = findings.sort_values("weight", ascending=False).drop_duplicates(["payment_id", "rule"]).reset_index(drop=True)

    # Risk = chance that at least one signal is real, treating them as independent: 1 - Π(1 - weight)
    keep = findings.groupby("payment_id").weight.apply(lambda w: 1 - np.prod(1 - w.to_numpy()))
    reasons = findings.groupby("payment_id").rule.apply(lambda r: ", ".join(LABELS.get(x, x) for x in r))
    scored = ds.payments.copy()
    scored["risk"] = (scored.payment_id.map(keep).fillna(0) * 100).round(1)
    scored["level"] = scored.risk.map(level)
    scored["reasons"] = scored.payment_id.map(reasons).fillna("")
    vendor_names = ds.vendors.set_index("vendor_id").name
    scored.insert(3, "vendor", scored.vendor_id.map(vendor_names))
    return Result(ds, findings, scored, settings)
