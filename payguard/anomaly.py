"""Statistical outliers: payments that look unlike the rest, found with an Isolation Forest.

The rules catch known patterns; this catches the unusual things nobody wrote a rule for. Each outlier gets a
plain-words reason (the features that make it stand out the most).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .data import Dataset
from .rules import FINDING_COLUMNS, money

def features(ds: Dataset) -> pd.DataFrame:
    p = ds.payments
    limits = dict(zip(ds.employees.employee_id, ds.employees.approval_limit))
    created = ds.vendors.set_index("vendor_id").created_date
    vendor_median = p.groupby("vendor_id").amount.transform("median")
    f = pd.DataFrame(index=p.index)
    f["log_amount"] = np.log10(p.amount.clip(lower=1))
    f["vs_vendor"] = np.log2((p.amount / vendor_median).clip(lower=1e-3))
    f["vendor_volume"] = np.log1p(p.groupby("vendor_id").amount.transform("count"))
    f["hour"] = p.payment_date.dt.hour + p.payment_date.dt.minute / 60
    lag = (p.payment_date.dt.normalize() - p.invoice_date).dt.days
    f["invoice_lag"] = lag.fillna(lag.median()).clip(-30, 365)
    age = (p.payment_date.dt.normalize() - p.vendor_id.map(created)).dt.days
    f["vendor_age"] = np.log1p(age.fillna(age.median()).clip(lower=0))
    lim = p.approved_by.map(limits).replace(0, np.nan)
    f["limit_use"] = (p.amount / lim).fillna(1.0).clip(0, 2)
    return f


def outliers(ds: Dataset, share: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """Flag roughly `share` of payments as outliers, with a reason for each."""
    if len(ds.payments) < 50:
        return pd.DataFrame(columns=FINDING_COLUMNS)
    f = features(ds)
    model = IsolationForest(n_estimators=300, contamination=share, random_state=seed).fit(f)
    score = -model.score_samples(f)  # higher = more unusual
    flagged = model.predict(f) == -1
    z = (f - f.median()) / (f.quantile(0.75) - f.quantile(0.25)).replace(0, 1)
    lo, hi = score[flagged].min() if flagged.any() else 0, score.max()
    p = ds.payments
    rows = []
    for i in np.where(flagged)[0]:
        top = z.iloc[i].abs().sort_values(ascending=False).index[:2]
        reasons = [_reason(c, f.iloc[i], z.iloc[i], p.iloc[i]) for c in top]
        strength = (score[i] - lo) / (hi - lo) if hi > lo else 1
        rows.append(dict(payment_id=p.payment_id.iloc[i], rule="outlier", weight=round(0.15 + 0.2 * strength, 3),
                         detail="Unusual compared with the other payments: " + "; ".join(reasons) + ".", related=[]))
    return pd.DataFrame(rows, columns=FINDING_COLUMNS)


def _reason(col: str, row: pd.Series, z: pd.Series, pay) -> str:
    up = z[col] > 0
    if col == "log_amount":
        return f"{money(pay.amount)} is unusually {'large' if up else 'small'}"
    if col == "vs_vendor":
        r = 2 ** row[col]
        return f"it is {r:.1f}× this vendor's usual amount" if up else f"it is only {r:.0%} of this vendor's usual amount"
    if col == "vendor_volume":
        return f"this vendor is paid unusually {'often' if up else 'rarely'}"
    if col == "hour":
        return f"it was entered at {pay.payment_date:%H:%M}"
    if col == "invoice_lag":
        return f"the invoice was {int(row[col])} days old when paid"
    if col == "vendor_age":
        return "the vendor is unusually " + ("long-standing" if up else "new")
    if col == "limit_use":
        return f"it uses {row[col]:.0%} of the approver's limit"
    return col
