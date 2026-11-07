"""Detection rules. Each rule looks for one known fraud or error pattern and explains every hit in plain words.

A rule returns findings: one row per flagged payment, with
  rule     – short name
  weight   – how strongly it suggests fraud or error (0–1), used for the risk score
  detail   – the explanation shown to the reviewer
  related  – other payment ids involved (e.g. the original of a duplicate)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .data import Dataset
from .holidays import is_holiday

FINDING_COLUMNS = ["payment_id", "rule", "weight", "detail", "related"]


@dataclass
class Settings:
    duplicate_window_days: int = 60          # an exact repeat of an invoice within this many days
    near_duplicate_window_days: int = 14     # same vendor + amount, similar invoice number
    invoice_similarity: int = 80             # 0–100, how alike two invoice numbers must be
    split_window_days: int = 3               # pieces of a split purchase fall within this many days
    split_size_factor: float = 2.5           # …and add up to this × the vendor's usual payment
    split_min_piece: float = 0.45            # …with pieces averaging at least this share of the limit
    just_below_share: float = 0.92           # "just under the limit" = at least this share of it
    just_below_min_count: int = 3            # …happening this many times for one approver + vendor
    bank_change_days: int = 30               # payment within this many days of a bank-account change
    bank_change_amount_factor: float = 2.0   # …and bigger than this × the vendor's usual payment
    new_vendor_days: int = 30
    new_vendor_amount_factor: float = 3.0
    round_amount_min: float = 5_000
    office_hours: tuple[int, int] = (7, 19)
    benford_min_payments: int = 40
    benford_mad_flag: float = 0.015          # Nigrini: MAD above 0.015 = nonconformity
    fuzzy_address: int = 90
    enabled: set[str] = field(default_factory=lambda: set(RULES))


def _findings(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=FINDING_COLUMNS)


def money(x: float) -> str:
    return f"EC${x:,.2f}"


def norm_invoice(s) -> str:
    return re.sub(r"[^0-9A-Z]", "", str(s).upper())


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


# ---------------------------------------------------------------- duplicates

def exact_duplicates(ds: Dataset, cfg: Settings) -> pd.DataFrame:
    """The same invoice from the same vendor paid more than once."""
    p = ds.payments.assign(_inv=ds.payments.invoice_no.map(norm_invoice))
    rows = []
    for _, g in p[p._inv != ""].groupby(["vendor_id", "_inv"]):
        if len(g) < 2:
            continue
        g = g.sort_values("payment_date")
        first = g.iloc[0]
        for _, r in g.iloc[1:].iterrows():
            gap = (r.payment_date - first.payment_date).days
            if gap > cfg.duplicate_window_days and r.amount != first.amount:
                continue
            same = r.amount == first.amount
            rows.append(dict(payment_id=r.payment_id, rule="exact_duplicate", weight=0.9 if same else 0.6,
                             detail=f"Invoice {r.invoice_no} was already paid on {first.payment_date:%d %b %Y} ({first.payment_id}, "
                                    f"{money(first.amount)}){'' if same else f'; this time {money(r.amount)}'}.",
                             related=[first.payment_id]))
    return _findings(rows)


def near_duplicates(ds: Dataset, cfg: Settings) -> pd.DataFrame:
    """Same vendor and amount within a short time, with an invoice number that is almost the same."""
    p = ds.payments.assign(_inv=ds.payments.invoice_no.map(norm_invoice))
    rows = []
    for _, g in p.groupby(["vendor_id", "amount"]):
        if len(g) < 2:
            continue
        g = g.sort_values("payment_date").to_dict("records")
        for i, b in enumerate(g):
            for a in g[:i]:
                gap = (b["payment_date"] - a["payment_date"]).days
                if gap > cfg.near_duplicate_window_days or a["_inv"] == b["_inv"]:
                    continue
                sim = fuzz.ratio(a["_inv"], b["_inv"])
                da, db = _digits(a["_inv"]), _digits(b["_inv"])
                digits_shuffled = len(da) > 2 and sorted(da) == sorted(db)
                if sim >= cfg.invoice_similarity or digits_shuffled or (da and da == db):
                    rows.append(dict(payment_id=b["payment_id"], rule="near_duplicate", weight=0.65,
                                     detail=f"Same vendor and amount ({money(b['amount'])}) as {a['payment_id']} {gap} day(s) earlier, "
                                            f"and invoice {b['invoice_no']} looks like {a['invoice_no']}.",
                                     related=[a["payment_id"]]))
                    break
    return _findings(rows)


# ---------------------------------------------------------------- approvals

def _limits(ds: Dataset) -> dict[str, float]:
    return dict(zip(ds.employees.employee_id, ds.employees.approval_limit))


def split_payments(ds: Dataset, cfg: Settings) -> pd.DataFrame:
    """Several payments to one vendor, close together, each under the approver's limit but over it in total.

    Busy vendors are legitimately paid several times a week, so a cluster only counts when it is unlikely by chance
    for that vendor and approver (Poisson tail probability below 1%)."""
    limits = _limits(ds)
    p = ds.payments
    start, end = ds.period
    days = max(1, (end - start).days + 1)
    usual = p.groupby("vendor_id").amount.median()
    rows, used = [], set()
    for (vendor, approver), g in p.groupby(["vendor_id", "approved_by"]):
        limit = limits.get(approver, 0)
        if len(g) < 2 or not limit:
            continue
        expected = len(g) / days * (cfg.split_window_days + 1)   # payments expected in one window
        g = g.sort_values("payment_date").to_dict("records")
        for i, a in enumerate(g):
            if a["payment_id"] in used:
                continue
            group = [a] + [b for b in g[i + 1:] if (b["payment_date"] - a["payment_date"]).days <= cfg.split_window_days
                           and b["payment_id"] not in used]
            total = sum(x["amount"] for x in group)
            if len(group) < 2 or total <= limit or any(x["amount"] > limit for x in group):
                continue
            # People splitting a purchase use as few pieces as they can, each fairly close to the limit
            if total / len(group) < cfg.split_min_piece * limit:
                continue
            chance = _poisson_at_least(len(group) - 1, expected)
            # A real purchase split in pieces adds up to far more than this vendor is usually paid at once
            big_total = total >= cfg.split_size_factor * usual[vendor]
            if not ((chance <= 0.01 and total >= 1.5 * usual[vendor]) or (big_total and chance <= 0.2)):
                continue
            ids = [x["payment_id"] for x in group]
            used.update(ids)
            for x in group:
                rows.append(dict(payment_id=x["payment_id"], rule="split_payment", weight=0.75,
                                 detail=f"One of {len(group)} payments to this vendor within {cfg.split_window_days} days, all approved by "
                                        f"{approver}. Together they come to {money(total)}, over {approver}'s {money(limit)} limit. "
                                        f"That is {total / usual[vendor]:.1f}× this vendor's usual payment.",
                                 related=[y for y in ids if y != x["payment_id"]]))
    return _findings(rows)


def _poisson_at_least(k: int, lam: float) -> float:
    """Chance of k or more events when lam are expected."""
    return 1 - sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k))


def just_below_limit(ds: Dataset, cfg: Settings) -> pd.DataFrame:
    """An approver repeatedly approving one vendor just under their own limit, far more often than chance."""
    limits = _limits(ds)
    p = ds.payments.assign(_limit=ds.payments.approved_by.map(limits).fillna(0))
    p["_near"] = (p._limit > 0) & (p.amount < p._limit) & (p.amount >= p._limit * cfg.just_below_share)
    # How often amounts land in the band for each approval limit, across everyone
    base = p[p._limit > 0].groupby("_limit")._near.mean()
    rows = []
    for (vendor, approver), g in p[p._limit > 0].groupby(["vendor_id", "approved_by"]):
        near = g[g._near]
        if len(near) < cfg.just_below_min_count:
            continue
        rate = base.get(g._limit.iloc[0], 0)
        chance = _binom_at_least(len(near), len(g), rate)
        if chance > 0.001:
            continue
        for _, r in near.iterrows():
            rows.append(dict(payment_id=r.payment_id, rule="just_below_limit", weight=0.6,
                             detail=f"{approver} approved {len(near)} of this vendor's {len(g)} payments between {money(r._limit * cfg.just_below_share)} "
                                    f"and their {money(r._limit)} limit (about {rate * len(g):.1f} expected).",
                             related=[x for x in near.payment_id if x != r.payment_id][:10]))
    return _findings(rows)


def _binom_at_least(k: int, n: int, q: float) -> float:
    if q <= 0:
        return 0.0
    return float(sum(math.comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1)))


def self_approval(ds: Dataset, cfg: Settings) -> pd.DataFrame:
    """The person who entered a payment also approved it (no separation of duties)."""
    p = ds.payments
    hit = p[p.created_by.notna() & (p.created_by == p.approved_by)]
    return _findings([dict(payment_id=r.payment_id, rule="self_approval", weight=0.7,
                           detail=f"{r.created_by} both entered and approved this payment.", related=[]) for r in hit.itertuples()])


def over_limit(ds: Dataset, cfg: Settings) -> pd.DataFrame:
    """Approved by someone whose limit is lower than the amount (or who can't approve at all)."""
    limits = _limits(ds)
    p = ds.payments
    rows = []
    for r in p.itertuples():
        lim = limits.get(r.approved_by)
        if lim is None:
            rows.append(dict(payment_id=r.payment_id, rule="unknown_approver", weight=0.6,
                             detail=f"Approved by {r.approved_by}, who is not in the employee list.", related=[]))
        elif r.amount > lim:
            rows.append(dict(payment_id=r.payment_id, rule="over_limit", weight=0.6,
                             detail=f"{money(r.amount)} approved by {r.approved_by}, whose limit is {money(lim)}.", related=[]))
    return _findings(rows)


# ---------------------------------------------------------------- vendors

def _norm_addr(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def employee_vendor_link(ds: Dataset, cfg: Settings) -> pd.DataFrame:
    """A vendor shares a bank account, phone or address with an employee: a possible ghost vendor."""
    e, v = ds.employees, ds.vendors
    by_acct = {a: r for a, r in zip(e.bank_account, e.itertuples()) if isinstance(a, str) and a}
    by_phone = {_digits(str(ph)): r for ph, r in zip(e.phone, e.itertuples()) if _digits(str(ph))}
    addrs = [(_norm_addr(r.address), r) for r in e.itertuples() if isinstance(r.address, str)]
    links = {}
    for vr in v.itertuples():
        why = []
        if vr.bank_account in by_acct:
            why.append((by_acct[vr.bank_account], "bank account"))
        ph = _digits(str(vr.phone))
        if ph and ph in by_phone:
            why.append((by_phone[ph], "phone number"))
        va = _norm_addr(vr.address)
        if va:
            for ea, er in addrs:
                # Typos are fine ("Mesopotamia" / "Mesopotomia"), a different house number is not
                if _digits(va) == _digits(ea) and fuzz.ratio(va, ea) >= cfg.fuzzy_address:
                    why.append((er, "address"))
                    break
        if why:
            links[vr.vendor_id] = why
    rows = []
    for r in ds.payments[ds.payments.vendor_id.isin(links)].itertuples():
        why = links[r.vendor_id]
        people = sorted({f"{w[0].name} ({w[0].employee_id})" for w in why})
        what = sorted({w[1] for w in why})
        acct = any(w[1] == "bank account" for w in why)
        rows.append(dict(payment_id=r.payment_id, rule="employee_vendor_link", weight=0.95 if acct else 0.8,
                         detail=f"This vendor has the same {' and '.join(what)} as employee {', '.join(people)}.", related=[]))
    return _findings(rows)


def bank_change(ds: Dataset, cfg: Settings) -> pd.DataFrame:
    """A large payment soon after the vendor's bank account was changed (a common business-email scam)."""
    v = ds.vendors.dropna(subset=["bank_changed_date"]).set_index("vendor_id")
    p = ds.payments
    usual = p.groupby("vendor_id").amount.median()
    rows = []
    for r in p[p.vendor_id.isin(v.index)].itertuples():
        changed = v.at[r.vendor_id, "bank_changed_date"]
        days = (r.payment_date.normalize() - changed).days
        if 0 <= days <= cfg.bank_change_days:
            big = r.amount >= cfg.bank_change_amount_factor * usual[r.vendor_id]
            rows.append(dict(payment_id=r.payment_id, rule="bank_change", weight=0.8 if big else 0.3,
                             detail=f"Paid {days} day(s) after this vendor's bank account was changed on {changed:%d %b %Y}"
                                    + (f", and {r.amount / usual[r.vendor_id]:.1f}× the vendor's usual payment ({money(usual[r.vendor_id])})." if big
                                       else ". Confirm the change with the vendor by phone."), related=[]))
    return _findings(rows)


def new_vendor_large(ds: Dataset, cfg: Settings) -> pd.DataFrame:
    """A big payment to a vendor set up very recently."""
    created = ds.vendors.set_index("vendor_id").created_date
    p = ds.payments
    typical = p.amount.median()
    rows = []
    for r in p.itertuples():
        c = created.get(r.vendor_id)
        if pd.isna(c):
            continue
        age = (r.payment_date.normalize() - c).days
        if 0 <= age <= cfg.new_vendor_days and r.amount >= cfg.new_vendor_amount_factor * typical:
            rows.append(dict(payment_id=r.payment_id, rule="new_vendor_large", weight=0.4,
                             detail=f"{money(r.amount)} paid to a vendor set up only {age} day(s) earlier.", related=[]))
    return _findings(rows)


# ---------------------------------------------------------------- the payment itself

def invoice_after_payment(ds: Dataset, cfg: Settings) -> pd.DataFrame:
    p = ds.payments.dropna(subset=["invoice_date"])
    hit = p[p.invoice_date > p.payment_date.dt.normalize()]
    return _findings([dict(payment_id=r.payment_id, rule="invoice_after_payment", weight=0.4,
                           detail=f"The invoice is dated {r.invoice_date:%d %b %Y}, after the payment was made.", related=[])
                      for r in hit.itertuples()])


def round_amount(ds: Dataset, cfg: Settings) -> pd.DataFrame:
    """Large, perfectly round amounts. Weak on its own (rent and contracts are often round), so only scored if it's unusual for the vendor."""
    p = ds.payments
    round_ = (p.amount >= cfg.round_amount_min) & (p.amount % 1000 == 0)
    share = round_.groupby(p.vendor_id).transform("mean")
    count = p.groupby("vendor_id").amount.transform("count")
    hit = p[round_ & ((share < 0.5) | (count < 4))]
    return _findings([dict(payment_id=r.payment_id, rule="round_amount", weight=0.15,
                           detail=f"A round {money(r.amount)}, unusual for this vendor's invoices.", related=[]) for r in hit.itertuples()])


def off_hours(ds: Dataset, cfg: Settings) -> pd.DataFrame:
    p = ds.payments
    start, end = cfg.office_hours
    rows = []
    for r in p.itertuples():
        d = r.payment_date
        why = "on a weekend" if d.weekday() >= 5 else ("on a public holiday (" + is_holiday(d) + ")" if is_holiday(d) else
                                                        (f"at {d:%H:%M}, outside office hours" if not start <= d.hour < end else None))
        if why:
            rows.append(dict(payment_id=r.payment_id, rule="off_hours", weight=0.1, detail=f"Entered {why}.", related=[]))
    return _findings(rows)


BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def benford(ds: Dataset, cfg: Settings) -> pd.DataFrame:
    """A vendor whose amounts don't follow Benford's law, which natural invoice amounts usually do. Invented numbers often don't."""
    p = ds.payments[ds.payments.amount >= 10]
    first = p.amount.astype(str).str.lstrip("0.").str[0].astype(int)
    rows = []
    for vendor, g in first.groupby(p.vendor_id):
        amounts = p.amount[g.index]
        # Benford's law only fits amounts spread over several orders of magnitude
        if len(g) < cfg.benford_min_payments or np.log10(amounts.quantile(0.95) / amounts.quantile(0.05)) < 1.5:
            continue
        seen = np.bincount(g, minlength=10)[1:] / len(g)
        mad = float(np.abs(seen - BENFORD).mean())
        if mad > cfg.benford_mad_flag:
            top = int(np.argmax(seen - BENFORD)) + 1
            for pid in p.payment_id[g.index]:
                rows.append(dict(payment_id=pid, rule="benford", weight=min(0.2, 0.1 + (mad - cfg.benford_mad_flag) * 5),
                                 detail=f"This vendor's {len(g)} amounts don't follow Benford's law (MAD {mad:.3f}); "
                                        f"too many start with {top} ({seen[top - 1]:.0%} vs {BENFORD[top - 1]:.0%} expected).", related=[]))
    return _findings(rows)


RULES = {
    "exact_duplicate": (exact_duplicates, "Duplicate invoice"),
    "near_duplicate": (near_duplicates, "Near-duplicate"),
    "split_payment": (split_payments, "Split payment"),
    "just_below_limit": (just_below_limit, "Just under approval limit"),
    "self_approval": (self_approval, "Self-approved"),
    "over_limit": (over_limit, "Over approval limit"),
    "employee_vendor_link": (employee_vendor_link, "Vendor linked to employee"),
    "bank_change": (bank_change, "Recent bank change"),
    "new_vendor_large": (new_vendor_large, "Large payment, new vendor"),
    "invoice_after_payment": (invoice_after_payment, "Invoice dated after payment"),
    "round_amount": (round_amount, "Round amount"),
    "off_hours": (off_hours, "Off hours"),
    "benford": (benford, "Benford's law"),
}
LABELS = {k: v[1] for k, v in RULES.items()} | {"unknown_approver": "Unknown approver", "outlier": "Statistical outlier"}


def run_rules(ds: Dataset, cfg: Settings | None = None) -> pd.DataFrame:
    cfg = cfg or Settings()
    parts = [fn(ds, cfg) for name, (fn, _) in RULES.items() if name in cfg.enabled]
    parts = [x for x in parts if len(x)]
    return pd.concat(parts, ignore_index=True) if parts else _findings([])
