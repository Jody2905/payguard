"""Realistic sample accounts-payable data for a small government treasury, with fraud planted in it.

Everything here is invented: vendors, people, accounts and phone numbers. The planted schemes are written to
truth.csv so you can measure how many PayGuard catches (see evaluate.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .holidays import is_holiday

VILLAGES = ["Kingstown", "Arnos Vale", "Calliaqua", "Georgetown", "Barrouallie", "Layou", "Mesopotamia", "Richland Park",
            "Villa", "Stubbs", "Biabou", "Chateaubelair", "Fancy", "Sandy Bay", "Campden Park", "Questelles", "Belair",
            "Marriaqua", "Union Island", "Port Elizabeth"]
STREETS = ["Main Road", "Back Street", "Bay Street", "Middle Street", "Lodge Road", "Murray Road", "Sion Hill",
           "Cane Garden", "Ratho Mill", "Dorsetshire Hill", "Montrose", "Kingstown Park"]
WORDS = ["Windward", "Leeward", "Grenadine", "Sunrise", "Coral", "Breadfruit", "Soufriere", "Blue Lagoon", "Island",
         "Harbour", "Trade Wind", "Arrowroot", "Mountain", "Palm", "Seaview", "Caribbean", "Valley", "Hummingbird",
         "Bamboo", "Tamarind", "Saltwhistle", "Black Sand", "Indian Bay", "Rainforest", "Flambeau"]
TRADES = [("Office Supplies", 900, 0.5), ("Hardware", 2500, 0.7), ("Construction", 38000, 0.8), ("Medical Supplies", 6500, 0.6),
          ("IT Services", 7800, 0.6), ("Catering", 1800, 0.5), ("Transport", 1400, 0.5), ("Cleaning Services", 3200, 0.3),
          ("Fuel", 4200, 0.4), ("Printing", 1600, 0.5), ("Security Services", 9000, 0.2), ("Furniture", 5200, 0.6),
          ("Electrical", 3600, 0.6), ("Agricultural Supplies", 2800, 0.6), ("Consulting", 15000, 0.7)]
SUFFIX = ["Ltd", "Inc.", "& Sons", "Ltd", "Enterprises", "Co. Ltd", "Services", "Ltd"]
DEPARTMENTS = ["Health", "Education", "Works", "Agriculture", "Tourism", "Finance", "Housing", "Youth & Sports", "Police", "Customs"]
FIRST = ["Akeem", "Shanice", "Kemar", "Rhea", "Delroy", "Natasha", "Jamal", "Keisha", "Marlon", "Tricia", "Devon", "Simone",
         "Andre", "Kerry-Ann", "Jason", "Latoya", "Rondell", "Sherlene", "Kirk", "Aneesa", "Glenroy", "Camille", "Shawn",
         "Petra", "Orande", "Kamala", "Terrence", "Yolanda", "Nigel", "Dionne", "Winston", "Germaine", "Lesroy", "Odessa"]
LAST = ["Samuel", "Daniel", "Browne", "John", "Charles", "Williams", "Ollivierre", "Jack", "Cupid", "Baptiste", "Hackshaw",
        "Providence", "Jackson", "Thomas", "Mills", "Ballantyne", "DaSilva", "Gonsalves", "Shallow", "Quow", "Bacchus",
        "King", "Lewis", "Child", "Stephens", "Matthews", "Alexander", "Hazell", "Joseph", "Peters"]
LIMITS = [0, 0, 5_000, 25_000, 100_000]  # clerks can't approve; three approval tiers


def _phone(rng) -> str:
    return f"784-{rng.integers(450, 535)}-{rng.integers(1000, 9999)}"


def _account(rng) -> str:
    return f"{rng.choice(['WB', 'GB', 'CU', 'NB'])}-{rng.integers(10_000_000, 99_999_999)}"


def _address(rng) -> str:
    return f"{rng.integers(1, 120)} {rng.choice(STREETS)}, {rng.choice(VILLAGES)}"


def generate(n_payments: int = 5000, seed: int = 7, year: int = 2025) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")

    # ---- employees ----
    names = set()
    employees = []
    while len(employees) < 48:
        nm = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if nm in names:
            continue
        names.add(nm)
        i = len(employees)
        limit = 250_000 if i == 0 else LIMITS[i % len(LIMITS)]
        employees.append(dict(employee_id=f"E{100 + i}", name=nm, address=_address(rng), phone=_phone(rng),
                              bank_account=_account(rng), department=DEPARTMENTS[i % len(DEPARTMENTS)], approval_limit=limit))
    emp = pd.DataFrame(employees)
    clerks = emp[emp.approval_limit == 0].employee_id.tolist()
    approvers = emp[emp.approval_limit > 0].sort_values("approval_limit")

    # ---- vendors ----
    vendors, seen = [], set()
    while len(vendors) < 160:
        trade, typical, spread = TRADES[rng.integers(len(TRADES))]
        nm = f"{rng.choice(WORDS)} {trade} {rng.choice(SUFFIX)}"
        if nm in seen:
            continue
        seen.add(nm)
        i = len(vendors)
        addr = _address(rng)
        if addr in set(emp.address):   # keep legitimate vendors clear of staff addresses
            continue
        created = start - pd.Timedelta(days=int(rng.integers(60, 2000)))
        vendors.append(dict(vendor_id=f"V{1000 + i}", name=nm, address=addr, phone=_phone(rng), bank_account=_account(rng),
                            created_date=created.normalize(), created_by=rng.choice(clerks), bank_changed_date=pd.NaT,
                            _typical=typical * rng.uniform(0.6, 1.6), _spread=spread, _weight=min(rng.pareto(1.5) + 0.3, 6.0),
                            _prefix=rng.choice(["INV", "IN", "", "SI", "BILL"]), _seq=int(rng.integers(100, 5000))))
    ven = pd.DataFrame(vendors)
    # A few vendors bill a fixed monthly amount (rent, security contract, leased lines): legit repeats and round sums.
    fixed = rng.choice(len(ven), 8, replace=False)
    ven.loc[fixed, "_fixed"] = rng.choice([2_500, 4_000, 6_000, 8_500, 10_000, 12_000], len(fixed))

    def invoice(vi: int) -> str:
        ven.at[vi, "_seq"] += int(rng.integers(1, 40))
        pre = ven.at[vi, "_prefix"]
        return f"{pre}-{ven.at[vi, '_seq']}" if pre else str(ven.at[vi, "_seq"])

    def when(day: pd.Timestamp) -> pd.Timestamp:
        # Office hours, weekdays (the odd late entry happens legitimately too)
        while day.weekday() >= 5 or is_holiday(day):
            day += pd.Timedelta(days=1)
        hour = rng.normal(11.5, 2.2) if rng.random() > 0.02 else rng.uniform(17.5, 20)
        return day + pd.Timedelta(minutes=int(np.clip(hour, 7.5, 20.5) * 60))

    def approver_for(amount: float, avoid: str | None = None) -> str:
        ok = approvers[(approvers.approval_limit >= amount) & (approvers.employee_id != avoid)]
        # usually the lowest tier that can approve it
        tier = ok.approval_limit.min()
        pool = ok[ok.approval_limit == tier] if rng.random() < 0.8 else ok
        return rng.choice(pool.employee_id.tolist())

    pay = []

    def add(vi: int, day: pd.Timestamp, amount: float, inv: str | None = None, account: str | None = None,
            created_by: str | None = None, approved_by: str | None = None, desc: str | None = None, dept: str | None = None):
        v = ven.loc[vi]
        amount = round(float(amount), 2)
        cb = created_by or rng.choice(clerks)
        pay.append(dict(payment_id=None, payment_date=when(day), vendor_id=v.vendor_id, invoice_no=inv or invoice(vi),
                        invoice_date=(day - pd.Timedelta(days=int(rng.integers(3, 35)))).normalize(), amount=amount,
                        bank_account=account or v.bank_account, created_by=cb,
                        approved_by=approved_by or approver_for(amount, avoid=cb),
                        department=dept or rng.choice(DEPARTMENTS), description=desc or v["name"].split(" ")[-2] + " " + rng.choice(
                            ["order", "invoice", "services", "supplies", "works", "contract payment"])))
        return len(pay) - 1

    # ---- ordinary payments ----
    weights = (ven._weight / ven._weight.sum()).to_numpy()
    days = (end - start).days
    for vi in np.where(ven._fixed.notna())[0]:
        for m in range(1, 13):
            add(vi, pd.Timestamp(year, m, int(rng.integers(1, 6))), ven.at[vi, "_fixed"], desc="Monthly contract")
    while len(pay) < n_payments:
        vi = int(rng.choice(len(ven), p=weights))
        if not pd.isna(ven.at[vi, "_fixed"]):
            continue
        amt = rng.lognormal(np.log(ven.at[vi, "_typical"]), ven.at[vi, "_spread"])
        amt = min(amt, 240_000)
        add(vi, start + pd.Timedelta(days=int(rng.integers(0, days))), amt)

    truth: list[dict] = []

    def plant(idx: int | list[int], scheme: str, note: str):
        for i in ([idx] if isinstance(idx, int) else idx):
            truth.append(dict(row=i, scheme=scheme, note=note))

    base = list(range(len(pay)))

    # 1. Exact duplicates: the same invoice paid twice.
    for i in rng.choice(base, 22, replace=False):
        src = pay[i]
        vi = int(ven.index[ven.vendor_id == src["vendor_id"]][0])
        j = add(vi, src["payment_date"].normalize() + pd.Timedelta(days=int(rng.integers(1, 25))), src["amount"], inv=src["invoice_no"])
        plant(j, "duplicate", f"Invoice {src['invoice_no']} paid twice")

    # 2. Near duplicates: same amount, invoice number lightly altered to get past an exact-match check.
    def alter(inv: str) -> str:
        r = rng.integers(4)
        if r == 0:
            return inv.replace("-", "")
        if r == 1:
            return inv + "A"
        if r == 2:
            return inv.replace("-", "/") if "-" in inv else inv + "-1"
        return inv[:-2] + inv[-1] + inv[-2] if len(inv) > 3 else inv + "0"
    for i in rng.choice(base, 18, replace=False):
        src = pay[i]
        vi = int(ven.index[ven.vendor_id == src["vendor_id"]][0])
        j = add(vi, src["payment_date"].normalize() + pd.Timedelta(days=int(rng.integers(1, 12))), src["amount"], inv=alter(src["invoice_no"]))
        plant(j, "near_duplicate", f"Re-submitted invoice {src['invoice_no']} with a changed number")

    # 3. Split payments: one purchase cut into pieces so each stays under an approver's limit.
    tiers = [5_000, 25_000, 100_000]
    for k in range(12):
        limit = tiers[k % 3]
        appr = rng.choice(approvers[approvers.approval_limit == limit].employee_id.tolist())
        vi = int(rng.integers(len(ven)))
        parts = int(rng.integers(2, 5))
        # Each piece kept under the limit, most of them close to it
        amounts = limit * rng.uniform(0.5, 0.995, parts)
        day = start + pd.Timedelta(days=int(rng.integers(0, days - 5)))
        ids = [add(vi, day + pd.Timedelta(days=int(rng.integers(0, 3))), a, approved_by=appr) for a in amounts]
        plant(ids, "split_payment", f"EC${amounts.sum():,.0f} split into {parts} payments under the EC${limit:,} limit")

    # 4. Ghost vendors: an employee sets up a fake vendor paid into their own account (or at their address).
    for k, e in enumerate(emp.sample(4, random_state=seed).itertuples()):
        vi = len(ven)
        ven.loc[vi] = dict(vendor_id=f"V{1000 + vi}", name=f"{rng.choice(WORDS)} {rng.choice(['Consulting', 'Supplies', 'Maintenance'])} Services",
                           address=e.address if k % 2 else _address(rng), phone=e.phone if k == 2 else _phone(rng),
                           bank_account=e.bank_account if k != 1 else _account(rng),
                           created_date=(start + pd.Timedelta(days=int(rng.integers(20, 200)))).normalize(), created_by=e.employee_id,
                           bank_changed_date=pd.NaT, _typical=3000, _spread=0.3, _weight=0, _prefix="INV", _seq=int(rng.integers(10, 99)),
                           _fixed=np.nan)
        ids = []
        for _ in range(int(rng.integers(3, 8))):
            day = ven.at[vi, "created_date"] + pd.Timedelta(days=int(rng.integers(5, 150)))
            amt = rng.uniform(1_800, 4_950)
            ids.append(add(vi, day, amt, created_by=e.employee_id, desc="Consulting services"))
        plant(ids, "ghost_vendor", f"Vendor set up by {e.employee_id} using their own details")

    # 5. Bank-change fraud: a vendor's account is switched (often by a fake email from "the vendor"), then a big payment goes out.
    for vi in rng.choice(len(ven) - 4, 6, replace=False):
        vi = int(vi)
        when_changed = start + pd.Timedelta(days=int(rng.integers(30, days - 30)))
        new_acct = _account(rng)
        ven.at[vi, "bank_changed_date"] = when_changed.normalize()
        ven.at[vi, "bank_account"] = new_acct
        old = ven.at[vi, "_typical"]
        ids = [add(vi, when_changed + pd.Timedelta(days=int(rng.integers(1, 12))), old * rng.uniform(3, 6), account=new_acct)]
        plant(ids, "bank_change", "Account changed shortly before an unusually large payment")
        # later payments go to the new account too
        for p in pay:
            if p["vendor_id"] == ven.at[vi, "vendor_id"] and p["payment_date"] > when_changed and p is not pay[ids[0]]:
                p["bank_account"] = new_acct

    # 6. Self-approval: the same person entered and approved the payment.
    for i in rng.choice(base, 14, replace=False):
        p = pay[i]
        able = approvers[approvers.approval_limit >= p["amount"]].employee_id.tolist()
        if able:
            p["created_by"] = p["approved_by"] = rng.choice(able)
            plant(int(i), "self_approval", "Entered and approved by the same person")

    # 7. Kickback pattern: an approver keeps approving one vendor just under their limit.
    appr = approvers[approvers.approval_limit == 5_000].iloc[0].employee_id
    vi = int(rng.integers(len(ven) - 4))
    ids = [add(vi, start + pd.Timedelta(days=int(rng.integers(0, days))), rng.uniform(4_700, 4_995), approved_by=appr) for _ in range(14)]
    plant(ids, "just_below_limit", f"{appr} approved this vendor just under their EC$5,000 limit 14 times")

    # ---- finish up ----
    payments = pd.DataFrame(pay)
    payments["_row"] = range(len(payments))
    payments = payments.sort_values("payment_date").reset_index(drop=True)
    payments["payment_id"] = [f"P{year % 100:02d}{i + 1:05d}" for i in range(len(payments))]
    row_to_id = dict(zip(payments["_row"], payments["payment_id"]))
    truth_df = pd.DataFrame(truth)
    truth_df["payment_id"] = truth_df["row"].map(row_to_id)
    truth_df = truth_df.drop(columns="row").drop_duplicates("payment_id")[["payment_id", "scheme", "note"]]

    vendors_out = ven[[c for c in ven.columns if not c.startswith("_")]].copy()
    for c in ("created_date", "bank_changed_date"):
        vendors_out[c] = pd.to_datetime(vendors_out[c]).dt.strftime("%Y-%m-%d")
    payments["payment_date"] = payments["payment_date"].dt.strftime("%Y-%m-%d %H:%M")
    payments["invoice_date"] = payments["invoice_date"].dt.strftime("%Y-%m-%d")
    return {"payments": payments.drop(columns="_row"), "vendors": vendors_out, "employees": emp, "truth": truth_df}


def write(folder: str | Path, **kw) -> dict[str, pd.DataFrame]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tables = generate(**kw)
    for name, df in tables.items():
        df.to_csv(folder / f"{name}.csv", index=False)
    return tables
