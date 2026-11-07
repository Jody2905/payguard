"""Command line: payguard generate | scan | evaluate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .data import DataError, load_folder
from .rules import RULES, Settings


def _settings(args) -> Settings:
    s = Settings()
    if getattr(args, "skip", None):
        unknown = set(args.skip) - set(RULES)
        if unknown:
            sys.exit(f"Unknown rule(s): {', '.join(sorted(unknown))}. Rules: {', '.join(RULES)}")
        s.enabled = set(RULES) - set(args.skip)
    return s


def cmd_generate(args) -> None:
    from .generate import write
    t = write(args.out, n_payments=args.payments, seed=args.seed, year=args.year)
    print(f"Wrote {len(t['payments']):,} payments, {len(t['vendors'])} vendors and {len(t['employees'])} employees to {args.out}")
    print(f"{len(t['truth'])} of the payments are planted fraud (answers in truth.csv).")


def cmd_scan(args) -> None:
    from .report import to_excel, to_html
    from .scoring import analyse
    ds = load_folder(args.folder)
    result = analyse(ds, _settings(args), outlier_share=args.outliers)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    xlsx = to_excel(result, out / "payguard_cases.xlsx")
    page = to_html(result, out / "payguard_report.html", title=args.title, dashboard_url=args.dashboard_url)
    result.scored.to_csv(out / "scored_payments.csv", index=False)
    c = result.cases
    print(f"Checked {len(ds.payments):,} payments (EC${ds.payments.amount.sum():,.2f}).")
    print(f"{len(c):,} to review, worth EC${c.amount.sum():,.2f}: "
          f"{(c.level == 'Critical').sum()} critical, {(c.level == 'High').sum()} high, {(c.level == 'Medium').sum()} medium.")
    if len(c):
        print("\nTop 5:")
        for r in c.head(5).itertuples():
            print(f"  {r.payment_id}  risk {r.risk:5.1f}  EC${r.amount:>12,.2f}  {r.vendor[:34]:<34}  {r.reasons}")
    print(f"\nReports: {xlsx}\n         {page}")


def cmd_evaluate(args) -> None:
    from .evaluate import evaluate
    from .scoring import analyse
    folder = Path(args.folder)
    truth_file = folder / "truth.csv"
    if not truth_file.exists():
        sys.exit(f"No truth.csv in {folder}. Evaluate needs data from 'payguard generate'.")
    result = analyse(load_folder(folder), _settings(args), outlier_share=args.outliers)
    e = evaluate(result, pd.read_csv(truth_file), threshold=args.threshold)
    print(f"Review threshold: risk {e['threshold']:.0f}+")
    print(f"Planted fraud caught: {e['caught']} of {e['planted']} (recall {e['recall']:.0%})")
    print(f"Flagged payments that were planted fraud: {e['caught']} of {e['flagged']} (precision {e['precision']:.0%})")
    print(f"F1 score: {e['f1']:.2f}\n")
    for r in e["per_scheme"].itertuples():
        print(f"  {r.scheme_label:<45} {r.caught:>3} / {r.planted:<3} {r.recall:.0%}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="payguard", description="Find duplicate, split and suspicious payments.")
    ap.add_argument("--version", action="version", version=f"payguard {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="make sample data with planted fraud")
    g.add_argument("--out", default="sample_data")
    g.add_argument("--payments", type=int, default=5000)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--year", type=int, default=2025)
    g.set_defaults(fn=cmd_generate)

    for name, fn, helptext in (("scan", cmd_scan, "check a folder of payments.csv, vendors.csv, employees.csv"),
                               ("evaluate", cmd_evaluate, "measure accuracy on generated data")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("folder")
        p.add_argument("--skip", nargs="*", metavar="RULE", help=f"rules to turn off: {', '.join(RULES)}")
        p.add_argument("--outliers", type=float, default=0.01, help="share of payments to flag as statistical outliers (0 = off)")
        p.set_defaults(fn=fn)
    sub.choices["scan"].add_argument("--out", default="payguard_output")
    sub.choices["scan"].add_argument("--title", default="Payment review")
    sub.choices["scan"].add_argument("--dashboard-url", help="add an 'Open the live dashboard' button to the HTML report")
    sub.choices["evaluate"].add_argument("--threshold", type=float, default=35)

    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except DataError as e:
        sys.exit(f"Problem with the input files: {e}")


if __name__ == "__main__":
    main()
