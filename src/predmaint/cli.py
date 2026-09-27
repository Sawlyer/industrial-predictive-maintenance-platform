"""Command line entry point.

    predmaint data          download the real NASA C-MAPSS dataset
    predmaint demo-data     write a synthetic fleet so everything runs offline
    predmaint train         build features, cross-validate, fit, save artifacts
    predmaint report        print the last training scorecard
    predmaint figures       export the README figures to reports/figures/
"""

from __future__ import annotations

import argparse
import json
import sys

from predmaint.config import DEFAULT_SUBSET, METRICS_PATH, SUBSETS


def _cmd_data(args: argparse.Namespace) -> int:
    from predmaint.data.download import fetch

    return 0 if fetch(args.subset, force=args.force) else 1


def _cmd_demo_data(args: argparse.Namespace) -> int:
    from predmaint.data.download import generate_demo_data

    generate_demo_data(args.subset, n_train_units=args.train_units, n_test_units=args.test_units)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from predmaint.models.train import train_models

    report = train_models(subset=args.subset, run_cv=not args.no_cv)
    _print_scorecard(report)
    return 0


def _cmd_report(_: argparse.Namespace) -> int:
    if not METRICS_PATH.exists():
        print(f"No metrics at {METRICS_PATH}. Run `predmaint train` first.")
        return 1
    _print_scorecard(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
    return 0


def _cmd_figures(args: argparse.Namespace) -> int:
    from predmaint.reporting import export_figures

    paths = export_figures(subset=args.subset)
    for path in paths:
        print(f"wrote {path}")
    return 0


def _print_scorecard(report: dict) -> None:
    test = report.get("test", {})
    model, baseline, risk = test.get("model", {}), test.get("baseline", {}), test.get("risk", {})
    print(f"\nSubset {report.get('subset')} | {report.get('n_features')} features | "
          f"{report.get('train_seconds')}s")
    print("\nHeld-out engines, last observed cycle")
    print(f"  {'metric':<22}{'baseline':>12}{'model':>12}")
    for key, label in (
        ("rmse", "RMSE (cycles)"),
        ("mae", "MAE (cycles)"),
        ("nasa_score_per_engine", "NASA score / engine"),
        ("late_prediction_rate", "Late prediction rate"),
    ):
        left = baseline.get(key, float("nan"))
        right = model.get(key, float("nan"))
        print(f"  {label:<22}{left:>12.3f}{right:>12.3f}")
    if risk:
        print(f"\nFailure within horizon: recall {risk.get('recall', 0):.3f} | "
              f"precision {risk.get('precision', 0):.3f} | PR-AUC {risk.get('pr_auc', 0):.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="predmaint", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def with_subset(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--subset", default=DEFAULT_SUBSET, choices=SUBSETS)
        return p

    data = with_subset(sub.add_parser("data", help="download the NASA C-MAPSS dataset"))
    data.add_argument("--force", action="store_true", help="re-download even if present")
    data.set_defaults(func=_cmd_data)

    demo = with_subset(sub.add_parser("demo-data", help="generate a synthetic fleet"))
    demo.add_argument("--train-units", type=int, default=100)
    demo.add_argument("--test-units", type=int, default=50)
    demo.set_defaults(func=_cmd_demo_data)

    train = with_subset(sub.add_parser("train", help="train and evaluate both heads"))
    train.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    train.set_defaults(func=_cmd_train)

    sub.add_parser("report", help="print the last scorecard").set_defaults(func=_cmd_report)

    with_subset(sub.add_parser("figures", help="export README figures")).set_defaults(
        func=_cmd_figures
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
