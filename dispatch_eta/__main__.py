import os

# Keep the small experiment bounded; never modify global/user environment.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Original CPU ETA regression and promise simulation")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo")
    demo.add_argument("--out", default="reports")
    demo.add_argument("--seed", type=int, default=73)
    demo.add_argument("--events-per-day", type=int, default=180)
    demo.add_argument("--alpha", type=float, default=0.1)
    report = commands.add_parser("report")
    report.add_argument("--out", default="reports")
    args = parser.parse_args()
    if args.command == "demo":
        from .experiment import run
        metrics = run(args.out, args.seed, args.events_per_day, args.alpha)
        print(json.dumps({"counts": metrics["counts"], "calibration": metrics["calibration"],
                          "test": {name: slices["all"] for name, slices in metrics["test"].items()}}, indent=2))
    else:
        from .report import render
        render(args.out)
        print("Rendered reports from saved metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
