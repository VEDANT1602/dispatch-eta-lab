"""Narrative and SVG are generated from the same executed metrics."""

from pathlib import Path
import html
import json
import os


ROOT = Path(__file__).resolve().parents[1]


def number(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


def comparison_summary(metrics):
    linear = metrics["test"]["linear"]["all"]
    tree = metrics["test"]["xgboost"]["all"]

    def relation(a, b, smaller, greater):
        return smaller if a < b else greater if a > b else "equal to"

    mae = relation(tree["mae_minutes"], linear["mae_minutes"], "lower than", "higher than")
    sentences = [f"XGBoost's test MAE is {mae} the linear baseline ({number(tree['mae_minutes'])} versus {number(linear['mae_minutes'])} minutes)."]
    tw, lw = tree["mean_interval_width_minutes"], linear["mean_interval_width_minutes"]
    if tw is not None and lw is not None:
        width = relation(tw, lw, "narrower than", "wider than")
        sentences.append(f"Its mean interval width is {width} the linear interval ({number(tw)} versus {number(lw)} minutes).")
    else:
        sentences.append("At least one interval is unbounded; a finite mean-width comparison is unavailable.")
    tl, ll = tree["upper_promise"]["mean_loss_units"], linear["upper_promise"]["mean_loss_units"]
    if tl is not None and ll is not None:
        loss = relation(tl, ll, "lower than", "higher than")
        sentences.append(f"Under the fixed simulation weights, XGBoost-upper promise loss is {loss} linear-upper loss ({number(tl)} versus {number(ll)} illustrative units).")
    else:
        sentences.append("An unbounded upper promise has no finite loss comparison.")
    sentences.append("These are separate measured comparisons; choosing a policy on one synthetic stream does not establish deployment readiness or causal business value.")
    return " ".join(sentences)


def coverage_svg(metrics):
    slices = ["before_shift", "mechanism_shift", "rush_hour", "off_peak", "zone_core", "zone_suburban", "zone_hills"]
    elements = []
    for index, name in enumerate(slices):
        y = 80 + index * 57
        elements.append(f'<text x="18" y="{y + 16}">{html.escape(name)}</text>')
        for offset, model, color in [(0, "linear", "#7c8290"), (21, "xgboost", "#176f8b")]:
            item = metrics["test"][model][name]
            if item["coverage"] is not None:
                elements.append(f'<rect x="176" y="{y + offset}" width="{item["coverage"] * 360:.2f}" height="17" fill="{color}"/>')
                elements.append(f'<text x="{187 + item["coverage"] * 360:.2f}" y="{y + offset + 13}">{item["covered"]}/{item["n"]}</text>')
    return "\n".join([
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="530" viewBox="0 0 760 530" role="img" aria-labelledby="title desc">',
        '<title id="title">Observed ETA interval coverage by synthetic test slice</title>',
        '<desc id="desc">Gray: linear baseline. Blue: XGBoost. Dashed line: nominal target. Drift-period undercoverage is retained.</desc>',
        '<rect width="760" height="530" fill="#f5f8fb"/>',
        '<g font-family="Arial, sans-serif" font-size="14" fill="#17334c">',
        '<text x="18" y="29" font-size="22" font-weight="bold">Does yesterday\'s interval survive today\'s dispatch?</text>',
        '<text x="18" y="52">Observed coverage: gray linear / blue XGBoost. Labels are covered / slice count.</text>',
        f'<line x1="{176 + 360 * (1 - metrics["calibration"]["xgboost"]["alpha"]):.2f}" x2="{176 + 360 * (1 - metrics["calibration"]["xgboost"]["alpha"]):.2f}" y1="70" y2="486" stroke="#af4a39" stroke-dasharray="5 4"/>',
        *elements,
        '<text x="18" y="518">Synthetic protocol audit; nominal coverage is not guaranteed under temporal shift.</text>',
        '</g></svg>', '',
    ])


def render(output):
    output = Path(output).resolve()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    overall = metrics["test"]
    rows = []
    for name, values in overall.items():
        item = values["all"]
        rows.append(f'| {name} | {item["n"]} | {number(item["mae_minutes"])} | {number(item["coverage"] * 100)}% | {number(item["mean_interval_width_minutes"])} | {item["underestimate_over_10_minutes_count"]} |')
    slice_rows = []
    for name, item in overall["xgboost"].items():
        membership = f'{item["covered"]}/{item["n"]}' if item["n"] else "N/A (no rows)"
        slice_rows.append(f'| {name} | {item["n"]} | {number(item["mae_minutes"])} | {membership} | {number(item["mean_interval_width_minutes"])} | {item["upper_misses"]} |')
    policy_rows = []
    for name, values in overall.items():
        for policy in ("point_promise", "upper_promise"):
            item = values["all"][policy]
            policy_rows.append(f'| {name}: {policy} | {item["missed_promises"]}/{item["n"]} | {number(item["mean_late_minutes"])} | {number(item["mean_slack_minutes"])} | {number(item["mean_loss_units"])} |')
    relative = lambda path: Path(os.path.relpath(ROOT / path, output)).as_posix()
    candidate = overall["xgboost"]
    cal = metrics["calibration"]["xgboost"]
    report = f'''# Dispatch ETA Lab: the promise is a decision

**Decision: Keep this as an offline protocol audit. Point error, interval coverage and promise loss require separate acceptance criteria.**

This is original independent work by Vedant Thaker. Every delivery is fictional. No Uber, DoorDash, Instacart, employer or customer data, code, benchmark result or production claim is used as this experiment's evidence.

## The topic, then the narrower question

An ETA is a regression output, but a displayed delivery promise is an operational commitment. I wanted to test whether a tree model improves a simple baseline and whether its uncertainty still describes the cases where a promise matters: rush hour, different service zones and a changed delivery mechanism.

## The usual workflow and the issue I chose to expose

It is easy to fit a model, report one average error and turn every prediction into a promise. That hides late arrivals behind early ones, obscures who receives wide intervals, and can let future route/courier information enter an apparently clean test. An uncertainty interval adds another failure mode: its familiar nominal level can sound like a guarantee even when calibration and future deliveries are no longer exchangeable.

## My framework: clock, estimate, calibrate, promise, audit

1. **Clock:** use known-at-dispatch inputs. Keep route/courier identities disjoint between train, calibration and test; exclude identities, absolute day, shift flags, outcomes and completion fields from features. Leave two-day gaps and require earlier labels to be available before the next partition.
2. **Estimate:** fit OLS and an actual CPU XGBoost regressor on the same training rows and feature columns. Parameters are fixed before the run; neither calibration nor test labels select a model or stop boosting.
3. **Calibrate:** use only the separate pre-test calibration residuals. For nominal coverage {100 * (1 - cal["alpha"]):.0f}%, select absolute-residual rank `ceil((n+1)*(1-alpha))`: {cal["rank"]} of {cal["n"]}. The XGBoost radius is {number(cal["radius_minutes"])} minutes. Lower endpoints are clipped to the known nonnegative duration support.
4. **Promise:** compare displaying the point estimate with displaying the interval's upper endpoint. Count missed promises, mean late minutes and extra promise slack. Use predeclared illustrative loss weights of four per late minute and one per slack minute.
5. **Audit:** keep all test predictions and show the original/shifted periods and service-zone/rush-hour slices. The shift flag is an analysis label, never a predictor or a trigger available to a deployed policy.

## Actual executed result

The generated stream contains {metrics["dataset"]["rows"]} events. The train/calibration/test counts are {metrics["counts"]["train"]}/{metrics["counts"]["calibration"]}/{metrics["counts"]["test"]}; {metrics["counts"]["embargo_excluded"]} events fall in excluded gap days.

| Model | Test deliveries | MAE minutes | Interval coverage | Mean interval width | Underestimated by >10 minutes |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

| XGBoost slice | Deliveries | MAE minutes | Covered / evaluated | Mean width | Actual above upper endpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(slice_rows)}

Before the declared shift, XGBoost coverage is {number(candidate["before_shift"]["coverage"] * 100)}%; during the shift it is {number(candidate["mechanism_shift"]["coverage"] * 100)}%. These are observed counts on one constructed stream. The standard split-conformal finite-sample guarantee needs exchangeability; it is not guaranteed by chronological splitting, and the explicit disruption violates it. Global marginal coverage also does not promise coverage in each zone or rush-hour subgroup.

![Coverage audit](coverage.svg)

## The decision layer makes the tradeoff visible

| Model and promise policy | Missed promises / deliveries | Mean late minutes | Mean slack minutes | Mean illustrative loss units |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(policy_rows)}

Late and slack minutes are averaged over all evaluated deliveries, including zeros. The upper endpoint can reduce missed promises while increasing the amount of time quoted beyond actual completion. The weights are transparent preferences, not customer-loss estimates, currency, observed satisfaction or ROI. This simulation does not model demand changes caused by a longer promise, dispatcher choices, cancellations or intervention feedback. It cannot establish causal business impact.

{comparison_summary(metrics)}

## The tradeoff and next action

Trees can capture interactions that a linear baseline misses. Calibration makes one uncertainty assumption inspectable; it does not repair a changed process. I would next test multiple predeclared time windows and seeds on authorized representative data, observe actual feature/label arrival times, and choose a promise-loss function with the operator. A monitored shadow rollout would compare interval misses and slack by group before a customer sees a new promise. Any recalibration policy needs fresh past-only labels and its own predeclared evaluation.

## Reproduce and challenge

Run `python -m dispatch_eta demo --out reports`, then `python -m unittest discover -s tests -v`. `python -m dispatch_eta report --out reports` regenerates the narrative/chart from the saved numerical evidence without fitting again.

- [Metrics and source hashes](metrics.json)
- [Every held-out prediction](predictions.csv)
- [Exact original synthetic dataset](data/deliveries.csv.gz)
- [Data card]({relative("data/DATA_CARD.md")})
- [Protocol and references]({relative("docs/protocol.md")})

Dataset CSV SHA-256: `{metrics["dataset"]["csv_sha256"]}`.
'''
    (output / "case-study.md").write_text(report, encoding="utf-8", newline="\n")
    (output / "coverage.svg").write_text(coverage_svg(metrics), encoding="utf-8", newline="\n")
