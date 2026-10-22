import csv
import json
from pathlib import Path
from statistics import median
from PIL import Image, ImageDraw

RULES = ("matches", "inlier_ratio", "reliability")


def accepted(row, rule, threshold):
    return row["corner_error"] is not None and threshold is not None and row[rule] >= threshold


def fit_threshold(rows, rule, target, good_px):
    scores = sorted({r[rule] for r in rows if r["corner_error"] is not None})
    for cut in scores:
        group = [r for r in rows if accepted(r, rule, cut)]
        if sum(r["corner_error"] <= good_px for r in group)/len(group) >= target:
            return cut
    return None


def summarize(rows, rule, cut, good_px):
    acc = [r for r in rows if accepted(r, rule, cut)]
    errors = [r["corner_error"] for r in acc]
    wrong = sum(x > good_px for x in errors)
    return dict(pairs=len(rows), accepted=len(acc), false_accepts=wrong,
                estimate_failures=sum(r["corner_error"] is None for r in rows),
                coverage=len(acc)/len(rows), false_accept_fraction=wrong/len(rows),
                accepted_risk=wrong/len(acc) if acc else None,
                median_accepted_error_px=median(errors) if errors else None)


def evaluate(rows, cfg, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary, failures = [], []
    for method in sorted({r["method"] for r in rows}):
        dev = [r for r in rows if r["method"] == method and r["split"] == "dev"]
        test = [r for r in rows if r["method"] == method and r["split"] == "test"]
        if not dev or not test:
            raise ValueError("Separate development and test sequences are required.")
        if {r["sequence"] for r in dev} & {r["sequence"] for r in test}:
            raise ValueError("Sequence leakage")
        for rule in RULES:
            cut = fit_threshold(dev, rule, cfg["target_dev_precision"], cfg["correct_corner_px"])
            calibrated = summarize(dev,rule,cut,cfg["correct_corner_px"])
            for kind in ("all", "illumination", "viewpoint"):
                group = test if kind == "all" else [r for r in test if r["kind"] == kind]
                if not group:
                    continue
                metrics = summarize(group, rule, cut, cfg["correct_corner_px"])
                summary.append(dict(method=method, rule=rule, test_group=kind, threshold=cut,
                                    dev_pairs=len(dev), dev_accepted=calibrated["accepted"],
                                    dev_false_accepts=calibrated["false_accepts"], **metrics))
            mistakes = [r for r in test if accepted(r, rule, cut) and r["corner_error"] > cfg["correct_corner_px"]]
            if mistakes:
                worst = max(mistakes, key=lambda r: r["corner_error"])
                failures.append(dict(method=method, rule=rule, sequence=worst["sequence"],
                                     target=worst["target"], corner_error=worst["corner_error"]))
    with (out/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    (out/"false_accepts.json").write_text(json.dumps(failures, indent=2)+"\n", encoding="utf-8")
    draw_chart([s for s in summary if s["test_group"] == "all"], out/"comparison.png")
    return summary


def draw_chart(rows, path):
    image = Image.new("RGB", (830, 92 + len(rows)*52), "#f8fafc")
    d = ImageDraw.Draw(image)
    d.text((18, 12), "HPatches | held-out reliability, same estimated homographies", fill="#182e47")
    d.text((18, 34), "Blue = accepted / all   |   Orange = incorrect accepts / all   |   correct at <= 5 px", fill="#384657")
    for i, r in enumerate(rows):
        y = 72 + i*52
        d.text((18, y), r["method"]+" / "+r["rule"], fill="#182e47")
        x, width = 180, 460
        for j, (key, color) in enumerate([("coverage","#397ab3"),("false_accept_fraction","#be6339")]):
            offset = y+j*14
            d.rectangle((x,offset,x+width,offset+9),fill="#e4e9ee")
            d.rectangle((x,offset,x+round(width*r[key]),offset+9),fill=color)
            d.text((x+width+12,offset-2), f"{r[key]:.1%}", fill="#182e47")
    image.save(path)
