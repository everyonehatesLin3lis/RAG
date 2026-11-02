"""Charts for the README, drawn as SVG from evaluation/results/analytics.json (run analytics_report.py first).

    python evaluation/charts.py

Writes three files into docs/images/:
    model_scorecard.svg      each answer model against the "good enough" thresholds below
    model_tradeoff.svg       answer accuracy vs cost per 1,000 answers, one dot per model
    pipeline_scorecard.svg   the same scorecard for the pipeline settings (context size, query translation)

The thresholds are what this project counts as good enough for a movie research assistant: almost every answer
right, practically no unsupported claims, an answer within a few seconds, and running costs under a dollar per 1,000
questions. Plain SVG with no plotting library, so the charts render on GitHub in light and dark mode.
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYTICS = ROOT / "evaluation" / "results" / "analytics.json"
OUT = ROOT / "docs" / "images"

THRESHOLDS = {"accuracy": 0.95, "groundedness": 0.99, "latency_s": 6.0, "cost_1000": 1.0}

MODEL_NAMES = {
    "mimo-v2.6-flash (baseline)": "MiMo v2.6 Flash (in use)",
    "gpt-6-luna": "GPT-6 Luna",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "gpt-6-astra": "GPT-6 Astra",
}
PIPELINE_NAMES = {
    "8 chunks (baseline)": "Default: 8 chunks, translation on",
    "5 chunks": "5 chunks",
    "10 chunks": "10 chunks",
    "translation off": "Query translation off",
}

STYLE = """<style>
  .ink { fill: #0b0b0b; } .ink2 { fill: #52514e; } .muted { fill: #898781; }
  .grid { stroke: #e1e0d9; stroke-width: 1; } .track { stroke: #e1e0d9; stroke-width: 6; stroke-linecap: round; }
  .limit { stroke: #52514e; stroke-width: 1.5; stroke-dasharray: 4 3; }
  .range { stroke: #898781; stroke-width: 2; } .zone { fill: #0ca30c; fill-opacity: 0.08; }
  .pass { fill: #0ca30c; } .fail { fill: #d03b3b; } .ring { stroke: #fcfcfb; stroke-width: 2; }
  .passText { fill: #006300; } .failText { fill: #b02a2a; }
  .s1 { fill: #2a78d6; } .s2 { fill: #eb6834; } .s3 { fill: #1baf7a; } .s4 { fill: #6250d6; }
  @media (prefers-color-scheme: dark) {
    .ink { fill: #f0efec; } .ink2 { fill: #c3c2b7; } .grid, .track { stroke: #2c2c2a; }
    .limit { stroke: #c3c2b7; } .ring { stroke: #1a1a19; } .zone { fill-opacity: 0.14; }
    .passText { fill: #0ca30c; } .failText { fill: #e66767; }
    .s1 { fill: #3987e5; } .s2 { fill: #d95926; } .s3 { fill: #199e70; } .s4 { fill: #9085e9; }
  }
</style>"""


def svg(width: int, height: int, title: str, body: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}" font-family="system-ui, -apple-system, \'Segoe UI\', sans-serif">\n'
            f"  <title>{title}</title>\n  {STYLE}\n" + "\n".join(body) + "\n</svg>\n")


def text(x, y, s, cls="ink", size=13, anchor="start", weight=None):
    w = f' font-weight="{weight}"' if weight else ""
    return f'  <text class="{cls}" x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}"{w}>{s}</text>'


def stat(setup: dict, key: str) -> tuple[float, float, float]:
    """(mean, min, max) of a judged metric; min = max = mean for a setup that ran once."""
    v = setup[key]
    return (v["mean"], v["min"], v["max"]) if isinstance(v, dict) else (v, v, v)


def metrics(setup: dict) -> dict:
    return {
        "accuracy": stat(setup, "answer_accuracy"),
        "groundedness": stat(setup, "groundedness"),
        "latency_s": (setup["latency_ms_median"] / 1000,) * 3,
        "cost_1000": (setup["cost_usd_per_1000_answers"],) * 3,
    }


# --- scorecard: one row per setup, one column per metric, dashed line = threshold ------------------------------

COLUMNS = [
    # key, title, threshold label, scale (lo, hi, log?), passes(value), value label
    ("accuracy", "Answer accuracy", "good enough ≥ 0.95", (0.90, 1.00, False), lambda v: v >= 0.95, "{:.3f}"),
    ("groundedness", "Groundedness", "good enough ≥ 0.99", (0.98, 1.00, False), lambda v: v >= 0.99, "{:.3f}"),
    ("latency_s", "Median time per answer", "good enough ≤ 6 s", (0.0, 12.0, False), lambda v: v <= 6.0, "{:.1f} s"),
    ("cost_1000", "Cost per 1,000 answers", "good enough ≤ $1", (0.1, 100.0, True), lambda v: v <= 1.0, "${:,.2f}"),
]


def position(value: float, scale: tuple, x0: float, width: float) -> float:
    lo, hi, log = scale
    if log:
        f = (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    else:
        f = (value - lo) / (hi - lo)
    return x0 + max(0.0, min(1.0, f)) * width


def scorecard(rows: list[tuple[str, dict, int]], title: str) -> str:
    label_w, col_w, gap, row_h, top = 250, 150, 22, 52, 86
    width = label_w + len(COLUMNS) * (col_w + gap) + 80
    height = top + len(rows) * row_h + 34
    body = []
    for i, (key, name, limit, scale, _, _) in enumerate(COLUMNS):
        x0 = label_w + i * (col_w + gap)
        body.append(text(x0, 30, name, "ink", 13, weight="600"))
        body.append(text(x0, 48, limit, "muted", 12))
        lx = position(THRESHOLDS[key], scale, x0, col_w)
        body.append(f'  <line class="limit" x1="{lx:.1f}" y1="58" x2="{lx:.1f}" y2="{height - 30}"/>')
    body.append(text(width - 12, 30, "Meets", "ink", 13, "end", "600"))
    body.append(text(width - 12, 48, "all four", "muted", 12, "end"))

    for r, (name, setup, runs) in enumerate(rows):
        y = top + r * row_h
        m = metrics(setup)
        body.append(f'  <line class="grid" x1="0" y1="{y + row_h - 10}" x2="{width}" y2="{y + row_h - 10}"/>')
        body.append(text(0, y + 12, name, "ink", 14, weight="600" if r == 0 else None))
        body.append(text(0, y + 30, f"{runs} run{'s' if runs > 1 else ''}" + (" (range shown)" if runs > 1 else ""),
                         "muted", 12))
        passed = 0
        for i, (key, _, _, scale, ok, fmt) in enumerate(COLUMNS):
            x0 = label_w + i * (col_w + gap)
            mean, lo, hi = m[key]
            good = ok(mean)
            passed += good
            body.append(f'  <line class="track" x1="{x0}" y1="{y + 8}" x2="{x0 + col_w}" y2="{y + 8}"/>')
            if hi > lo:
                body.append(f'  <line class="range" x1="{position(lo, scale, x0, col_w):.1f}" y1="{y + 8}" '
                            f'x2="{position(hi, scale, x0, col_w):.1f}" y2="{y + 8}"/>')
            cx = position(mean, scale, x0, col_w)
            body.append(f'  <circle class="{"pass" if good else "fail"} ring" cx="{cx:.1f}" cy="{y + 8}" r="7"/>')
            mark, cls = ("✓", "passText") if good else ("✗", "failText")
            body.append(text(x0, y + 30, f"{mark} {fmt.format(mean)}", cls, 13))
        body.append(text(width - 12, y + 14, f"{passed}/4", "passText" if passed == 4 else "failText", 15, "end",
                         "600"))
    body.append(text(0, height - 8, "Cost axis is logarithmic. Dashed line = the good-enough threshold; "
                     "green ✓ meets it, red ✗ misses it.", "muted", 12))
    return svg(width, height, title, body)


# --- trade-off: accuracy (y) against cost (x, log), shaded = good enough on both --------------------------------

def tradeoff(models: list[tuple[str, dict, int]]) -> str:
    width, height = 880, 440
    x0, x1, y0, y1 = 90, 840, 380, 50  # plot box: x0..x1 left to right, y0 bottom, y1 top
    cost_scale, acc_lo, acc_hi = (0.1, 100.0, True), 0.90, 1.00

    def px(cost):
        return position(cost, cost_scale, x0, x1 - x0)

    def py(acc):
        return y0 - max(0.0, min(1.0, (acc - acc_lo) / (acc_hi - acc_lo))) * (y0 - y1)

    body = [f'  <rect class="zone" x="{x0}" y="{y1}" width="{px(1.0) - x0:.1f}" height="{py(0.95) - y1:.1f}"/>',
            text(x0 + 10, y1 + 20, "good enough", "passText", 12, weight="600"),
            text(x0 + 10, y1 + 36, "accuracy ≥ 0.95, ≤ $1 per 1,000", "passText", 12)]
    for cost in (0.1, 1, 10, 100):
        x = px(cost)
        body.append(f'  <line class="grid" x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y0}"/>')
        body.append(text(x, y0 + 20, f"${cost:g}", "muted", 12, "middle"))
    for acc in (0.90, 0.92, 0.94, 0.96, 0.98, 1.00):
        y = py(acc)
        body.append(f'  <line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        body.append(text(x0 - 10, y + 4, f"{acc:.2f}", "muted", 12, "end"))
    body += [
        f'  <line class="limit" x1="{px(1.0):.1f}" y1="{y1}" x2="{px(1.0):.1f}" y2="{y0}"/>',
        f'  <line class="limit" x1="{x0}" y1="{py(0.95):.1f}" x2="{x1}" y2="{py(0.95):.1f}"/>',
        text((x0 + x1) / 2, height - 12, "Cost per 1,000 answers (log scale; cheaper is left)", "ink2", 13, "middle"),
        f'  <text class="ink2" font-size="13" text-anchor="middle" transform="translate(22 {(y0 + y1) / 2}) '
        f'rotate(-90)">Answer accuracy (higher is better)</text>',
    ]
    points = []
    for i, (name, setup, runs) in enumerate(models):
        m = metrics(setup)
        mean, lo, hi = m["accuracy"]
        x, y = px(m["cost_1000"][0]), py(mean)
        if hi > lo:
            body.append(f'  <line class="range" x1="{x:.1f}" y1="{py(lo):.1f}" x2="{x:.1f}" y2="{py(hi):.1f}"/>')
        body.append(f'  <circle class="s{i + 1} ring" cx="{x:.1f}" cy="{y:.1f}" r="8"/>')
        detail = (f"{mean:.3f} · ${m['cost_1000'][0]:,.2f} · {m['latency_s'][0]:.1f} s · "
                  f"{runs} run{'s' if runs > 1 else ''}")
        points.append((x, y, name, detail))

    # Labels go beside their dot: right or left, above or below, whichever overlaps no other label or dot.
    label_w, label_h, placed = 215, 32, []
    dots = [(x - 9, y - 9, x + 9, y + 9) for x, y, _, _ in points]

    def overlaps(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    for x, y, name, detail in points:
        for dx, dy in ((14, -4), (14, 30), (-14, -4), (-14, 30), (14, -38), (-14, -38)):
            left = x + dx if dx > 0 else x + dx - label_w
            box = (left, y + dy - 14, left + label_w, y + dy - 14 + label_h)
            if box[2] <= x1 + 30 and box[0] >= x0 and not any(overlaps(box, b) for b in placed + dots):
                break
        placed.append(box)
        anchor = "start" if dx > 0 else "end"
        body.append(text(x + dx, y + dy, name, "ink", 13, anchor, "600"))
        body.append(text(x + dx, y + dy + 17, detail, "ink2", 12, anchor))
    return svg(width, height, "Answer accuracy vs cost per 1,000 answers, by answer model", body)


def main() -> None:
    data = json.loads(ANALYTICS.read_text(encoding="utf-8"))
    models = [(MODEL_NAMES.get(k, k), v, v["runs"]) for k, v in data["answer_models"].items()]
    pipeline = [(PIPELINE_NAMES[k], data["context_size"][k] if k in data["context_size"] else data["translation"][k],
                 (data["context_size"].get(k) or data["translation"].get(k))["runs"])
                for k in PIPELINE_NAMES if k in data["context_size"] or k in data["translation"]]
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "model_scorecard.svg": scorecard(models, "Answer models against the good-enough thresholds"),
        "model_tradeoff.svg": tradeoff(models),
        "pipeline_scorecard.svg": scorecard(pipeline, "Pipeline settings against the good-enough thresholds"),
    }
    for name, content in files.items():
        (OUT / name).write_text(content, encoding="utf-8")
        print(f"saved docs/images/{name}")


if __name__ == "__main__":
    main()
