"""
Fit ECI on an arbitrary benchmark subset, *consistently with the website*.

Why this exists
---------------
The website's domain ECIs (Math, SWE, ...) are computed in TypeScript
(`epoch-website-astro/legacy/vizs/benchmarks/eciSubsetMath.ts`) by a PROJECTION:

  - freeze each benchmark's difficulty + slope at the published full-fit values,
  - re-fit ONLY each model's 1-D capability on the selected benchmark subset,
  - keep ONE global scale (no per-domain re-anchoring).

`scripts/fit_baskets.py` also projects, but then calls `compute_eci_scores`
*per basket*, which re-derives the (a, b) scale from the anchor models'
capabilities WITHIN that basket -> per-basket re-anchoring. That pins the anchor
models to 130/150 inside every domain and makes the numbers NOT comparable to
the website's domain tabs (or to the general ECI).

This script reproduces the website's method instead, so a subset ECI it produces
is directly comparable to the general ECI and can be sanity-checked against the
site's Math/SWE tabs. It reuses the SAME `eci` package projection function the
website's production fit imports (`fit_capabilities_given_benchmarks`), fed with
the website's OWN published, frozen benchmark parameters.

Correctness details (the easy things to get wrong)
--------------------------------------------------
1. Inputs are the website's published artifacts (no Airtable access needed):
     - eci_benchmarks.csv : the exact per-(model, benchmark) frame the site fit used
     - edi_scores.csv     : frozen, ECI-scaled benchmark difficulty (`edi`) + slope
     - eci_scaling.csv    : the single global affine map (a, b)
2. The website projection is UNREGULARIZED. We pass regularization_strength=0;
   the package default (0.1) would drag capabilities toward 0.
3. `fit_capabilities_given_benchmarks` bounds capability to [-10, 10] (raw
   scale). So we project against RAW difficulty/slope recovered from the scaled
   published params via (a, b), then map the raw capability back onto the ECI
   scale with eci = a + b * capability. This is algebraically identical to
   projecting against the scaled params (the sigmoid argument is invariant), but
   keeps capability inside the function's bounds.

Usage (run via the project venv; do NOT use `uv run` here — it writes a uv.lock)
-------------------------------------------------------------------------------
  .venv/bin/python scripts/fit_subset.py                 # math+swe+knowledge+private, plus index.html
  .venv/bin/python scripts/fit_subset.py --subset private
  .venv/bin/python scripts/fit_subset.py --subset math
  .venv/bin/python scripts/fit_subset.py --subset all    # reassurance: reproduce the general ECI
  .venv/bin/python scripts/fit_subset.py --benchmarks "WeirdML,FrontierMath-2025-02-28-Private"

Open outputs/index.html to see every domain-vs-general comparison in one place.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eci.fitting import fit_capabilities_given_benchmarks

# Published website artifacts (the same files served at epoch.ai/data/).
BASE_URL = "https://epoch.ai/data"
ECI_BENCHMARKS_URL = f"{BASE_URL}/eci_benchmarks.csv"
EDI_SCORES_URL = f"{BASE_URL}/edi_scores.csv"
ECI_SCALING_URL = f"{BASE_URL}/eci_scaling.csv"
ECI_SCORES_URL = f"{BASE_URL}/eci_scores.csv"  # general ECI, for validation/comparison

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

# --- Named subsets -----------------------------------------------------------
# Source of truth for the private set is context/is-private_manual_list.md.
# (`is_private` is NOT currently in BENCHMARK_METADATA on this repo; keep the
# list here while the private-ECI work is experimental. If it graduates, move it
# to a single shared home rather than duplicating it.)
PRIVATE_BENCHMARKS = [
    "Chess Puzzles",
    "WeirdML",
    "FrontierMath-2025-02-28-Private",
    "FrontierMath-Tier-4-2025-07-01-Private",
    "ARC-AGI",
    "GeoBench",
    "Fiction.LiveBench",
    "SimpleBench",
    "DeepResearch Bench",
]

# Best-effort reconciliation of benchmarks.yml `domains` -> edi_scores.csv names.
# The all-benchmarks --validate check is the rigorous one; these domain lists
# may differ slightly from the live site's tab (alias / benchmark-version
# differences), so treat math/swe agreement as "should be close", not exact.
MATH_BENCHMARKS = [
    "MATH level 5",
    "OTIS Mock AIME 2024-2025",
    "GSM8K",
    "FrontierMath-2025-02-28-Private",
    "FrontierMath-Tier-4-2025-07-01-Private",
    "FrontierMath-Tiers-1-3-v2-Private",
    "FrontierMath-Tier-4-v2-Private",
]
SWE_BENCHMARKS = [
    "Aider polyglot",
    "SWE-Bench verified",
    "WeirdML",
    "Terminal Bench",
    "GSO-Bench",
    "Cybench",
    "PostTrainBench",
]
# Knowledge has NO website tab (it exists only as a fit_baskets.py basket), so
# the knowledge page is informational — there is no live site number to match.
KNOWLEDGE_BENCHMARKS = [
    "GPQA diamond",
    "MMLU",
    "ARC AI2",
    "OpenBookQA",
    "SimpleQA Verified",
    "ScienceQA",
    "TriviaQA",
]

NAMED_SUBSETS = {
    "all": None,  # every benchmark present in edi_scores.csv
    "math": MATH_BENCHMARKS,
    "swe": SWE_BENCHMARKS,
    "knowledge": KNOWLEDGE_BENCHMARKS,
    "private": PRIVATE_BENCHMARKS,
}

# Subsets generated by default (no --subset given). "all" is the reassurance
# check; the four domains are the comparisons against the site. Which have a
# live website counterpart to eyeball against:
DEFAULT_SUBSETS = ["math", "swe", "knowledge", "private", "uncontaminated"]
HAS_WEBSITE_TAB = {"math": True, "swe": True, "knowledge": False, "private": False,
                   "uncontaminated": False, "all": True}
# "uncontaminated" is special: not a fixed benchmark list but a per-row date mask.
SUBSET_CHOICES = list(NAMED_SUBSETS) + ["uncontaminated"]


def _esc(s) -> str:
    """Minimal HTML escape for model names in SVG/table text."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _load_csv(url: str, cache_name: str, use_cache: bool) -> pd.DataFrame:
    """Read a published CSV, caching to .cache/ so repeated runs are offline-friendly."""
    cache_path = CACHE_DIR / cache_name
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path)
    df = pd.read_csv(url)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df


def load_inputs(use_cache: bool = True):
    """Return (benchmarks_df, raw_bench_df, scaling) ready for projection.

    raw_bench_df has columns [benchmark, difficulty, discriminability] on the RAW
    IRT scale (recovered from the published, ECI-scaled edi/slope), as required by
    `fit_capabilities_given_benchmarks`.
    """
    benchmarks_df = _load_csv(ECI_BENCHMARKS_URL, "eci_benchmarks.csv", use_cache)
    edi_df = _load_csv(EDI_SCORES_URL, "edi_scores.csv", use_cache)
    scaling = _load_csv(ECI_SCALING_URL, "eci_scaling.csv", use_cache).iloc[0]

    a, b = float(scaling["a"]), float(scaling["b"])

    # Invert the global affine map to recover raw benchmark params:
    #   edi = a + b * difficulty            -> difficulty = (edi - a) / b
    #   slope_scaled = discriminability / b -> discriminability = slope_scaled * b
    raw_bench_df = pd.DataFrame(
        {
            "benchmark": edi_df["benchmark_name"],
            "difficulty": (edi_df["edi"] - a) / b,
            "discriminability": edi_df["estimated_slope_scaled"] * b,
        }
    )
    return benchmarks_df, raw_bench_df, (a, b)


def _fit_and_scale(obs_df, raw_bench_df, scaling, min_benchmarks, bootstrap_samples, msg):
    """Project the given observations onto the frozen params and return ECI-scaled scores.

    Shared core for both project_subset and project_uncontaminated. Returns a frame
    sorted by eci desc: Model, eci, eci_ci_low, eci_ci_high, n_benchmarks, date (if present).
    """
    a, b = scaling
    # Keep only observations whose benchmark has published frozen params.
    bench_names = set(raw_bench_df["benchmark"])
    df = obs_df[obs_df["benchmark"].isin(bench_names)].copy()
    counts = df.groupby("model_id")["benchmark"].nunique()
    keep = counts[counts >= min_benchmarks].index
    dropped = counts.size - len(keep)
    df = df[df["model_id"].isin(keep)]
    if df.empty:
        raise ValueError(f"No model has >= {min_benchmarks} benchmarks in the {msg} set.")
    bench_df = raw_bench_df[raw_bench_df["benchmark"].isin(set(df["benchmark"]))].copy()

    print(f"  {msg}: {df['benchmark'].nunique()} benchmark(s) x {df['model_id'].nunique()} model(s) "
          f"({dropped} models dropped below min-benchmarks={min_benchmarks})...")

    # The website projection is UNREGULARIZED -> regularization_strength=0.
    model_df = fit_capabilities_given_benchmarks(
        df, bench_df, regularization_strength=0.0,
        bootstrap_samples=bootstrap_samples, bootstrap_seed=12345,
    )
    # Map raw capability back onto the global ECI scale (b > 0 preserves order).
    model_df["eci"] = a + b * model_df["capability"]
    model_df["eci_ci_low"] = a + b * model_df["capability_ci_low"]
    model_df["eci_ci_high"] = a + b * model_df["capability_ci_high"]
    n_by_model = df.groupby("model_id")["benchmark"].nunique()
    model_df["n_benchmarks"] = model_df["model_id"].map(n_by_model)

    cols = ["Model", "eci", "eci_ci_low", "eci_ci_high", "n_benchmarks"]
    if "date" in model_df.columns:
        cols.append("date")
    return model_df[cols].sort_values("eci", ascending=False).reset_index(drop=True)


def project_subset(benchmarks_df, raw_bench_df, scaling, subset, min_benchmarks=3, bootstrap_samples=0):
    """Project models onto a fixed benchmark `subset` (None = all benchmarks)."""
    available = set(raw_bench_df["benchmark"])
    if subset is None:
        subset_names = sorted(available)
    else:
        subset_names = [bm for bm in subset if bm in available]
        missing = [bm for bm in subset if bm not in available]
        if missing:
            print(f"  NOTE: {len(missing)} requested benchmark(s) absent from edi_scores.csv: {missing}")
    if not subset_names:
        raise ValueError("None of the requested benchmarks are available.")
    obs = benchmarks_df[benchmarks_df["benchmark"].isin(subset_names)].copy()
    return _fit_and_scale(obs, raw_bench_df, scaling, min_benchmarks, bootstrap_samples,
                          f"{len(subset_names)}-benchmark subset")


def uncontaminated_observations(benchmarks_df, private_set):
    """Per-(model, benchmark) mask: keep a score iff the benchmark is private OR it was
    released AFTER the model (so the model could not have trained on it).

    Uses only `benchmark_release_date` and the model `date` -- both already in
    eci_benchmarks.csv, so this needs no external data. Unlike a fixed subset this is a
    ROW filter: a given benchmark counts for late models but is dropped for early ones.
    Rows with a missing date (and not private) are dropped -- can't prove uncontaminated.
    """
    df = benchmarks_df.copy()
    bench_date = pd.to_datetime(df["benchmark_release_date"], errors="coerce")
    model_date = pd.to_datetime(df["date"], errors="coerce")
    is_private = df["benchmark"].isin(set(private_set))
    released_after_model = bench_date > model_date  # NaN comparisons -> False (conservative)
    kept = df[is_private | released_after_model].copy()
    n_priv = int(is_private.sum())
    n_after = int((released_after_model & ~is_private).sum())
    print(f"  uncontaminated mask: kept {len(kept)}/{len(df)} obs "
          f"({n_priv} private + {n_after} released-after-model)")
    return kept


def project_uncontaminated(benchmarks_df, raw_bench_df, scaling, private_set,
                           min_benchmarks=3, bootstrap_samples=0):
    """Project each model on only the scores it could not have been contaminated by."""
    obs = uncontaminated_observations(benchmarks_df, private_set)
    return _fit_and_scale(obs, raw_bench_df, scaling, min_benchmarks, bootstrap_samples,
                          "uncontaminated")


def load_general_eci(use_cache: bool = True) -> pd.DataFrame:
    """Published general ECI, keyed by Model (= model_group), for compare/validate."""
    g = _load_csv(ECI_SCORES_URL, "eci_scores.csv", use_cache)
    return g[["Model", "eci"]].rename(columns={"eci": "general_eci"})


def add_general_comparison(result: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """Attach general ECI and the domain-minus-general delta (the interesting quantity)."""
    merged = result.merge(load_general_eci(use_cache), on="Model", how="left")
    merged["delta_vs_general"] = merged["eci"] - merged["general_eci"]
    return merged


def _svg_scatter(rows: list[dict], label: str, sat_threshold: float = 30.0) -> str:
    """Inline SVG: general ECI (x) vs subset ECI (y), with the y=x line.

    Points on the diagonal agree; points below it score lower on the subset than
    their general ECI (for a private subset, a sign of benchmaxxing on public
    benchmarks). Saturated outliers (|delta| > sat_threshold, e.g. a model that
    aces every subset benchmark and hits the capability bound) are excluded from
    the plot so they don't wreck the scale; they remain in the table.
    """
    pts = [r for r in rows if r["general_eci"] == r["general_eci"]  # not NaN
           and abs(r["delta"]) <= sat_threshold]
    excluded = len(rows) - len(pts)
    if not pts:
        return "<p><em>No plottable points.</em></p>"

    vals = [r["general_eci"] for r in pts] + [r["subset_eci"] for r in pts]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.06 or 1.0
    lo, hi = lo - pad, hi + pad
    M, inner = 70, 460
    span = hi - lo

    def px(v): return M + (v - lo) / span * inner
    def py(v): return M + inner - (v - lo) / span * inner

    parts = [f'<svg width="{M*2+inner}" height="{M*2+inner}" '
             f'style="font-family:system-ui,sans-serif;font-size:12px;background:#fff">']
    # axes box + y=x diagonal
    parts.append(f'<rect x="{M}" y="{M}" width="{inner}" height="{inner}" fill="none" stroke="#ccc"/>')
    parts.append(f'<line x1="{px(lo)}" y1="{py(lo)}" x2="{px(hi)}" y2="{py(hi)}" '
                 f'stroke="#999" stroke-dasharray="4 4"/>')
    # gridline ticks every 10 ECI points
    t = int(lo // 10 * 10)
    while t <= hi:
        if t >= lo:
            parts.append(f'<line x1="{px(t)}" y1="{M}" x2="{px(t)}" y2="{M+inner}" stroke="#f0f0f0"/>')
            parts.append(f'<line x1="{M}" y1="{py(t)}" x2="{M+inner}" y2="{py(t)}" stroke="#f0f0f0"/>')
            parts.append(f'<text x="{px(t)}" y="{M+inner+16}" text-anchor="middle" fill="#888">{t}</text>')
            parts.append(f'<text x="{M-10}" y="{py(t)+4}" text-anchor="end" fill="#888">{t}</text>')
        t += 10
    # points (red = below diagonal / worse on subset; green = above)
    labelled = sorted(pts, key=lambda r: -abs(r["delta"]))[:6]
    for r in pts:
        color = "#d9534f" if r["delta"] < 0 else "#41a35d"
        tip = (f'{r["Model"]}  —  general {r["general_eci"]:.2f} / {label} {r["subset_eci"]:.2f}'
               f'  —  Δ {r["delta"]:+.2f}  (n={r["n_benchmarks"]})')
        data_tip = _esc(tip).replace('"', "&quot;")
        parts.append(f'<circle class="pt" data-tip="{data_tip}" '
                     f'cx="{px(r["general_eci"]):.1f}" cy="{py(r["subset_eci"]):.1f}" '
                     f'r="5" fill="{color}" fill-opacity="0.55" stroke="{color}" '
                     f'style="cursor:pointer"/>')
    for r in labelled:
        parts.append(f'<text x="{px(r["general_eci"])+6:.1f}" y="{py(r["subset_eci"])+3:.1f}" '
                     f'fill="#333">{_esc(r["Model"])} ({r["delta"]:+.1f})</text>')
    parts.append(f'<text x="{M+inner/2}" y="{M+inner+40}" text-anchor="middle" '
                 f'fill="#555">General ECI (from website)</text>')
    parts.append(f'<text x="20" y="{M+inner/2}" text-anchor="middle" fill="#555" '
                 f'transform="rotate(-90 20 {M+inner/2})">{label}-subset ECI (computed)</text>')
    if excluded:
        parts.append(f'<text x="{M}" y="{M-16}" fill="#b3801f">'
                     f'{excluded} saturated model(s) excluded from plot (see table)</text>')
    parts.append("</svg>")
    return "".join(parts)


def write_comparison_html(rows: list[dict], label: str, path: Path) -> None:
    """One self-contained page: scatter + a full sorted table. No dependencies."""
    deltas = [r["delta"] for r in rows if r["delta"] == r["delta"]]
    med = sorted(abs(d) for d in deltas)[len(deltas)//2] if deltas else float("nan")
    rows_sorted = sorted(rows, key=lambda r: (r["delta"] != r["delta"], -abs(r["delta"])))

    body = [f"<tr><th>#</th><th>Model</th><th>General ECI<br>(website)</th>"
            f"<th>{label} ECI<br>(computed)</th><th>&Delta; (subset&minus;general)</th>"
            f"<th>n&nbsp;benchmarks</th></tr>"]
    for i, r in enumerate(rows_sorted, 1):
        d = r["delta"]
        bg = "#fff"
        if d == d:
            shade = min(abs(d) / 6.0, 1.0)
            bg = (f"rgba(217,83,79,{0.10+0.45*shade:.2f})" if d < 0
                  else f"rgba(65,163,93,{0.10+0.45*shade:.2f})")
        gen = "" if r["general_eci"] != r["general_eci"] else f'{r["general_eci"]:.2f}'
        dly = "" if d != d else f"{d:+.2f}"
        body.append(
            f'<tr><td>{i}</td><td style="text-align:left">{_esc(r["Model"])}</td>'
            f'<td>{gen}</td><td>{r["subset_eci"]:.2f}</td>'
            f'<td style="background:{bg}">{dly}</td><td>{r["n_benchmarks"]}</td></tr>')

    html = f"""<!doctype html><meta charset="utf-8">
<title>{label} subset ECI vs general</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:30px;color:#222;max-width:900px}}
 table{{border-collapse:collapse;margin-top:16px;font-size:13px}}
 td,th{{border:1px solid #e3e3e3;padding:4px 8px;text-align:right}}
 th{{background:#f7f7f7}} caption{{text-align:left}}
 .note{{color:#555;font-size:13px;line-height:1.5}}
 #tip{{position:fixed;display:none;pointer-events:none;background:#222;color:#fff;
       padding:5px 9px;border-radius:5px;font-size:12px;white-space:nowrap;z-index:50;
       box-shadow:0 2px 8px rgba(0,0,0,.25)}}
 circle.pt:hover{{fill-opacity:.95;r:7}}
</style>
<div id="tip"></div>
<h2>{label}-subset ECI vs general ECI &mdash; {len(rows)} models</h2>
<p class="note">
 <b>General ECI</b> is read verbatim from the website's <code>eci_scores.csv</code> (the exact number on the site).<br>
 <b>{label} ECI</b> is the only computed value: a projection onto the {label} benchmark subset using the
 website's frozen <code>edi_scores.csv</code> params (same method the site's domain tabs use).<br>
 <b>&Delta; &lt; 0</b> (red) = the model scores <i>lower</i> on the subset than its overall ECI implies.
 For the <code>private</code>/<code>uncontaminated</code> subsets that is the benchmaxxing signal.
 Median |&Delta;| = {med:.2f} ECI points. <i>Hover a point for the model.</i>
</p>
{_svg_scatter(rows, label)}
<table><caption class="note">Sorted by |&Delta;| (biggest movers first).</caption>
{''.join(body)}
</table>
<script>
(function(){{
  var tip=document.getElementById('tip');
  document.querySelectorAll('circle.pt').forEach(function(c){{
    c.addEventListener('mousemove',function(e){{
      tip.textContent=c.getAttribute('data-tip');
      tip.style.display='block';
      tip.style.left=(e.clientX+14)+'px';
      tip.style.top=(e.clientY+14)+'px';
    }});
    c.addEventListener('mouseleave',function(){{tip.style.display='none';}});
  }});
}})();
</script>
"""
    path.write_text(html)


def build_rows(result: pd.DataFrame, use_cache: bool) -> list[dict]:
    """Merge computed subset ECI with website general ECI -> list of row dicts."""
    g = load_general_eci(use_cache)
    merged = result.merge(g, on="Model", how="left")
    merged["delta"] = merged["eci"] - merged["general_eci"]
    return [
        {"Model": r["Model"], "subset_eci": r["eci"], "general_eci": r["general_eci"],
         "delta": r["delta"], "n_benchmarks": int(r["n_benchmarks"])}
        for _, r in merged.iterrows()
    ]


def run_subset(label, subset, benchmarks_df, raw_bench_df, scaling, out_dir, args):
    """Compute one subset-vs-general comparison; write its HTML + CSV; return a summary dict."""
    # "all" recomputes the general ECI to show it reproduces the website's file
    # (reassurance only; in real use you read the general ECI, never recompute it).
    min_bm = 1 if label == "all" else args.min_benchmarks
    if label == "uncontaminated":
        result = project_uncontaminated(benchmarks_df, raw_bench_df, scaling, PRIVATE_BENCHMARKS,
                                        min_benchmarks=min_bm, bootstrap_samples=args.bootstrap_samples)
    else:
        result = project_subset(benchmarks_df, raw_bench_df, scaling, subset,
                                min_benchmarks=min_bm, bootstrap_samples=args.bootstrap_samples)
    rows = build_rows(result, use_cache=not args.no_cache)

    write_comparison_html(rows, label, out_dir / f"{label}_vs_general.html")
    pd.DataFrame(rows).to_csv(out_dir / f"{label}_vs_general.csv", index=False)

    deltas = [abs(r["delta"]) for r in rows if r["delta"] == r["delta"]]
    return {
        "label": label,
        "n_models": len(rows),
        "median_abs_delta": (sorted(deltas)[len(deltas)//2] if deltas else float("nan")),
        "max_abs_delta": (max(deltas) if deltas else float("nan")),
        "has_tab": HAS_WEBSITE_TAB.get(label, False),
    }


def write_index_html(summaries: list[dict], out_dir: Path) -> None:
    """Landing page linking each subset comparison, with at-a-glance spread stats."""
    rows = ["<tr><th>Subset</th><th>Models</th><th>Median |&Delta;|</th>"
            "<th>Max |&Delta;|</th><th>Website tab to compare?</th><th></th></tr>"]
    for s in summaries:
        tab = ("yes — eyeball vs the site's domain explorer" if s["has_tab"]
               else "no live tab (informational)")
        rows.append(
            f'<tr><td style="text-align:left"><b>{s["label"]}</b></td>'
            f'<td>{s["n_models"]}</td><td>{s["median_abs_delta"]:.2f}</td>'
            f'<td>{s["max_abs_delta"]:.2f}</td><td style="text-align:left">{tab}</td>'
            f'<td><a href="{s["label"]}_vs_general.html">open &rarr;</a></td></tr>')
    html = f"""<!doctype html><meta charset="utf-8"><title>Subset ECI vs general</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:30px;color:#222;max-width:820px}}
 table{{border-collapse:collapse;margin-top:14px;font-size:14px}}
 td,th{{border:1px solid #e3e3e3;padding:6px 10px;text-align:right}} th{{background:#f7f7f7}}
 .note{{color:#555;line-height:1.5}}
</style>
<h2>Subset ECI vs general ECI</h2>
<p class="note">Each page projects models onto a benchmark subset (the website's own
frozen params + method) and compares to the general ECI read verbatim from
<code>eci_scores.csv</code>. Hover any point to see the model. Bigger median |&Delta;|
= the subset disagrees more with the overall ECI.</p>
<table>{''.join(rows)}</table>
"""
    (out_dir / "index.html").write_text(html)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--subset", choices=SUBSET_CHOICES,
                   help="Run one named subset (incl. 'uncontaminated'). Omit to run all + index.html.")
    g.add_argument("--benchmarks", help="Comma-separated explicit benchmark names (edi_scores.csv naming).")
    p.add_argument("--min-benchmarks", type=int, default=3,
                   help="Min distinct subset benchmarks per model (site uses 2). Default 3.")
    p.add_argument("--bootstrap-samples", type=int, default=0,
                   help="Bootstrap resamples for CIs (0 = skip; note: site CI method differs).")
    p.add_argument("--no-cache", action="store_true", help="Always re-download published CSVs.")
    args = p.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_df, raw_bench_df, scaling = load_inputs(use_cache=not args.no_cache)

    if args.benchmarks:
        jobs = [("custom", [b.strip() for b in args.benchmarks.split(",") if b.strip()])]
    elif args.subset:
        jobs = [(args.subset, NAMED_SUBSETS.get(args.subset))]
    else:
        jobs = [(lbl, NAMED_SUBSETS.get(lbl)) for lbl in DEFAULT_SUBSETS]

    summaries = []
    for label, subset in jobs:
        print(f"\n[{label}]")
        summaries.append(run_subset(label, subset, benchmarks_df, raw_bench_df, scaling, out_dir, args))

    print(f"\n  {'subset':10} {'models':>7} {'median|Δ|':>10} {'max|Δ|':>8}  website-tab")
    for s in summaries:
        print(f"  {s['label']:10} {s['n_models']:7d} {s['median_abs_delta']:10.2f} "
              f"{s['max_abs_delta']:8.2f}  {'yes' if s['has_tab'] else 'no'}")

    if len(summaries) > 1:
        write_index_html(summaries, out_dir)
        print(f"\n  Open the overview:  open {out_dir / 'index.html'}")
    else:
        print(f"\n  Open the visual:    open {out_dir / (summaries[0]['label'] + '_vs_general.html')}")


if __name__ == "__main__":
    main()
