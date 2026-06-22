"""
Subset / uncontaminated ECI vs general ECI, computed with the WEBSITE'S OWN code.

How the fit matches the website (no ambiguity)
----------------------------------------------
The website's domain-ECI tabs (Math, SWE, ...) compute each model's subset ECI in
`epoch-website-astro/legacy/vizs/benchmarks/eciSubsetMath.ts` via `fitModelECI`: a
per-model 1-D fit on the ECI scale (bounds [-100, 300]) with an analytical starting
guess + Gauss-Newton + Brent fallback. This script does NOT re-implement that. It
calls the **real, unmodified `fitModelECI`** from that .ts file via a tiny Node
harness (`scripts/website_fit.mjs`), so the fit logic is byte-for-byte the website's.

Division of labour:
  - Python (here): download the website's published CSVs, build each model's
    observations exactly as the site's `computeResults` does
    (obs = {perf: clamp01(score), edi, slope}; clamp01 = max(.001, min(.999, p))),
    apply the subset / uncontaminated selection, then render the comparison HTML.
  - Node (`website_fit.mjs`): import the real `fitModelECI` and run it per model.

Inputs are the website's published artifacts (no Airtable, no recompute):
  - eci_benchmarks.csv : per-(model, benchmark) performance the site fit used
  - edi_scores.csv     : frozen ECI-scaled benchmark `edi` + `estimated_slope_scaled`
  - eci_scores.csv     : the published GENERAL ECI, read verbatim (never recomputed)

Requirements: Node (>=22, for native TS) and a local checkout of epoch-website-astro
as a sibling of this repo (or set WEBSITE_ECI_TS to the eciSubsetMath.ts path).

Usage (run via the project venv; do NOT use `uv run` here -- it writes a uv.lock)
--------------------------------------------------------------------------------
  .venv/bin/python scripts/fit_subset.py                 # math+swe+knowledge+private+uncontaminated, + index.html
  .venv/bin/python scripts/fit_subset.py --subset private
  .venv/bin/python scripts/fit_subset.py --subset all    # reproduce the general ECI (sanity check)
  .venv/bin/python scripts/fit_subset.py --benchmarks "WeirdML,FrontierMath-2025-02-28-Private"

Open outputs/index.html to see every domain-vs-general comparison in one place.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

# Published website artifacts (the same files served at epoch.ai/data/).
BASE_URL = "https://epoch.ai/data"
ECI_BENCHMARKS_URL = f"{BASE_URL}/eci_benchmarks.csv"
EDI_SCORES_URL = f"{BASE_URL}/edi_scores.csv"
ECI_SCORES_URL = f"{BASE_URL}/eci_scores.csv"  # the published general ECI

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE.parent / ".cache"

# --- Named subsets -----------------------------------------------------------
# Source of truth for the private set is context/is-private_manual_list.md.
# (`is_private` is NOT currently in BENCHMARK_METADATA on this repo; keep the
# list here while the private-ECI work is experimental.)
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
# The all-benchmarks --subset all check is the rigorous one; these domain lists
# may differ slightly from the live site's tab (alias / benchmark-version diffs).
MATH_BENCHMARKS = [
    "MATH level 5", "OTIS Mock AIME 2024-2025", "GSM8K",
    "FrontierMath-2025-02-28-Private", "FrontierMath-Tier-4-2025-07-01-Private",
    "FrontierMath-Tiers-1-3-v2-Private", "FrontierMath-Tier-4-v2-Private",
]
SWE_BENCHMARKS = [
    "Aider polyglot", "SWE-Bench verified", "WeirdML", "Terminal Bench",
    "GSO-Bench", "Cybench", "PostTrainBench",
]
# Knowledge has NO website tab (it exists only as a fit_baskets.py basket).
KNOWLEDGE_BENCHMARKS = [
    "GPQA diamond", "MMLU", "ARC AI2", "OpenBookQA",
    "SimpleQA Verified", "ScienceQA", "TriviaQA",
]
NAMED_SUBSETS = {
    "all": None,  # every benchmark present in edi_scores.csv
    "math": MATH_BENCHMARKS,
    "swe": SWE_BENCHMARKS,
    "knowledge": KNOWLEDGE_BENCHMARKS,
    "private": PRIVATE_BENCHMARKS,
}
DEFAULT_SUBSETS = ["math", "swe", "knowledge", "private", "uncontaminated"]
HAS_WEBSITE_TAB = {"math": True, "swe": True, "knowledge": False, "private": False,
                   "uncontaminated": False, "all": True}
SUBSET_CHOICES = list(NAMED_SUBSETS) + ["uncontaminated"]


# --- data loading ------------------------------------------------------------

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


def load_obs_source(use_cache: bool = True) -> pd.DataFrame:
    """Per-(model, benchmark) frame joined with the frozen ECI-scaled params.

    Columns: Model, benchmark, performance, edi, slope, benchmark_release_date, date.
    Inner join on benchmark drops benchmarks with no published params (the website's
    `bmMap[bmName]` check does the same).
    """
    bench = _load_csv(ECI_BENCHMARKS_URL, "eci_benchmarks.csv", use_cache)
    edi = _load_csv(EDI_SCORES_URL, "edi_scores.csv", use_cache).rename(
        columns={"benchmark_name": "benchmark", "estimated_slope_scaled": "slope"})
    obs = bench.merge(edi[["benchmark", "edi", "slope"]], on="benchmark", how="inner")
    obs = obs.dropna(subset=["performance", "edi", "slope"])
    keep = ["Model", "benchmark", "performance", "edi", "slope", "benchmark_release_date", "date"]
    return obs[keep].copy()


def load_general_eci(use_cache: bool = True) -> dict:
    """Published general ECI as {Model: eci} -- read verbatim, never recomputed."""
    g = _load_csv(ECI_SCORES_URL, "eci_scores.csv", use_cache)
    return dict(zip(g["Model"], g["eci"]))


def _clamp01(p: float) -> float:
    """Verbatim port of the website's clamp01 (eciSubsetMath.ts)."""
    return max(0.001, min(0.999, p))


# --- observation building (mirrors computeResults) ---------------------------

def select_rows(obs_source: pd.DataFrame, label: str, subset) -> pd.DataFrame:
    """Apply the subset filter, or the per-(model,benchmark) 'uncontaminated' date mask."""
    if label == "uncontaminated":
        bd = pd.to_datetime(obs_source["benchmark_release_date"], errors="coerce")
        md = pd.to_datetime(obs_source["date"], errors="coerce")
        is_private = obs_source["benchmark"].isin(set(PRIVATE_BENCHMARKS))
        released_after = bd > md  # NaN comparisons -> False (conservative)
        kept = obs_source[is_private | released_after]
        print(f"  uncontaminated mask: kept {len(kept)}/{len(obs_source)} obs "
              f"({int(is_private.sum())} private + {int((released_after & ~is_private).sum())} released-after-model)")
        return kept
    if subset is None:
        return obs_source
    return obs_source[obs_source["benchmark"].isin(set(subset))]


def build_items(obs_source, label, subset, min_benchmarks):
    """Return (items, meta): per-model observations for the Node fit, keyed by id."""
    df = select_rows(obs_source, label, subset)
    items, meta = [], {}
    for model, g in df.groupby("Model"):
        if len(g) < min_benchmarks:
            continue
        obs = [[_clamp01(p), float(e), float(s)]
               for p, e, s in zip(g["performance"], g["edi"], g["slope"])]
        items.append({"id": f"{label}|||{model}", "obs": obs})
        meta[model] = len(obs)
    print(f"  [{label}] {df['benchmark'].nunique()} benchmark(s) -> {len(items)} model(s) "
          f"(min-benchmarks={min_benchmarks})")
    return items, meta


# --- the website fit (Node) --------------------------------------------------

def run_website_fit(all_items, out_dir, website_ts=None) -> dict:
    """Run the REAL website fitModelECI over all items via the Node harness."""
    in_path, out_path = out_dir / "_fit_in.json", out_dir / "_fit_out.json"
    in_path.write_text(json.dumps({"items": all_items}))
    env = dict(os.environ)
    if website_ts:
        env["WEBSITE_ECI_TS"] = str(website_ts)
    cmd = ["node", "--import", str(HERE / "_website_fit_register.mjs"),
           str(HERE / "website_fit.mjs"), str(in_path), str(out_path)]
    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError:
        raise SystemExit("ERROR: `node` not found. Install Node >=22 (native TypeScript).")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"ERROR: website fit failed (exit {e.returncode}). "
                         f"Is epoch-website-astro a sibling repo, or WEBSITE_ECI_TS set?")
    return json.loads(out_path.read_text())


def build_rows(label, meta, fit, general):
    """Combine the website fit with the published general ECI -> row dicts."""
    rows = []
    for model, n in meta.items():
        eci = fit.get(f"{label}|||{model}")
        if eci is None:  # fitModelECI returned null (e.g. degenerate obs)
            continue
        gen = general.get(model, float("nan"))
        rows.append({"Model": model, "subset_eci": eci, "general_eci": gen,
                     "delta": eci - gen, "n_benchmarks": n})
    return rows


# --- presentation ------------------------------------------------------------

def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg_scatter(rows, label, sat_threshold=60.0):
    """Inline SVG: general ECI (x) vs subset ECI (y), with the y=x line. Hover for model."""
    pts = [r for r in rows if r["general_eci"] == r["general_eci"]
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
    parts.append(f'<rect x="{M}" y="{M}" width="{inner}" height="{inner}" fill="none" stroke="#ccc"/>')
    parts.append(f'<line x1="{px(lo)}" y1="{py(lo)}" x2="{px(hi)}" y2="{py(hi)}" '
                 f'stroke="#999" stroke-dasharray="4 4"/>')
    t = int(lo // 10 * 10)
    while t <= hi:
        if t >= lo:
            parts.append(f'<line x1="{px(t)}" y1="{M}" x2="{px(t)}" y2="{M+inner}" stroke="#f0f0f0"/>')
            parts.append(f'<line x1="{M}" y1="{py(t)}" x2="{M+inner}" y2="{py(t)}" stroke="#f0f0f0"/>')
            parts.append(f'<text x="{px(t)}" y="{M+inner+16}" text-anchor="middle" fill="#888">{t}</text>')
            parts.append(f'<text x="{M-10}" y="{py(t)+4}" text-anchor="end" fill="#888">{t}</text>')
        t += 10
    labelled = sorted(pts, key=lambda r: -abs(r["delta"]))[:6]
    for r in pts:
        color = "#d9534f" if r["delta"] < 0 else "#41a35d"
        tip = (f'{r["Model"]}  —  general {r["general_eci"]:.2f} / {label} {r["subset_eci"]:.2f}'
               f'  —  Δ {r["delta"]:+.2f}  (n={r["n_benchmarks"]})')
        parts.append(f'<circle class="pt" data-tip="{_esc(tip).replace(chr(34), "&quot;")}" '
                     f'cx="{px(r["general_eci"]):.1f}" cy="{py(r["subset_eci"]):.1f}" '
                     f'r="5" fill="{color}" fill-opacity="0.55" stroke="{color}" style="cursor:pointer"/>')
    for r in labelled:
        parts.append(f'<text x="{px(r["general_eci"])+6:.1f}" y="{py(r["subset_eci"])+3:.1f}" '
                     f'fill="#333">{_esc(r["Model"])} ({r["delta"]:+.1f})</text>')
    parts.append(f'<text x="{M+inner/2}" y="{M+inner+40}" text-anchor="middle" '
                 f'fill="#555">General ECI (from website)</text>')
    parts.append(f'<text x="20" y="{M+inner/2}" text-anchor="middle" fill="#555" '
                 f'transform="rotate(-90 20 {M+inner/2})">{label}-subset ECI (website fit)</text>')
    if excluded:
        parts.append(f'<text x="{M}" y="{M-16}" fill="#b3801f">'
                     f'{excluded} model(s) beyond ±{sat_threshold:.0f} excluded from plot (see table)</text>')
    parts.append("</svg>")
    return "".join(parts)


def write_comparison_html(rows, label, path):
    deltas = [r["delta"] for r in rows if r["delta"] == r["delta"]]
    med = sorted(abs(d) for d in deltas)[len(deltas)//2] if deltas else float("nan")
    rows_sorted = sorted(rows, key=lambda r: (r["delta"] != r["delta"], -abs(r["delta"])))
    body = [f"<tr><th>#</th><th>Model</th><th>General ECI<br>(website)</th>"
            f"<th>{label} ECI<br>(website fit)</th><th>&Delta; (subset&minus;general)</th>"
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
 <b>{label} ECI</b> is computed by the website's own <code>fitModelECI</code> (run unmodified via Node) on the
 frozen <code>edi_scores.csv</code> params &mdash; identical logic to the site's domain tabs.<br>
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


def write_index_html(summaries, out_dir):
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
<p class="note">Each page is computed by the website's own <code>fitModelECI</code> (run unmodified via Node)
and compared to the general ECI read verbatim from <code>eci_scores.csv</code>. Hover any point to see the model.
Bigger median |&Delta;| = the subset disagrees more with the overall ECI.</p>
<table>{''.join(rows)}</table>
"""
    (out_dir / "index.html").write_text(html)


# --- orchestration -----------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--subset", choices=SUBSET_CHOICES,
                   help="Run one named subset (incl. 'uncontaminated'). Omit to run all + index.html.")
    g.add_argument("--benchmarks", help="Comma-separated explicit benchmark names (edi_scores.csv naming).")
    p.add_argument("--min-benchmarks", type=int, default=2,
                   help="Min benchmarks per model (default 2, matching the site's subset explorer).")
    p.add_argument("--website-ts", help="Path to eciSubsetMath.ts (else sibling epoch-website-astro / $WEBSITE_ECI_TS).")
    p.add_argument("--no-cache", action="store_true", help="Always re-download published CSVs.")
    args = p.parse_args()

    out_dir = HERE.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_source = load_obs_source(use_cache=not args.no_cache)
    general = load_general_eci(use_cache=not args.no_cache)

    if args.benchmarks:
        jobs = [("custom", [b.strip() for b in args.benchmarks.split(",") if b.strip()])]
    elif args.subset:
        jobs = [(args.subset, NAMED_SUBSETS.get(args.subset))]
    else:
        jobs = [(lbl, NAMED_SUBSETS.get(lbl)) for lbl in DEFAULT_SUBSETS]

    all_items, metas = [], {}
    for label, subset in jobs:
        items, meta = build_items(obs_source, label, subset, args.min_benchmarks)
        all_items += items
        metas[label] = meta

    fit = run_website_fit(all_items, out_dir, website_ts=args.website_ts)

    summaries = []
    for label, _ in jobs:
        rows = build_rows(label, metas[label], fit, general)
        write_comparison_html(rows, label, out_dir / f"{label}_vs_general.html")
        pd.DataFrame(rows).to_csv(out_dir / f"{label}_vs_general.csv", index=False)
        deltas = [abs(r["delta"]) for r in rows if r["delta"] == r["delta"]]
        summaries.append({
            "label": label, "n_models": len(rows),
            "median_abs_delta": (sorted(deltas)[len(deltas)//2] if deltas else float("nan")),
            "max_abs_delta": (max(deltas) if deltas else float("nan")),
            "has_tab": HAS_WEBSITE_TAB.get(label, False),
        })

    print(f"\n  {'subset':14} {'models':>7} {'median|Δ|':>10} {'max|Δ|':>8}  website-tab")
    for s in summaries:
        print(f"  {s['label']:14} {s['n_models']:7d} {s['median_abs_delta']:10.2f} "
              f"{s['max_abs_delta']:8.2f}  {'yes' if s['has_tab'] else 'no'}")

    if len(summaries) > 1:
        write_index_html(summaries, out_dir)
        print(f"\n  Open the overview:  open {out_dir / 'index.html'}")
    else:
        print(f"\n  Open the visual:    open {out_dir / (summaries[0]['label'] + '_vs_general.html')}")


if __name__ == "__main__":
    main()
