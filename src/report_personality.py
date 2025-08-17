# src/report_personality.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List
import math
import numpy as np
import pandas as pd

from load_data import load_config  # reuse your config loader

# ---------- Paths ----------
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ---------- IO ----------
def _load_personality(cfg: Dict[str, Any]) -> pd.DataFrame:
    # path from YAML, else default
    pth = (cfg.get("paths", {}) or {}).get("personality", "outputs/calibration/personality.csv")
    f = (_project_root() / pth).resolve()
    if not f.exists():
        raise FileNotFoundError(f"[report_personality] personality file not found at: {f}")

    df = pd.read_csv(f)
    # expected columns: driver,aggression,aggression_se,defence,defence_se,risk,risk_se,...
    need = ["driver","aggression","aggression_se","defence","defence_se","risk","risk_se"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"[report_personality] missing column '{c}' in {f}")
    df["driver"] = df["driver"].astype(str)
    for c in ["aggression","aggression_se","defence","defence_se","risk","risk_se"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------- Helpers ----------
def _ci95(mean: pd.Series, se: pd.Series) -> Tuple[pd.Series, pd.Series]:
    lo = (mean - 1.96 * se).clip(lower=0.0, upper=1.0)
    hi = (mean + 1.96 * se).clip(lower=0.0, upper=1.0)
    return lo, hi

def _fmt_row(v: float, se: float) -> str:
    if not np.isfinite(v): return "—"
    if not np.isfinite(se): return f"{v:.3f}"
    return f"{v:.3f} ± {se:.3f}"

def _rankings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Higher is "more" for aggression/defence/risk
    out["rank_aggression"] = out["aggression"].rank(method="min", ascending=False)
    out["rank_defence"]   = out["defence"].rank(method="min",   ascending=False)
    out["rank_risk"]      = out["risk"].rank(method="min",      ascending=False)

    # Also compute "safest" rank (lower risk is better)
    out["rank_safest"]    = out["risk"].rank(method="min",      ascending=True)

    # 95% CI bands
    out["aggr_lo"], out["aggr_hi"] = _ci95(out["aggression"], out["aggression_se"])
    out["def_lo"],  out["def_hi"]  = _ci95(out["defence"],    out["defence_se"])
    out["risk_lo"], out["risk_hi"] = _ci95(out["risk"],       out["risk_se"])

    # Pretty columns
    out["aggression_str"] = [_fmt_row(v, s) for v, s in zip(out["aggression"], out["aggression_se"])]
    out["defence_str"]    = [_fmt_row(v, s) for v, s in zip(out["defence"], out["defence_se"])]
    out["risk_str"]       = [_fmt_row(v, s) for v, s in zip(out["risk"], out["risk_se"])]

    return out

def _print_table(df: pd.DataFrame, cols: List[str], title: str, top: int = 10) -> None:
    print(f"\n=== {title} (top {top}) ===")
    show = df[cols].head(top)
    # simple console table
    widths = [max(len(str(x)) for x in [c] + show[c].astype(str).tolist()) for c in cols]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for _, row in show.iterrows():
        print("  ".join(str(row[c]).ljust(w) for c, w in zip(cols, widths)))

# ---------- Main ----------
def main():
    cfg = load_config("config/config.yaml")

    df = _load_personality(cfg)
    df = _rankings(df)

    # Sort for each view
    aggr_sorted  = df.sort_values(["rank_aggression","driver"])
    def_sorted   = df.sort_values(["rank_defence","driver"])
    risk_sorted  = df.sort_values(["rank_risk","driver"])
    safest_sorted= df.sort_values(["rank_safest","driver"])

    # Console summaries
    _print_table(
        aggr_sorted.assign(rank=aggr_sorted["rank_aggression"].astype(int))[
            ["rank","driver","aggression_str","n_attacks","n_opps"] if "n_attacks" in df.columns and "n_opps" in df.columns
            else ["rank","driver","aggression_str"]
        ],
        cols=["rank","driver","aggression_str"] + (["n_attacks","n_opps"] if "n_attacks" in df.columns else []),
        title="Most Aggressive"
    )

    _print_table(
        def_sorted.assign(rank=def_sorted["rank_defence"].astype(int))[
            ["rank","driver","defence_str","n_defences","n_threats"] if "n_defences" in df.columns and "n_threats" in df.columns
            else ["rank","driver","defence_str"]
        ],
        cols=["rank","driver","defence_str"] + (["n_defences","n_threats"] if "n_defences" in df.columns else []),
        title="Best Defence (hold probability)"
    )

    _print_table(
        risk_sorted.assign(rank=risk_sorted["rank_risk"].astype(int))[
            ["rank","driver","risk_str","n_incidents","exposure_laps"] if "n_incidents" in df.columns and "exposure_laps" in df.columns
            else ["rank","driver","risk_str"]
        ],
        cols=["rank","driver","risk_str"] + (["n_incidents","exposure_laps"] if "n_incidents" in df.columns else []),
        title="Riskiest (higher incident/DNF propensity)"
    )

    _print_table(
        safest_sorted.assign(rank=safest_sorted["rank_safest"].astype(int))[
            ["rank","driver","risk_str","n_incidents","exposure_laps"] if "n_incidents" in df.columns and "exposure_laps" in df.columns
            else ["rank","driver","risk_str"]
        ],
        cols=["rank","driver","risk_str"] + (["n_incidents","exposure_laps"] if "n_incidents" in df.columns else []),
        title="Safest (lowest risk)"
    )

    # Save a tidy rankings CSV + a short markdown
    out_dir = _project_root() / "outputs" / "reports"
    _ensure_dir(out_dir)

    keep_cols = [
        "driver",
        "aggression","aggression_se","aggr_lo","aggr_hi","rank_aggression",
        "defence","defence_se","def_lo","def_hi","rank_defence",
        "risk","risk_se","risk_lo","risk_hi","rank_risk","rank_safest",
    ] + [c for c in ["n_attacks","n_opps","n_defences","n_threats","n_incidents","exposure_laps"] if c in df.columns]

    df[keep_cols].sort_values("rank_aggression").to_csv(out_dir / "personality_rankings.csv", index=False)

    # Minimal Markdown summary
    md = []
    md.append("# Personality Rankings\n")
    def _md_section(title: str, table: pd.DataFrame, metric_col: str):
        md.append(f"## {title}\n")
        md.append("| Rank | Driver | Value |\n|---:|:---|---:|\n")
        for _, r in table.head(10).iterrows():
            md.append(f"| {int(r['rank'])} | {r['driver']} | {r[metric_col]} |\n")

    _md_section("Most Aggressive", aggr_sorted.assign(rank=aggr_sorted["rank_aggression"].astype(int)), "aggression_str")
    _md_section("Best Defence", def_sorted.assign(rank=def_sorted["rank_defence"].astype(int)), "defence_str")
    _md_section("Riskiest", risk_sorted.assign(rank=risk_sorted["rank_risk"].astype(int)), "risk_str")
    _md_section("Safest", safest_sorted.assign(rank=safest_sorted["rank_safest"].astype(int)), "risk_str")

    (out_dir / "personality_report.md").write_text("".join(md), encoding="utf-8")

    print(f"\n[INFO] Wrote: {out_dir / 'personality_rankings.csv'}")
    print(f"[INFO] Wrote: {out_dir / 'personality_report.md'}")

if __name__ == "__main__":
    main()
