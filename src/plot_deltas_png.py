# src/plot_deltas_png.py
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parent.parent

TEAM_COLORS = {
    "Red Bull": "#1E41FF", "Ferrari": "#DC0000", "Mercedes": "#00D2BE",
    "McLaren": "#FF8700", "Aston Martin": "#006F62", "Alpine": "#0090FF",
    "Williams": "#005AFF", "RB": "#2B4562", "Sauber": "#006F3C",
    "Haas": "#B6BABD", "UNKNOWN": "#888888",
}

def latest_event_file(substr: str) -> Path | None:
    mdir = PROJ / "outputs" / "metrics"
    if not mdir.exists():
        return None
    ss = substr.lower()
    cands = [p for p in sorted(mdir.glob("*-event_metrics.csv")) if ss in p.name.lower()]
    return cands[-1] if cands else None

def load_global() -> pd.DataFrame:
    f = PROJ / "outputs" / "aggregate" / "driver_ranking.csv"
    df = pd.read_csv(f)
    low = {c.lower(): c for c in df.columns}
    driver = low.get("driver", list(df.columns)[0])
    delta  = low.get("agg_delta_s") or low.get("equal_delta_s") or low.get("delta_s")
    se     = low.get("agg_se_s")
    team   = low.get("label_team") or low.get("team")
    cols = [driver, delta] + ([se] if se else []) + ([team] if team else [])
    out = df[cols].copy()
    out.rename(columns={
        driver: "driver",
        delta: "delta_s",
        (se or "agg_se_s"): "se_s",
        (team or "team"): "team"
    }, inplace=True)
    if "team" not in out.columns: out["team"] = "UNKNOWN"
    return out

def load_event(f: Path) -> pd.DataFrame:
    df = pd.read_csv(f)
    low = {c.lower(): c for c in df.columns}
    driver = low.get("driver", list(df.columns)[0])
    team   = low.get("team") or low.get("label_team")
    delta  = low.get("event_delta_s_shrunk") or low.get("event_delta_s") or low.get("race_delta_s")
    se     = low.get("event_se_s") or low.get("race_se_s")
    cols = [driver, delta] + ([se] if se else []) + ([team] if team else [])
    out = df[cols].copy()
    out.rename(columns={
        driver: "driver",
        delta: "delta_s",
        (se or "se_s"): "se_s",
        (team or "team"): "team"
    }, inplace=True)
    if "team" not in out.columns: out["team"] = "UNKNOWN"
    return out

def main():
    parser = argparse.ArgumentParser(description="Plot driver deltas (PNG).")
    parser.add_argument("--event", type=str, default=None,
                        help="Substring to pick latest *-event_metrics.csv (e.g., 'canadian').")
    parser.add_argument("--outfile", type=str, default=None,
                        help="Optional output filename (.png).")
    args = parser.parse_args()

    if args.event:
        f_evt = latest_event_file(args.event)
        if f_evt is None:
            print(f"[WARN] No event file matching '{args.event}'. Falling back to global.")
            data = load_global()
            title_suffix = "Global"
            out_name = "driver_deltas_global.png"
        else:
            data = load_event(f_evt)
            title_suffix = f_evt.stem
            out_name = f"driver_deltas_{args.event.lower()}.png"
    else:
        data = load_global()
        title_suffix = "Global"
        out_name = "driver_deltas_global.png"

    data = data.copy()
    data["delta_s"] = pd.to_numeric(data["delta_s"], errors="coerce")
    if "se_s" in data.columns:
        data["se_s"] = pd.to_numeric(data["se_s"], errors="coerce")
    data["team"] = data["team"].astype(str)
    data = data.dropna(subset=["delta_s"]).sort_values("delta_s", ascending=True)  # fastest (most negative) first

    colors = [TEAM_COLORS.get(t, TEAM_COLORS["UNKNOWN"]) for t in data["team"]]

    # --- Plot (horizontal bar) ---
    plt.figure(figsize=(9, 8), dpi=160)
    ax = plt.gca()
    y = range(len(data))
    x = data["delta_s"].to_numpy()

    if "se_s" in data.columns and data["se_s"].notna().any():
        ax.barh(y, x, xerr=data["se_s"].fillna(0.0).to_numpy(), color=colors, edgecolor="#222", linewidth=0.8, capsize=3)
    else:
        ax.barh(y, x, color=colors, edgecolor="#222", linewidth=0.8)

    ax.set_yticks(list(y))
    ax.set_yticklabels(data["driver"].tolist())
    ax.invert_yaxis()  # fastest at the top
    ax.axvline(0, color="#444", linewidth=1, linestyle="--")

    ax.set_xlabel("Δ vs field (seconds — lower is faster)")
    ax.set_title(f"Driver Pace Deltas (fastest → slowest) • {title_suffix}")

    # Annotate bar ends with values
    for i, val in enumerate(x):
        txt = f"{val:+.3f}"
        if val >= 0:
            ax.text(val + 0.01, i, txt, va="center", ha="left", fontsize=9)
        else:
            ax.text(val - 0.01, i, txt, va="center", ha="right", fontsize=9)

    ax.grid(axis="x", linestyle=":", alpha=0.35)
    plt.tight_layout()

    out_dir = PROJ / "outputs" / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (args.outfile if args.outfile else out_name)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[INFO] wrote PNG: {out_path}")

if __name__ == "__main__":
    main()
