# scripts/gen_track_meta.py
from pathlib import Path
import pandas as pd

ROWS = [
    {"event_key":"__default__","track_type":"permanent","downforce_index":0.60,"drs_zones":2,"speed_bias":0.00,"overtaking_difficulty":0.50},
    {"event_key":"bahrain","track_type":"permanent","downforce_index":0.55,"drs_zones":3,"speed_bias":0.30,"overtaking_difficulty":0.50},
    {"event_key":"jeddah","track_type":"street","downforce_index":0.45,"drs_zones":3,"speed_bias":0.60,"overtaking_difficulty":0.40},
    {"event_key":"australia","track_type":"permanent","downforce_index":0.55,"drs_zones":4,"speed_bias":0.15,"overtaking_difficulty":0.45},
    {"event_key":"china","track_type":"permanent","downforce_index":0.55,"drs_zones":2,"speed_bias":0.20,"overtaking_difficulty":0.45},
    {"event_key":"miami","track_type":"street","downforce_index":0.50,"drs_zones":3,"speed_bias":0.30,"overtaking_difficulty":0.50},
    {"event_key":"imola","track_type":"permanent","downforce_index":0.70,"drs_zones":2,"speed_bias":-0.10,"overtaking_difficulty":0.60},
    {"event_key":"monaco","track_type":"street","downforce_index":0.95,"drs_zones":1,"speed_bias":-0.80,"overtaking_difficulty":0.95},
    {"event_key":"canadian","track_type":"street","downforce_index":0.50,"drs_zones":2,"speed_bias":0.20,"overtaking_difficulty":0.45},
    {"event_key":"spain","track_type":"permanent","downforce_index":0.70,"drs_zones":2,"speed_bias":-0.10,"overtaking_difficulty":0.55},
    {"event_key":"austria","track_type":"permanent","downforce_index":0.40,"drs_zones":3,"speed_bias":0.60,"overtaking_difficulty":0.35},
    {"event_key":"silverstone","track_type":"permanent","downforce_index":0.65,"drs_zones":2,"speed_bias":0.10,"overtaking_difficulty":0.40},
    {"event_key":"hungary","track_type":"permanent","downforce_index":0.85,"drs_zones":2,"speed_bias":-0.50,"overtaking_difficulty":0.80},
    {"event_key":"belgian","track_type":"permanent","downforce_index":0.45,"drs_zones":2,"speed_bias":0.50,"overtaking_difficulty":0.40},
    {"event_key":"zandvoort","track_type":"permanent","downforce_index":0.80,"drs_zones":2,"speed_bias":-0.30,"overtaking_difficulty":0.70},
    {"event_key":"monza","track_type":"permanent","downforce_index":0.20,"drs_zones":2,"speed_bias":0.90,"overtaking_difficulty":0.25},
    {"event_key":"singapore","track_type":"street","downforce_index":0.90,"drs_zones":3,"speed_bias":-0.60,"overtaking_difficulty":0.80},
    {"event_key":"japan","track_type":"permanent","downforce_index":0.75,"drs_zones":1,"speed_bias":-0.20,"overtaking_difficulty":0.65},
    {"event_key":"qatar","track_type":"permanent","downforce_index":0.50,"drs_zones":3,"speed_bias":0.40,"overtaking_difficulty":0.45},
    {"event_key":"usa_cota","track_type":"permanent","downforce_index":0.60,"drs_zones":2,"speed_bias":0.10,"overtaking_difficulty":0.50},
    {"event_key":"mexico","track_type":"permanent","downforce_index":0.55,"drs_zones":3,"speed_bias":0.40,"overtaking_difficulty":0.45},
    {"event_key":"sao_paulo","track_type":"permanent","downforce_index":0.60,"drs_zones":2,"speed_bias":0.20,"overtaking_difficulty":0.45},
    {"event_key":"vegas","track_type":"street","downforce_index":0.25,"drs_zones":2,"speed_bias":0.80,"overtaking_difficulty":0.35},
    {"event_key":"abu_dhabi","track_type":"permanent","downforce_index":0.60,"drs_zones":3,"speed_bias":0.20,"overtaking_difficulty":0.50},
    {"event_key":"baku","track_type":"street","downforce_index":0.35,"drs_zones":2,"speed_bias":0.70,"overtaking_difficulty":0.30},
]

COLUMNS = ["event_key","track_type","downforce_index","drs_zones","speed_bias","overtaking_difficulty"]

def main():
    # Write to <repo_root>/data/track_meta.csv
    proj = Path(__file__).resolve().parents[1]  # adjust if you place this elsewhere
    out_dir = proj / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "track_meta.csv"

    df = pd.DataFrame(ROWS)[COLUMNS]
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    main()
