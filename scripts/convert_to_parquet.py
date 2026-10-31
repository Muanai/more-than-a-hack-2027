"""Convert raw match assets (ATD tracking, smart tagging, homography) to fast, compressed Parquet files."""

import argparse
import os
import time
import xml.etree.ElementTree as ET
import numpy as np
import polars as pl

# Pitch dimensions in meters
PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0


def parse_atd_meta(meta_path: str):
    """Parse ATD XML metadata: teams, player track slots, and team mapping."""
    root = ET.parse(meta_path).getroot()
    teams = [(team.get("id"), team.findtext("Name")) for team in root.iter("Team")]
    if len(teams) < 2:
        raise ValueError(f"Expected at least 2 teams in {meta_path}, found {len(teams)}")

    home_id, home_name = teams[0]
    away_id, away_name = teams[1]

    slots = []
    for player in root.iter("Player"):
        slot_id = int(player.get("id"))
        team_id = player.get("teamId")
        team_code = "H" if team_id == home_id else "A"
        slots.append({
            "slot_id": slot_id,
            "name": player.findtext("Name"),
            "shirt_number": int(player.findtext("ShirtNumber") or slot_id),
            "team": team_code,
            "team_name": home_name if team_code == "H" else away_name,
        })

    slots_df = pl.DataFrame(slots)
    return {
        "home_name": home_name,
        "away_name": away_name,
        "slots_df": slots_df,
        "n_slots": len(slots),
    }


def convert_atd(raw_path: str, meta_info: dict, out_tracking_path: str, out_slots_path: str):
    """Parse raw ATD text and save tracking.parquet and player_slots.parquet."""
    print(f"Reading raw ATD from {raw_path}...")
    t0 = time.time()
    n_slots = meta_info["n_slots"]

    # First pass: count lines
    with open(raw_path, "r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)

    print(f"Total frames: {n_lines}, slots: {n_slots}")

    # Pre-allocate arrays
    ball_x = np.full(n_lines, np.nan, dtype=np.float32)
    ball_y = np.full(n_lines, np.nan, dtype=np.float32)
    ball_speed = np.full(n_lines, np.nan, dtype=np.float32)

    px = np.full((n_lines, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_lines, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n_lines, n_slots), np.nan, dtype=np.float32)

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue

            frame_id = int(parts[0])
            idx = frame_id - 1  # 0-indexed video frame

            # Parse ball: x,y,speed
            b = parts[2].split(",")
            if b[0] != "NaN":
                ball_x[idx] = float(b[0])
                ball_y[idx] = float(b[1])
                if len(b) > 2 and b[2] != "NaN":
                    ball_speed[idx] = float(b[2])

            # Parse players: x,y,speed for each slot
            slots = parts[1].split(";")
            for s_idx, slot_str in enumerate(slots):
                if s_idx >= n_slots or slot_str.startswith("NaN"):
                    continue
                vals = slot_str.split(",")
                px[idx, s_idx] = float(vals[0])
                py[idx, s_idx] = float(vals[1])
                if len(vals) > 2 and vals[2] != "NaN":
                    ps[idx, s_idx] = float(vals[2])

    # Construct Polars DataFrame
    cols = {
        "frame": np.arange(n_lines, dtype=np.int32),
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_speed": ball_speed,
    }
    for s in range(n_slots):
        slot_num = s + 1
        cols[f"p{slot_num}_x"] = px[:, s]
        cols[f"p{slot_num}_y"] = py[:, s]
        cols[f"p{slot_num}_speed"] = ps[:, s]

    df = pl.DataFrame(cols)
    os.makedirs(os.path.dirname(out_tracking_path), exist_ok=True)
    df.write_parquet(out_tracking_path, compression="zstd")
    meta_info["slots_df"].write_parquet(out_slots_path, compression="zstd")

    file_size_mb = os.path.getsize(out_tracking_path) / (1024 * 1024)
    print(f"Saved {out_tracking_path} ({file_size_mb:.1f} MB) in {time.time() - t0:.2f}s")
    print(f"Saved {out_slots_path}")


def convert_homography(raw_csv_path: str, out_parquet_path: str):
    """Convert homography CSV to Parquet."""
    print(f"Converting homography from {raw_csv_path}...")
    t0 = time.time()
    df = pl.read_csv(raw_csv_path)
    os.makedirs(os.path.dirname(out_parquet_path), exist_ok=True)
    df.write_parquet(out_parquet_path, compression="zstd")
    file_size_mb = os.path.getsize(out_parquet_path) / (1024 * 1024)
    print(f"Saved {out_parquet_path} ({file_size_mb:.1f} MB) in {time.time() - t0:.2f}s")


def convert_smart_tagging(raw_xml_path: str, out_parquet_path: str, fps: float = 25.0):
    """Convert Smart Tagging XML intervals to Parquet."""
    print(f"Converting smart tagging from {raw_xml_path}...")
    t0 = time.time()
    root = ET.parse(raw_xml_path).getroot()

    records = []
    for instance in root.iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        code = instance.findtext("code")

        tags = {}
        for label in instance.findall("label"):
            group = label.findtext("group")
            text = label.findtext("text")
            if group:
                tags[group] = text

        records.append({
            "code": code,
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            "team": tags.get("Team"),
            "half": tags.get("Half"),
            "type": tags.get("Type"),
            "side": tags.get("Side"),
            "ball_entry_direction": tags.get("Direction of ball entry"),
            "max_players_in_box": tags.get("Max Players in the box"),
        })

    schema = {
        "code": pl.String,
        "start_s": pl.Float32,
        "end_s": pl.Float32,
        "start_frame": pl.Int32,
        "end_frame": pl.Int32,
        "team": pl.String,
        "half": pl.String,
        "type": pl.String,
        "side": pl.String,
        "ball_entry_direction": pl.String,
        "max_players_in_box": pl.String,
    }
    df = pl.DataFrame(records, schema=schema)
    os.makedirs(os.path.dirname(out_parquet_path), exist_ok=True)
    df.write_parquet(out_parquet_path, compression="zstd")
    print(f"Saved {out_parquet_path} ({len(df)} events) in {time.time() - t0:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Convert raw match assets to Parquet")
    parser.add_argument("--source", choices=["tactical", "broadcast", "all"], default="tactical",
                        help="Data source to convert (default: tactical)")
    args = parser.parse_args()

    sources = ["tactical"] if args.source == "tactical" else (["broadcast"] if args.source == "broadcast" else ["tactical", "broadcast"])

    for src in sources:
        print(f"\n=== Processing {src} ===")
        raw_dir = os.path.join("data", "raw", src)
        processed_dir = os.path.join("data", "processed", src)

        prefix = f"BAR-ATH_{src}"
        atd_raw = os.path.join(raw_dir, f"{prefix}_atd.txt")
        atd_meta = os.path.join(raw_dir, f"{prefix}_atd.xml")
        homography_csv = os.path.join(raw_dir, f"{prefix}_homography.csv")
        smart_tagging_xml = os.path.join(raw_dir, f"{prefix}_smart_tagging.xml")

        # 1. ATD Tracking & Slots
        if os.path.exists(atd_raw) and os.path.exists(atd_meta):
            meta_info = parse_atd_meta(atd_meta)
            convert_atd(
                raw_path=atd_raw,
                meta_info=meta_info,
                out_tracking_path=os.path.join(processed_dir, "tracking.parquet"),
                out_slots_path=os.path.join(processed_dir, "player_slots.parquet"),
            )

        # 2. Homography
        if os.path.exists(homography_csv):
            convert_homography(
                raw_csv_path=homography_csv,
                out_parquet_path=os.path.join(processed_dir, "homography.parquet"),
            )

        # 3. Smart Tagging (tactical only)
        if os.path.exists(smart_tagging_xml):
            convert_smart_tagging(
                raw_xml_path=smart_tagging_xml,
                out_parquet_path=os.path.join(processed_dir, "smart_tagging.parquet"),
            )

    print("\nAll done!")


if __name__ == "__main__":
    main()
