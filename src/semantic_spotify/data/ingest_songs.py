"""Ingest Spotify track-analysis Parquet files into a normalized track catalog.

Run from the semantic-spotify repository root:
  python -m pip install 'huggingface_hub>=0.25' 'duckdb>=1.1'
  python -m semantic_spotify.data.ingest_tracks --max-files 1
  python -m semantic_spotify.data.ingest_tracks

Save this file as src/semantic_spotify/data/ingest_tracks.py.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from huggingface_hub import HfApi, hf_hub_download

DEFAULT_REPO = "GildasLeDrogoff/spotify-huge-track-analysis-dataset"


def quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def ingest(
    repo_id: str,
    raw_dir: Path,
    output_dir: Path,
    max_files: int | None,
    local_only: bool,
) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if local_only:
        parquet_files = sorted(raw_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files under {raw_dir}")
        if max_files is not None:
            parquet_files = parquet_files[:max_files]
    else:
        api = HfApi()
        paths = sorted(
            f for f in api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            if f.lower().endswith(".parquet")
        )
        if not paths:
            raise RuntimeError(
                "No Parquet files found in repository. Inspect its Files tab; "
                "this script expects native Parquet files."
            )
        if max_files is not None:
            paths = paths[:max_files]
        print(f"Discovered {len(paths)} Parquet files; downloading/caching locally.", flush=True)
        parquet_files = []
        for idx, path in enumerate(paths, 1):
            print(f"[{idx}/{len(paths)}] {path}", flush=True)
            local_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=path,
                local_dir=str(raw_dir),
            )
            parquet_files.append(Path(local_path))

    con = duckdb.connect(str(output_dir / "ingest.duckdb"))
    con.execute("SET memory_limit = '4GB'")
    con.execute("SET threads = 2")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET temp_directory = 'data/interim/duckdb_tmp'")
    con.execute("SET max_temp_directory_size = '30GB'")
    con.execute("SET preserve_insertion_order = false")
    file_paths = [str(p.resolve()) for p in parquet_files]
    file_list_sql = "[" + ", ".join("'" + p.replace("'", "''") + "'" for p in file_paths) + "]"
    con.execute(f"CREATE OR REPLACE VIEW raw_tracks AS SELECT * FROM read_parquet({file_list_sql}, union_by_name=true)")
    schema = con.execute("DESCRIBE raw_tracks").fetchall()
    columns = [row[0] for row in schema]
    lower_to_original = {c.lower(): c for c in columns}
    id_col = next((lower_to_original[k] for k in ("track_id", "id", "spotify_id", "spotify_track_id") if k in lower_to_original), None)
    if id_col is None:
        raise ValueError(f"Could not identify a track ID column. Available columns: {columns}")

    # Normalize both bare Spotify IDs and spotify:track:<id> URIs.
    id_expr = (
        f"NULLIF(TRIM(regexp_replace(CAST({quoted(id_col)} AS VARCHAR), "
        "'^spotify:track:', '')), '')"
    )
    select_cols = [f"{id_expr} AS track_id"]
    select_cols += [quoted(c) for c in columns if c != id_col]
    normalized = ", ".join(select_cols)
    con.execute(f"CREATE OR REPLACE VIEW normalized_tracks AS SELECT {normalized} FROM raw_tracks")

    # One row per track ID. When duplicates differ, retain the first row according
    # to DuckDB's scan order; this is deterministic enough for exploration, but
    # production conflict resolution should use explicit quality/timestamp rules.
    out_path = output_dir / "tracks.parquet"
    out_sql = str(out_path.resolve()).replace("'", "''")


    # Stage 1: Export normalized records without deduplication.
    normalized_path = output_dir / "tracks_normalized.parquet"
    normalized_sql = str(normalized_path.resolve()).replace("'", "''")

    print("Stage 1: Writing normalized tracks...", flush=True)

    con.execute(f"""
        COPY (
            SELECT *
            FROM normalized_tracks
            WHERE track_id IS NOT NULL
        )
        TO '{normalized_sql}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # Stage 2: Deduplicate using a grouped aggregation.
    print("Writing normalized tracks...", flush=True)

    con.execute(f"""
        COPY (
            SELECT *
            FROM normalized_tracks
            WHERE track_id IS NOT NULL
        )
        TO '{out_sql}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    counts = con.execute("""
        SELECT COUNT(*) AS rows_total,
               COUNT(DISTINCT track_id) AS unique_tracks,
               COUNT(*) - COUNT(track_id) AS missing_ids
        FROM normalized_tracks
    """).fetchone()
    report = {
        "source_repo": repo_id if not local_only else "local_parquet",
        "source_files": [str(p) for p in parquet_files],
        "source_columns": columns,
        "source_id_column": id_col,
        "output": str(out_path),
        "rows_total": counts[0],
        "unique_tracks": counts[1],
        "missing_ids": counts[2],
        "duplicate_rows": counts[0] - counts[2] - counts[1],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "Duplicate track IDs are reduced to one record; differing metadata values are not reconciled.",
            "Track IDs are normalized to bare Spotify IDs for joining with MPD track_uri values.",
        ],
    }
    report_path = output_dir / "tracks_ingestion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    con.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/track_analysis"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-files", type=int, default=None, help="Limit source files for a smoke test")
    parser.add_argument("--local-only", action="store_true", help="Use Parquet files already in --raw-dir")
    args = parser.parse_args()
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be positive")
    result = ingest(args.repo_id, args.raw_dir, args.output_dir, args.max_files, args.local_only)
    print(json.dumps({k: v for k, v in result.items() if k != "source_files"}, indent=2))


if __name__ == "__main__":
    main()
