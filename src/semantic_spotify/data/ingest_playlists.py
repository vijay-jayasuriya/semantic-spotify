"""Ingest Spotify playlists from local CSV or Hugging Face into Parquet.

Run from the project root:
    python -m semantic_spotify.data.ingest_playlists --input data/raw/playlists.csv

Or run directly:
    python src/semantic_spotify/data/ingest_playlists.py --input data/raw/playlists.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any
import sys
import csv

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging

csv.field_size_limit(sys.maxsize)

# Suppress verbose Hugging Face HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

LOGGER = logging.getLogger(__name__)

PLAYLIST_SCHEMA = pa.schema([
    ("playlist_id", pa.string()),
    ("name", pa.string()),
    ("description", pa.string()),
    ("followers_count", pa.int64()),
    ("total_tracks", pa.int64()),
    ("available_tracks", pa.int64()),
    ("valid_track_ids", pa.int64()),
    ("coverage", pa.float64()),
    ("is_complete", pa.bool_()),
    ("has_description", pa.bool_()),
    ("encoding_depth", pa.int32()),
    ("tracks_offset", pa.int64()),
    ("has_next_page", pa.bool_()),
])

TRACK_SCHEMA = pa.schema([
    ("playlist_id", pa.string()),
    ("track_id", pa.string()),
    ("position", pa.int64()),
    ("added_at", pa.string()),
])


def decode_playlist(raw: Any, max_depth: int = 3) -> tuple[dict[str, Any], int]:
    """Decode singly or multiply encoded JSON; return payload and decode count."""
    value = raw
    for depth in range(max_depth + 1):
        if isinstance(value, dict):
            return value, depth
        if depth == max_depth or not isinstance(value, str):
            break
        value = json.loads(value)
    raise ValueError(f"Expected JSON object within {max_depth} decodes")


def parse_playlist(row: pd.Series) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize one CSV record; raise ValueError for malformed playlist data."""
    playlist, depth = decode_playlist(row["raw_data"])
    playlist_id = playlist.get("id") or row.get("playlist_id")
    if not isinstance(playlist_id, str) or not playlist_id.strip():
        raise ValueError("Missing playlist ID")

    tracks = playlist.get("tracks")
    if not isinstance(tracks, dict):
        raise ValueError("Missing or invalid tracks object")
    items = tracks.get("items")
    if not isinstance(items, list):
        raise ValueError("Missing or invalid tracks.items list")

    total = tracks.get("total")
    offset = tracks.get("offset", 0)
    if not isinstance(total, int) or total < 0:
        raise ValueError("Invalid tracks.total")
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("Invalid tracks.offset")

    description = playlist.get("description")
    description = description if isinstance(description, str) else ""
    followers = playlist.get("followers")
    followers_count = followers.get("total") if isinstance(followers, dict) else None
    if not isinstance(followers_count, int):
        followers_count = None

    edges: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        track = item.get("track")
        if not isinstance(track, dict):
            continue
        track_id = track.get("id")
        if not isinstance(track_id, str) or not track_id.strip():
            continue
        added_at = item.get("added_at")
        edges.append({
            "playlist_id": playlist_id,
            "track_id": track_id,
            "position": offset + index,
            "added_at": added_at if isinstance(added_at, str) else None,
        })

    available = len(items)
    metadata = {
        "playlist_id": playlist_id,
        "name": playlist.get("name") if isinstance(playlist.get("name"), str) else None,
        "description": description,
        "followers_count": followers_count,
        "total_tracks": total,
        "available_tracks": available,
        "valid_track_ids": len(edges),
        "coverage": min(available / total, 1.0) if total else (1.0 if available == 0 else 0.0),
        "is_complete": offset == 0 and available == total and tracks.get("next") is None,
        "has_description": bool(description.strip()),
        "encoding_depth": depth,
        "tracks_offset": offset,
        "has_next_page": tracks.get("next") is not None,
    }
    return metadata, edges


def _write_batch(writer: pq.ParquetWriter, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def _local_rows(input_csv: Path, chunksize: int):
    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)
    for chunk in pd.read_csv(input_csv, chunksize=chunksize,
                             usecols=["playlist_id", "raw_data"],
                             dtype={"playlist_id": "string", "raw_data": "string"},
                             keep_default_na=False, on_bad_lines="error"):
        yield from (row for _, row in chunk.iterrows())


def _hf_rows(dataset_id: str, split: str, token: bool):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install Hugging Face datasets: pip install datasets") from exc
    dataset = load_dataset(dataset_id, split=split, streaming=True, token=token)
    for row in dataset:
        yield row


def ingest(input_csv: Path | None, output_dir: Path, chunksize: int = 1000,
           compression: str = "zstd", error_examples: int = 25,
           dataset_id: str | None = None, split: str = "train",
           token: bool = False, max_rows: int | None = None) -> dict[str, Any]:
    """Normalize local CSV or Hugging Face streaming rows into Parquet.

    Stream iteration can be slow for large CSV-backed datasets. Writes happen
    incrementally, but partial outputs are not resumable after interruption.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    if (input_csv is None) == (dataset_id is None):
        raise ValueError("Supply exactly one of input_csv or dataset_id")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    playlist_target = output_dir / "playlists.parquet"
    track_target = output_dir / "playlist_tracks.parquet"
    playlist_tmp = output_dir / "playlists.parquet.partial"
    track_tmp = output_dir / "playlist_tracks.parquet.partial"
    report_target = output_dir / "ingestion_report.json"
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    seen_ids: set[str] = set()  # Memory grows with unique playlist IDs.
    source = (_hf_rows(dataset_id, split, token) if dataset_id is not None
              else _local_rows(input_csv, chunksize))

    try:
        with pq.ParquetWriter(playlist_tmp, PLAYLIST_SCHEMA, compression=compression) as playlist_writer, \
             pq.ParquetWriter(track_tmp, TRACK_SCHEMA, compression=compression) as track_writer:
            playlist_batch: list[dict[str, Any]] = []
            edge_batch: list[dict[str, Any]] = []
            for row in source:
                if max_rows is not None and counts["rows_seen"] >= max_rows:
                    break
                counts["rows_seen"] += 1
                try:
                    metadata, edges = parse_playlist(row)
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                    counts["rows_failed"] += 1
                    counts[f"error_{type(exc).__name__}"] += 1
                    if len(errors) < error_examples:
                        errors.append({"source_row": counts["rows_seen"], "reason": str(exc)[:250]})
                    continue
                pid = metadata["playlist_id"]
                if pid in seen_ids:
                    counts["duplicate_playlist_ids"] += 1
                seen_ids.add(pid)
                counts["rows_succeeded"] += 1
                counts["track_edges"] += len(edges)
                counts["reported_total_tracks"] += metadata["total_tracks"]
                counts["available_track_items"] += metadata["available_tracks"]
                counts["complete_playlists"] += int(metadata["is_complete"])
                counts["playlists_with_description"] += int(metadata["has_description"])
                counts[f"encoding_depth_{metadata['encoding_depth']}"] += 1
                playlist_batch.append(metadata)
                edge_batch.extend(edges)
                if counts["rows_seen"] % chunksize == 0:
                    _write_batch(playlist_writer, playlist_batch, PLAYLIST_SCHEMA)
                    _write_batch(track_writer, edge_batch, TRACK_SCHEMA)
                    playlist_batch.clear()
                    edge_batch.clear()
                    LOGGER.info("Processed %s rows: %s successes, %s edges",
                                counts["rows_seen"], counts["rows_succeeded"], counts["track_edges"])
            _write_batch(playlist_writer, playlist_batch, PLAYLIST_SCHEMA)
            _write_batch(track_writer, edge_batch, TRACK_SCHEMA)
        playlist_tmp.replace(playlist_target)
        track_tmp.replace(track_target)
    except Exception:
        playlist_tmp.unlink(missing_ok=True)
        track_tmp.unlink(missing_ok=True)
        raise

    report = {
        "source": {"dataset_id": dataset_id, "split": split} if dataset_id else {"input_csv": str(input_csv)},
        "playlists_parquet": str(playlist_target),
        "playlist_tracks_parquet": str(track_target),
        "counts": dict(counts),
        "observed_item_coverage": (
            counts["available_track_items"] / counts["reported_total_tracks"]
            if counts["reported_total_tracks"] else None
        ),
        "error_examples": errors,
        "notes": [
            "Tracks beyond the stored Spotify API page are not available in raw_data.",
            "Rows with repeated playlist IDs are preserved, not deduplicated.",
            "Streaming is not resumable; interrupted runs must be restarted.",
        ],
    }
    report_target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Source local playlists.csv")
    source.add_argument("--stream", action="store_true", help="Stream from Hugging Face datasets")
    parser.add_argument("--dataset-id", default="devmaxjasper/spotify_public_playlists")
    parser.add_argument("--split", default="train")
    parser.add_argument("--token", action="store_true", help="Use cached Hugging Face login")
    parser.add_argument("--max-rows", type=int, help="Stop after N rows for testing")
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--chunksize", type=int, default=1000)
    parser.add_argument("--compression", choices=["zstd", "snappy", "gzip"], default="zstd")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = ingest(args.input, args.output_dir, args.chunksize, args.compression,
                    dataset_id=args.dataset_id if args.stream else None,
                    split=args.split, token=args.token, max_rows=args.max_rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
