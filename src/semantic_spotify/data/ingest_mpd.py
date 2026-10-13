"""Ingest the available Spotify MPD JSON slices into partitioned Parquet.

Run: python -m semantic_spotify.data.ingest_mpd --max-files 2
Dependencies: pip install huggingface_hub pyarrow
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_REPO = "jaxliu/Spotify_Million_Playlist_Dataset_Challenge"

PLAYLIST_SCHEMA = pa.schema([
    ("playlist_id", pa.int64()), ("name", pa.string()),
    ("collaborative", pa.bool_()), ("modified_at", pa.int64()),
    ("num_tracks", pa.int32()), ("num_albums", pa.int32()),
    ("num_followers", pa.int32()), ("num_edits", pa.int32()),
    ("duration_ms", pa.int64()), ("num_artists", pa.int32()),
    ("source_file", pa.string()),
])
EDGE_SCHEMA = pa.schema([
    ("playlist_id", pa.int64()), ("position", pa.int32()),
    ("track_id", pa.string()), ("artist_name", pa.string()),
    ("track_name", pa.string()), ("album_name", pa.string()),
    ("artist_id", pa.string()), ("album_id", pa.string()),
    ("duration_ms", pa.int64()),
])


def spotify_id(uri: str | None, kind: str) -> str | None:
    if not uri:
        return None
    prefix = f"spotify:{kind}:"
    return uri[len(prefix):] if uri.startswith(prefix) and len(uri) > len(prefix) else None


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return None


def write_parquet(records: list[dict], schema: pa.Schema, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(records, schema=schema), temp, compression="zstd")
    temp.replace(destination)


def ingest(repo_id: str, output_dir: Path, max_files: int | None,
           local_dir: Path | None, force: bool) -> dict:
    if local_dir:
        sources = sorted(local_dir.glob("mpd.slice.*.json"))
        if not sources:
            sources = sorted(local_dir.glob("data/mpd.slice.*.json"))
        names = [p.name for p in sources]
    else:
        names = sorted(
            f for f in HfApi().list_repo_files(repo_id, repo_type="dataset")
            if f.startswith("data/mpd.slice.") and f.endswith(".json")
        )
        sources = []
    if not names:
        raise RuntimeError("No MPD JSON slices found")
    if max_files is not None:
        names = names[:max_files]
    print(f"Found {len(names)} MPD slices to process", flush=True)
    totals = {"slices_written": 0, "slices_skipped": 0, "playlists_written": 0,
              "edges_written": 0, "invalid_track_uris": 0}
    for index, name in enumerate(names, start=1):
        stem = Path(name).stem
        playlist_out = output_dir / "playlists" / f"{stem}.parquet"
        edges_out = output_dir / "playlist_tracks" / f"{stem}.parquet"
        if not force and playlist_out.exists() and edges_out.exists():
            totals["slices_skipped"] += 1
            print(f"[{index}/{len(names)}] skip {name} (already ingested)", flush=True)
            continue
        if local_dir:
            src = sources[index - 1]
        else:
            src = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=name))
        with src.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        playlist_rows = []
        edge_rows = []
        for playlist in payload["playlists"]:
            pid = playlist["pid"]
            playlist_rows.append({
                "playlist_id": pid, "name": playlist.get("name"),
                "collaborative": as_bool(playlist.get("collaborative")),
                "modified_at": playlist.get("modified_at"),
                "num_tracks": playlist.get("num_tracks"),
                "num_albums": playlist.get("num_albums"),
                "num_followers": playlist.get("num_followers"),
                "num_edits": playlist.get("num_edits"),
                "duration_ms": playlist.get("duration_ms"),
                "num_artists": playlist.get("num_artists"),
                "source_file": name,
            })
            for track in playlist.get("tracks", []):
                tid = spotify_id(track.get("track_uri"), "track")
                if tid is None:
                    totals["invalid_track_uris"] += 1
                    continue
                edge_rows.append({
                    "playlist_id": pid, "position": track.get("pos"),
                    "track_id": tid, "artist_name": track.get("artist_name"),
                    "track_name": track.get("track_name"),
                    "album_name": track.get("album_name"),
                    "artist_id": spotify_id(track.get("artist_uri"), "artist"),
                    "album_id": spotify_id(track.get("album_uri"), "album"),
                    "duration_ms": track.get("duration_ms"),
                })
        # Write edges first: presence of both files is our completion marker.
        write_parquet(edge_rows, EDGE_SCHEMA, edges_out)
        write_parquet(playlist_rows, PLAYLIST_SCHEMA, playlist_out)
        totals["slices_written"] += 1
        totals["playlists_written"] += len(playlist_rows)
        totals["edges_written"] += len(edge_rows)
        print(f"[{index}/{len(names)}] {name}: {len(playlist_rows)} playlists, "
              f"{len(edge_rows)} edges", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source_repo": str(local_dir) if local_dir else repo_id,
        "slices_selected": len(names), "metrics_this_run": totals,
        "playlist_glob": str(output_dir / "playlists" / "*.parquet"),
        "edge_glob": str(output_dir / "playlist_tracks" / "*.parquet"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": ["Playlist IDs are MPD internal pids, not Spotify playlist IDs.",
                  "MPD contains playlist titles, not descriptions.",
                  "Outputs are per-slice Parquet for incremental, resumable ingestion.",
                  "Track metadata repeats on playlist edges; canonicalize after join."],
    }
    (output_dir / "ingestion_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/mpd"))
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--local-dir", type=Path, help="Use locally downloaded MPD JSON files")
    parser.add_argument("--force", action="store_true", help="Reprocess completed slices")
    args = parser.parse_args()
    print(json.dumps(ingest(args.repo_id, args.output_dir, args.max_files,
                            args.local_dir, args.force), indent=2))


if __name__ == "__main__":
    main()
