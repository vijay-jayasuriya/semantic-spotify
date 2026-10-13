"""Build the canonical Semantic Spotify training catalog from MPD and audio Parquet.

Run from the repository root:
    python -m semantic_spotify.data.build_catalog

Requires: duckdb (pip install duckdb)

Outputs one row per MPD track, retains tracks with missing audio, preserves
playlist edges and artist associations, and assigns deterministic playlist-level
train/val/test splits (80/10/10).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

AUDIO_FEATURES = (
    "tempo", "key", "mode", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
)


def sql_path(path: Path | str) -> str:
    return str(path).replace("'", "''")


def build(args: argparse.Namespace) -> dict:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    temp_dir = args.temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(output / "catalog_build.duckdb"))
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute(f"SET threads = {args.threads}")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    con.execute(f"SET max_temp_directory_size = '{args.max_temp_size}'")

    mpd = sql_path(args.mpd_dir)
    audio = sql_path(args.audio_candidates)
    con.execute(f"CREATE OR REPLACE VIEW raw_edges AS SELECT * FROM read_parquet('{mpd}/playlist_tracks/*.parquet')")
    con.execute(f"CREATE OR REPLACE VIEW raw_playlists AS SELECT * FROM read_parquet('{mpd}/playlists/*.parquet')")
    con.execute(f"CREATE OR REPLACE VIEW raw_audio AS SELECT * FROM read_parquet('{audio}')")

    def export(name: str, query: str) -> None:
        target = output / f"{name}.parquet"
        tmp = output / f"{name}.parquet.partial"
        print(f"Writing {target} ...", flush=True)
        con.execute(f"COPY ({query}) TO '{sql_path(tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        tmp.replace(target)

    # Use the playlist occurrence with the most complete identifiers for each
    # track; resolve ties deterministically. This preserves MPD metadata even
    # when the audio database has no matching track ID.
    export("mpd_track_metadata", """
        WITH ranked AS (
            SELECT
                track_id,
                track_name,
                artist_name,
                artist_id,
                album_name,
                album_id,
                duration_ms,
                ROW_NUMBER() OVER (
                    PARTITION BY track_id
                    ORDER BY
                        CASE WHEN artist_id IS NULL THEN 1 ELSE 0 END,
                        CASE WHEN track_name IS NULL THEN 1 ELSE 0 END,
                        COALESCE(track_name, ''),
                        COALESCE(artist_name, ''),
                        COALESCE(artist_id, ''),
                        COALESCE(album_id, ''),
                        playlist_id,
                        p.position
                ) AS rn
            FROM raw_edges AS p
            WHERE track_id IS NOT NULL
        )
        SELECT
            track_id,
            track_name,
            artist_name,
            artist_id,
            album_name,
            album_id,
            duration_ms
        FROM ranked
        WHERE rn = 1
    """)
    con.execute(f"CREATE OR REPLACE VIEW mpd_meta AS SELECT * FROM read_parquet('{sql_path(output / 'mpd_track_metadata.parquet')}')")

    # The 467k matching audio records can be deduplicated safely after the
    # MPD semijoin. Artist-specific source fields are NOT canonical track fields.
    # Prefer records with more populated audio features, then stable metadata.
    feature_count = " + ".join(f"CAST({f} IS NOT NULL AS INTEGER)" for f in AUDIO_FEATURES)
    audio_cols = ", ".join(f"a.{f}" for f in AUDIO_FEATURES)
    export("tracks", f"""
        WITH ranked_audio AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY track_id
                ORDER BY ({feature_count}) DESC,
                         coalesce(artist_name, '') ASC,
                         coalesce(album_name, '') ASC,
                         coalesce(track_name, '') ASC
            ) AS rn
            FROM raw_audio WHERE track_id IS NOT NULL
        )
        SELECT m.track_id, m.track_name, m.artist_name, m.artist_id,
               m.album_name, m.album_id, m.duration_ms,
               (a.track_id IS NOT NULL) AS has_audio_features,
               {audio_cols},
               a.track_popularity, a.album_popularity,
               a.album_release_date, a.explicit
        FROM mpd_meta AS m
        LEFT JOIN ranked_audio AS a
          ON m.track_id = a.track_id AND a.rn = 1
    """)

    # MPD artist IDs identify actual Spotify artists. Source audio rows have
    # names but no artist IDs; keep those separately with an explicit source.
    # Do not conflate equal names with equal artist identities.
    export("track_artists", """
        SELECT DISTINCT track_id, artist_id, artist_name, 'mpd' AS source
        FROM raw_edges
        WHERE track_id IS NOT NULL AND artist_id IS NOT NULL
        UNION
        SELECT DISTINCT track_id, NULL::VARCHAR AS artist_id,
               artist_name, 'audio_features' AS source
        FROM raw_audio
        WHERE track_id IS NOT NULL AND artist_name IS NOT NULL
    """)

    export("playlists", """
        SELECT playlist_id, name, collaborative, modified_at, num_tracks,
               num_albums, num_followers, num_edits, duration_ms, num_artists,
               source_file,
               CASE WHEN playlist_id % 10 < 8 THEN 'train'
                    WHEN playlist_id % 10 = 8 THEN 'val'
                    ELSE 'test' END AS split
        FROM raw_playlists
    """)
    export("playlist_tracks", """
        SELECT playlist_id, position, track_id, artist_name, artist_id,
               track_name, album_name, album_id, duration_ms
        FROM raw_edges
    """)

    # Fail fast if catalog invariants are broken.
    tracks = sql_path(output / "tracks.parquet")
    edges = sql_path(output / "playlist_tracks.parquet")
    playlists = sql_path(output / "playlists.parquet")
    artists = sql_path(output / "track_artists.parquet")
    counts = con.execute(f"""
        SELECT
          (SELECT COUNT(*) FROM read_parquet('{tracks}')) AS tracks,
          (SELECT COUNT(DISTINCT track_id) FROM read_parquet('{tracks}')) AS unique_tracks,
          (SELECT COUNT(*) FROM read_parquet('{tracks}') WHERE has_audio_features) AS audio_tracks,
          (SELECT COUNT(*) FROM read_parquet('{playlists}')) AS playlists,
          (SELECT COUNT(DISTINCT playlist_id) FROM read_parquet('{playlists}')) AS unique_playlists,
          (SELECT COUNT(*) FROM read_parquet('{edges}')) AS edges,
          (SELECT COUNT(*) FROM read_parquet('{artists}')) AS artist_links
    """).fetchone()
    names = ["tracks", "unique_tracks", "audio_tracks", "playlists", "unique_playlists", "edges", "artist_links"]
    stats = dict(zip(names, counts))
    missing_track_refs = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{edges}') e
        ANTI JOIN read_parquet('{tracks}') t ON e.track_id = t.track_id
    """).fetchone()[0]
    missing_playlist_refs = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{edges}') e
        ANTI JOIN read_parquet('{playlists}') p ON e.playlist_id = p.playlist_id
    """).fetchone()[0]
    duplicate_edges = con.execute(f"""
        SELECT COUNT(*) FROM (
          SELECT playlist_id, position, COUNT(*) n
          FROM read_parquet('{edges}')
          GROUP BY playlist_id, position HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    stats.update(missing_track_references=missing_track_refs,
                 missing_playlist_references=missing_playlist_refs,
                 duplicate_playlist_positions=duplicate_edges,
                 audio_coverage=stats["audio_tracks"] / stats["tracks"] if stats["tracks"] else 0)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"mpd_dir": str(args.mpd_dir), "audio_candidates": str(args.audio_candidates)},
        "metrics": stats,
        "split": "Playlist-level: playlist_id modulo 10 (0-7 train, 8 val, 9 test)",
        "notes": [
            "No-audio tracks are retained with NULL audio features.",
            "Canonical metadata comes from MPD; audio source contributes numeric features.",
            "Artist-specific popularity/follower fields are intentionally omitted from canonical tracks.",
            "MPD artist identities and audio-source artist names are kept separately.",
            "mpd_track_metadata.parquet is an intermediate file retained for auditability.",
        ],
    }
    (output / "catalog_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if (stats["tracks"] != stats["unique_tracks"] or
        stats["playlists"] != stats["unique_playlists"] or
        missing_track_refs or missing_playlist_refs or duplicate_edges):
        raise RuntimeError("Catalog validation failed; inspect catalog_report.json")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mpd-dir", type=Path, default=Path("data/interim/mpd"))
    p.add_argument("--audio-candidates", type=Path,
                   default=Path("data/processed/mpd_audio_candidates.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed/catalog"))
    p.add_argument("--temp-dir", type=Path, default=Path("data/interim/duckdb_tmp"))
    p.add_argument("--memory-limit", default="4GB")
    p.add_argument("--max-temp-size", default="30GB")
    p.add_argument("--threads", type=int, default=2)
    build(p.parse_args())


if __name__ == "__main__":
    main()
