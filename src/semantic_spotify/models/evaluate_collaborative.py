"""Evaluate Semantic Spotify Item2Vec embeddings against track popularity.

Place at src/semantic_spotify/models/evaluate_collaborative.py and run:
    python -m semantic_spotify.models.evaluate_collaborative \
        --catalog-dir data/processed/catalog \
        --model-dir models/collaborative \
        --eval-playlists 200 --seed 42

Uses the SAME warm-start validation protocol as train_collaborative.py:
filter each validation playlist to training-seen tracks, shuffle with a seeded RNG,
use half as the query and half as hidden targets, and rank the training-seen catalog.

Also reports full-catalog recall with cold-start targets counted as misses.
No test playlists are accessed. No model is trained or modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np


def load_catalog(catalog_dir: Path):
    tracks_path = catalog_dir / "tracks.parquet"
    edges_path = catalog_dir / "playlist_tracks.parquet"
    for path in (tracks_path, edges_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing catalog file: {path}")

    con = duckdb.connect()
    try:
        track_ids = con.execute(
            "SELECT track_id FROM read_parquet(?) ORDER BY track_id", [str(tracks_path)]
        ).fetchnumpy()["track_id"].astype(str)
        rows = con.execute("""
            SELECT playlist_id, track_id
            FROM read_parquet(?)
            WHERE playlist_id IS NOT NULL AND track_id IS NOT NULL
            ORDER BY playlist_id, position
        """, [str(edges_path)]).fetchnumpy()
    finally:
        con.close()

    pid = np.asarray(rows["playlist_id"], dtype=np.int64)
    edge_ids = np.asarray(rows["track_id"], dtype=str)
    tid = np.searchsorted(track_ids, edge_ids)
    if np.any(tid >= len(track_ids)) or not np.all(track_ids[tid] == edge_ids):
        raise ValueError("Playlist edges reference tracks absent from catalog")
    return track_ids, pid, tid.astype(np.int32)


def get_playlists_and_counts(pid, tid, n_tracks):
    """Mirror train_collaborative.group_playlists and training_arrays."""
    cuts = np.r_[0, np.flatnonzero(pid[1:] != pid[:-1]) + 1, len(pid)]
    train_counts = np.zeros(n_tracks, dtype=np.int64)
    validation = []
    n_train = 0
    for a, b in zip(cuts[:-1], cuts[1:]):
        songs = np.unique(tid[a:b])
        if len(songs) < 2:
            continue
        split = int(pid[a]) % 10
        if split < 8:
            train_counts += np.bincount(songs, minlength=n_tracks)
            n_train += 1
        elif split == 8:
            validation.append(songs)
    return validation, train_counts, n_train


def ndcg_at_k(ranked, targets, k):
    """NDCG with binary relevance, ideal DCG based on number of targets."""
    if len(targets) == 0:
        return 0.0
    relevant = np.isin(ranked[:k], targets, assume_unique=False)
    dcg = float(np.sum(relevant / np.log2(np.arange(2, len(relevant) + 2))))
    ideal_n = min(k, len(targets))
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_n + 2))))
    return dcg / idcg if idcg else 0.0


def topk_excluding(scores, allowed, excluded, k):
    """Exact top-k, deterministic tie-breaking by lower track index."""
    candidates = np.flatnonzero(allowed)
    if len(excluded):
        candidates = candidates[~np.isin(candidates, excluded, assume_unique=False)]
    if len(candidates) == 0:
        return np.empty(0, dtype=np.int64)
    # Lexsort uses the final key (negative score) as primary and track index as tie-break.
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order[:k]]


def evaluate(embeddings, seen, validation, train_counts, ks, n_eval, seed):
    if not validation:
        raise ValueError("No eligible validation playlists found")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(validation), min(n_eval, len(validation)), replace=False)
    # The popularity baseline recommends tracks by number of training playlists
    # containing each track. Ties break by canonical track index.
    pop_scores = train_counts.astype(np.float64)
    methods = ("item2vec", "popularity")
    metrics = {
        method: {
            "warm_micro_recall": {k: 0 for k in ks},
            "full_micro_recall": {k: 0 for k in ks},
            "warm_macro_recall": {k: [] for k in ks},
            "warm_macro_ndcg": {k: [] for k in ks},
            "full_macro_ndcg": {k: [] for k in ks},
        } for method in methods
    }
    warm_targets = full_targets = n_used = query_tracks = 0
    max_k = max(ks)

    for j in chosen:
        all_songs = validation[j]
        warm_songs = all_songs[seen[all_songs]]
        if len(warm_songs) < 2:
            continue
        # Match training-script validation exactly for warm-start metrics:
        # RNG is used only for the subset and for permutations of warm songs.
        shuffled = rng.permutation(warm_songs)
        n_query = max(1, len(shuffled) // 2)
        query = shuffled[:n_query]
        hidden_warm = shuffled[n_query:]
        # Cold-start songs cannot be used as queries for this model, but are
        # counted as misses in the full-catalog evaluation.
        hidden_cold = all_songs[~seen[all_songs]]
        hidden_full = np.concatenate((hidden_warm, hidden_cold))
        if not len(hidden_warm):
            continue
        n_used += 1
        query_tracks += len(query)
        warm_targets += len(hidden_warm)
        full_targets += len(hidden_full)

        q = embeddings[query].mean(axis=0)
        q_norm = np.linalg.norm(q)
        sim_scores = embeddings @ (q / q_norm) if q_norm > 0 else np.zeros(len(seen))
        score_map = {"item2vec": sim_scores, "popularity": pop_scores}
        for method, scores in score_map.items():
            ranked = topk_excluding(scores, seen, query, max_k)
            for k in ks:
                selected = ranked[:k]
                hits = int(np.isin(hidden_warm, selected).sum())
                acc = metrics[method]
                acc["warm_micro_recall"][k] += hits
                acc["full_micro_recall"][k] += hits
                acc["warm_macro_recall"][k].append(hits / len(hidden_warm))
                acc["warm_macro_ndcg"][k].append(ndcg_at_k(selected, hidden_warm, k))
                acc["full_macro_ndcg"][k].append(ndcg_at_k(selected, hidden_full, k))

    if n_used == 0:
        raise ValueError("No validation playlists contain at least two training-seen tracks")

    output = {}
    for method in methods:
        acc = metrics[method]
        output[method] = {}
        for k in ks:
            output[method][f"warm_micro_recall_at_{k}"] = acc["warm_micro_recall"][k] / warm_targets
            output[method][f"warm_macro_recall_at_{k}"] = float(np.mean(acc["warm_macro_recall"][k]))
            output[method][f"warm_macro_ndcg_at_{k}"] = float(np.mean(acc["warm_macro_ndcg"][k]))
            output[method][f"full_micro_recall_at_{k}"] = acc["full_micro_recall"][k] / full_targets
            output[method][f"full_macro_ndcg_at_{k}"] = float(np.mean(acc["full_macro_ndcg"][k]))
    summary = {
        "validation_playlists_sampled": int(len(chosen)),
        "validation_playlists_evaluated": n_used,
        "query_track_occurrences": query_tracks,
        "warm_hidden_track_occurrences": warm_targets,
        "cold_hidden_track_occurrences": full_targets - warm_targets,
        "full_hidden_track_occurrences": full_targets,
        "hidden_track_warm_coverage": warm_targets / full_targets,
        "catalog_warm_coverage": float(seen.mean()),
        "note": "Warm evaluation filters to training-seen songs before query/hidden split, matching train_collaborative.py. Full recall counts other unseen songs as misses. NDCG is macro-averaged per playlist.",
    }
    return summary, output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog-dir", type=Path, default=Path("data/processed/catalog"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/collaborative"))
    parser.add_argument("--output", type=Path, default=None, help="Default: MODEL_DIR/evaluation.json")
    parser.add_argument("--eval-playlists", type=int, default=200, help="Use 10000 to evaluate all validation playlists")
    parser.add_argument("--seed", type=int, default=42, help="Use training --seed to match its Recall@50")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 50])
    args = parser.parse_args()
    if args.eval_playlists < 1 or not args.ks or any(k < 1 for k in args.ks):
        parser.error("--eval-playlists and all --ks must be positive")
    ks = sorted(set(args.ks))

    print("Loading catalog and model artifacts...", flush=True)
    track_ids, pid, tid = load_catalog(args.catalog_dir)
    validation, train_counts, n_train = get_playlists_and_counts(pid, tid, len(track_ids))
    model_ids = np.load(args.model_dir / "track_ids.npy", allow_pickle=False)
    embeddings = np.load(args.model_dir / "embeddings.npy", mmap_mode="r")
    model_seen = np.load(args.model_dir / "seen_in_train.npy", allow_pickle=False)
    if not np.array_equal(track_ids, model_ids.astype(str)):
        raise ValueError("Model track_ids.npy does not match canonical catalog track ordering")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(track_ids):
        raise ValueError(f"Invalid embedding shape: {embeddings.shape}")
    if model_seen.shape != (len(track_ids),):
        raise ValueError("Invalid seen_in_train.npy shape")
    seen = train_counts > 0
    if not np.array_equal(seen, model_seen.astype(bool)):
        raise ValueError("Model seen_in_train.npy differs from catalog train split")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinity")

    print(f"tracks={len(track_ids):,} train_playlists={n_train:,} val_playlists={len(validation):,} "
          f"seen_train_tracks={seen.sum():,} dim={embeddings.shape[1]}", flush=True)
    summary, scores = evaluate(embeddings, seen, validation, train_counts, ks, args.eval_playlists, args.seed)
    result = {
        "config": {"catalog_dir": str(args.catalog_dir), "model_dir": str(args.model_dir),
                   "eval_playlists": args.eval_playlists, "seed": args.seed, "ks": ks},
        "summary": summary, "metrics": scores,
    }
    print(json.dumps(result, indent=2), flush=True)
    output = args.output or (args.model_dir / "evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved evaluation to {output}", flush=True)


if __name__ == "__main__":
    main()
