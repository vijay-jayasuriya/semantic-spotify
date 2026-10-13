"""Model-agnostic nearest-neighbor recommendations for Semantic Spotify.

Run from repository root (editable install):
  python -m semantic_spotify.models.recommend --model collaborative_v2 --track 'Weird Fishes/Arpeggi' --artist Radiohead
  python -m semantic_spotify.models.recommend --model collaborative_v2 --track-id SPOTIFY_ID
  python -m semantic_spotify.models.recommend --model collaborative_v2 --track-ids ID1 ID2 ID3 --k 20
  python -m semantic_spotify.models.recommend --model collaborative_v2 --list-models

Model contract (any future model can export these files):
  models/<model_name>/embeddings.npy    shape (N, D), track vectors
  models/<model_name>/track_ids.npy     shape (N,), Spotify track IDs
  models/<model_name>/seen_in_train.npy optional boolean eligibility mask

The CLI --model selects a model directory, not a hard-coded architecture.
For custom model storage or scoring, register a ModelAdapter subclass below.
Dependencies: numpy, duckdb (catalog lookup); no PyTorch required.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import duckdb
import numpy as np


@dataclass(frozen=True)
class Hit:
    rank: int
    track_id: str
    track_name: str | None
    artist_name: str | None
    similarity: float


class RetrievalModel(Protocol):
    """Interface for embedding-based models; extend for other scoring systems."""
    track_ids: np.ndarray

    def query(self, track_ids: list[str]) -> np.ndarray: ...
    def search(self, query: np.ndarray, k: int, exclude: set[str]) -> list[tuple[str, float]]: ...
    def eligible(self, track_id: str) -> bool: ...


class EmbeddingModel:
    """Shared artifact format for Item2Vec, audio, text, and fused track embeddings."""

    def __init__(self, model_dir: Path, chunk_size: int = 100_000):
        self.model_dir = model_dir
        self.chunk_size = chunk_size
        for filename in ('embeddings.npy', 'track_ids.npy'):
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(f'Missing {model_dir / filename}')
        self.vectors = np.load(model_dir / 'embeddings.npy', mmap_mode='r')
        self.track_ids = np.load(model_dir / 'track_ids.npy', allow_pickle=False).astype(str)
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.track_ids):
            raise ValueError('embeddings.npy and track_ids.npy have incompatible shapes')
        if self.vectors.shape[1] < 1:
            raise ValueError('Embedding dimension must be positive')
        if len(set(self.track_ids.tolist())) != len(self.track_ids):
            raise ValueError('Duplicate track IDs in model artifacts')
        mask_path = model_dir / 'seen_in_train.npy'
        self.mask = (np.load(mask_path, allow_pickle=False).astype(bool)
                     if mask_path.is_file() else np.ones(len(self.track_ids), dtype=bool))
        if self.mask.shape != (len(self.track_ids),):
            raise ValueError('seen_in_train.npy has incompatible shape')
        self.index = {track_id: i for i, track_id in enumerate(self.track_ids)}

    def eligible(self, track_id: str) -> bool:
        i = self.index.get(track_id)
        return i is not None and bool(self.mask[i])

    def query(self, track_ids: list[str]) -> np.ndarray:
        missing = [tid for tid in track_ids if not self.eligible(tid)]
        if missing:
            raise ValueError(f'No usable embeddings for track ID(s): {", ".join(missing[:8])}')
        idx = [self.index[tid] for tid in track_ids]
        vecs = np.asarray(self.vectors[idx], dtype=np.float32)
        # Match playlist-completion evaluation: mean vectors, then L2 normalize.
        q = vecs.mean(axis=0)
        norm = float(np.linalg.norm(q))
        if norm <= 1e-12:
            raise ValueError('Query embedding has zero norm')
        return q / norm

    def search(self, query: np.ndarray, k: int, exclude: set[str]) -> list[tuple[str, float]]:
        if k <= 0:
            raise ValueError('k must be positive')
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.vectors.shape[1],):
            raise ValueError('Query dimensionality mismatch')
        excluded = {self.index[tid] for tid in exclude if tid in self.index}
        # Chunked exact cosine similarity avoids materializing a giant float64 matrix.
        # Handles non-normalized future embedding exports too.
        best_idx = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float32)
        for start in range(0, len(self.track_ids), self.chunk_size):
            end = min(start + self.chunk_size, len(self.track_ids))
            block = np.asarray(self.vectors[start:end], dtype=np.float32)
            norms = np.linalg.norm(block, axis=1)
            scores = (block @ q) / np.maximum(norms, 1e-12)
            scores[~self.mask[start:end]] = -np.inf
            for i in excluded:
                if start <= i < end:
                    scores[i - start] = -np.inf
            n = min(k, len(scores))
            if not n:
                continue
            local = np.argpartition(scores, -n)[-n:]
            cand_idx = np.concatenate((best_idx, local.astype(np.int64) + start))
            cand_scores = np.concatenate((best_scores, scores[local]))
            keep_n = min(k, len(cand_idx))
            keep = np.argpartition(cand_scores, -keep_n)[-keep_n:]
            best_idx, best_scores = cand_idx[keep], cand_scores[keep]
        order = np.argsort(-best_scores, kind='stable')
        return [(str(self.track_ids[best_idx[i]]), float(best_scores[i]))
                for i in order if np.isfinite(best_scores[i])][:k]


# Adapter registry. Future models with the standard .npy contract require NO edits.
# For an incompatible model type, implement RetrievalModel and register its loader.
MODEL_ADAPTERS = {'embeddings': EmbeddingModel}


def load_model(name: str, models_dir: Path, adapter: str = 'embeddings') -> RetrievalModel:
    # Prevent path traversal; --model is a directory name, not an arbitrary path.
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('--model must be a single model directory name')
    if adapter not in MODEL_ADAPTERS:
        raise ValueError(f'Unknown adapter {adapter!r}; available: {list(MODEL_ADAPTERS)}')
    return MODEL_ADAPTERS[adapter](models_dir / name)


class Catalog:
    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(path)
        self.con = duckdb.connect(database=':memory:')
        self.path = str(path)

    def lookup(self, title: str, artist: str | None = None, limit: int = 15) -> list[dict]:
        # Case-insensitive exact title; artist optional; no accidental arbitrary substring match.
        sql = '''SELECT track_id, track_name, artist_name, has_audio_features
                 FROM read_parquet(?) WHERE lower(track_name) = lower(?)'''
        params: list = [self.path, title]
        if artist:
            sql += ' AND lower(artist_name) = lower(?)'
            params.append(artist)
        sql += ' ORDER BY track_id LIMIT ?'
        params.append(limit)
        rows = self.con.execute(sql, params).fetchall()
        return [dict(zip(('track_id', 'track_name', 'artist_name', 'has_audio_features'), row)) for row in rows]

    def describe(self, ids: list[str]) -> dict[str, tuple[str | None, str | None]]:
        if not ids:
            return {}
        # Parameterized join rather than constructing SQL from user-supplied IDs.
        rows = self.con.execute('''
            SELECT t.track_id, t.track_name, t.artist_name
            FROM read_parquet(?) t JOIN unnest(?::VARCHAR[]) AS q(track_id)
              ON t.track_id = q.track_id
        ''', [self.path, ids]).fetchall()
        return {tid: (title, artist) for tid, title, artist in rows}

    def close(self):
        self.con.close()


def recommend(model: RetrievalModel, catalog: Catalog, ids: list[str], k: int = 20) -> list[Hit]:
    if not ids:
        raise ValueError('At least one query track is required')
    unique_ids = list(dict.fromkeys(ids))
    ranked = model.search(model.query(unique_ids), k=k, exclude=set(unique_ids))
    meta = catalog.describe([tid for tid, _ in ranked])
    return [Hit(rank=n, track_id=tid, track_name=meta.get(tid, (None, None))[0],
                artist_name=meta.get(tid, (None, None))[1], similarity=score)
            for n, (tid, score) in enumerate(ranked, 1)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', help='Model directory name, e.g. collaborative_v2')
    p.add_argument('--models-dir', type=Path, default=Path('models'))
    p.add_argument('--catalog-dir', type=Path, default=Path('data/processed/catalog'))
    p.add_argument('--adapter', default='embeddings', choices=sorted(MODEL_ADAPTERS))
    p.add_argument('--track', help='Exact track title in canonical catalog')
    p.add_argument('--artist', help='Exact artist name (with --track)')
    p.add_argument('--track-id', help='One Spotify track ID')
    p.add_argument('--track-ids', nargs='+', help='Multiple Spotify track IDs (playlist query)')
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--json', action='store_true', help='Print machine-readable JSON')
    p.add_argument('--list-models', action='store_true', help='List directories with standard embedding artifacts')
    args = p.parse_args()

    if args.list_models:
        available = sorted(x.name for x in args.models_dir.iterdir()
                           if x.is_dir() and (x / 'embeddings.npy').is_file()
                           and (x / 'track_ids.npy').is_file()) if args.models_dir.is_dir() else []
        print('\n'.join(available) if available else '(no exported models found)')
        return
    if not args.model:
        p.error('--model is required unless --list-models is specified')
    modes = sum(bool(x) for x in (args.track, args.track_id, args.track_ids))
    if modes != 1:
        p.error('Provide exactly one of --track, --track-id, or --track-ids')
    if args.artist and not args.track:
        p.error('--artist requires --track')
    if args.k <= 0:
        p.error('--k must be positive')

    model = load_model(args.model, args.models_dir, args.adapter)
    catalog = Catalog(args.catalog_dir / 'tracks.parquet')
    try:
        if args.track:
            matches = catalog.lookup(args.track, args.artist)
            if not matches:
                raise ValueError('No exact title/artist match in catalog; try --track-id')
            eligible = [m for m in matches if model.eligible(m['track_id'])]
            if len(eligible) != 1:
                detail = '\n'.join(f"  {m['track_id']} | {m['track_name']} — {m['artist_name']} | eligible={model.eligible(m['track_id'])}" for m in matches)
                raise ValueError(f'Expected one eligible track, found {len(eligible)}. Use --track-id to disambiguate:\n{detail}')
            ids = [eligible[0]['track_id']]
        else:
            ids = [args.track_id] if args.track_id else args.track_ids
        hits = recommend(model, catalog, ids, k=args.k)
        if args.json:
            print(json.dumps({'model': args.model, 'query_track_ids': ids,
                              'recommendations': [h.__dict__ for h in hits]}, indent=2))
        else:
            print(f'Model: {args.model} | query: {", ".join(ids)} | top {len(hits)}')
            print(f'{"#":>3}  {"Similarity":>10}  {"Track":<42}  Artist')
            for h in hits:
                print(f'{h.rank:>3}  {h.similarity:>10.4f}  {(h.track_name or h.track_id)[:42]:<42}  {h.artist_name or ""}')
    finally:
        catalog.close()


if __name__ == '__main__':
    main()
