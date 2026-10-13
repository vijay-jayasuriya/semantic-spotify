"""Train Item2Vec-style track embeddings from canonical MPD playlist edges.

Run from the Semantic Spotify repository root:
  python -m semantic_spotify.models.train_collaborative \
      --catalog-dir data/processed/catalog --output-dir models/collaborative

Dependencies: pip install duckdb numpy torch pyarrow
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import duckdb
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def read_catalog(catalog_dir: Path):
    tracks = catalog_dir / 'tracks.parquet'
    edges = catalog_dir / 'playlist_tracks.parquet'
    for path in (tracks, edges):
        if not path.exists():
            raise FileNotFoundError(path)
    con = duckdb.connect()
    ids = con.execute('SELECT track_id FROM read_parquet(?) ORDER BY track_id', [str(tracks)]).fetchnumpy()['track_id']
    # Query split by playlist ID; train=0..7, val=8, test=9.
    # We use train and validation only. Never train on held-out playlists.
    rows = con.execute('''
        SELECT playlist_id, track_id
        FROM read_parquet(?)
        WHERE track_id IS NOT NULL AND playlist_id IS NOT NULL
        ORDER BY playlist_id, position
    ''', [str(edges)]).fetchnumpy()
    con.close()
    track_ids = np.asarray(ids, dtype=str)
    pid = np.asarray(rows['playlist_id'], dtype=np.int64)
    edge_track_ids = np.asarray(rows['track_id'], dtype=str)
    # Both arrays sorted lexicographically, so searchsorted gives stable IDs.
    tid = np.searchsorted(track_ids, edge_track_ids)
    if np.any(tid >= len(track_ids)) or not np.all(track_ids[tid] == edge_track_ids):
        raise ValueError('playlist_tracks references unknown track IDs')
    return track_ids, pid, tid.astype(np.int32)


def group_playlists(pid: np.ndarray, tid: np.ndarray, split: int | None):
    """Compact unique-track sets per playlist, excluding singletons."""
    cuts = np.r_[0, np.flatnonzero(pid[1:] != pid[:-1]) + 1, len(pid)]
    playlists = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if split is None:
            if int(pid[a]) % 10 >= 8:
                continue
        elif int(pid[a]) % 10 != split:
            continue
        unique = np.unique(tid[a:b])
        if len(unique) >= 2:
            playlists.append(unique)
    return playlists


def training_arrays(playlists: list[np.ndarray], n_tracks: int):
    lengths = np.fromiter((len(p) for p in playlists), dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    flat = np.concatenate(playlists)
    counts = np.bincount(flat, minlength=n_tracks)
    return flat, offsets, counts


def sample_pairs(flat, offsets, n_pairs, rng):
    # Sample playlists proportional to size (not size squared), reducing
    # domination by unusually large playlists.
    lengths = np.diff(offsets)
    choices = rng.choice(len(lengths), size=n_pairs, p=lengths / lengths.sum())
    n = lengths[choices]
    a = rng.integers(0, n)
    b = rng.integers(0, n - 1)
    b += (b >= a)
    start = offsets[choices]
    return flat[start + a].astype(np.int64), flat[start + b].astype(np.int64)


class Item2Vec(nn.Module):
    def __init__(self, n_tracks, dim):
        super().__init__()
        self.target = nn.Embedding(n_tracks, dim)
        self.context = nn.Embedding(n_tracks, dim)
        nn.init.normal_(self.target.weight, std=0.02)
        nn.init.normal_(self.context.weight, std=0.02)

    def forward(self, anchors, positives, negatives):
        u = self.target(anchors)
        v = self.context(positives)
        neg = self.context(negatives)
        pos_logits = (u * v).sum(-1)
        neg_logits = torch.bmm(neg, u.unsqueeze(-1)).squeeze(-1)
        return -(F.logsigmoid(pos_logits) + F.logsigmoid(-neg_logits).sum(-1)).mean()


@torch.no_grad()
def evaluate(model, playlists, seen, k=50, max_playlists=200, seed=123):
    """Warm-start playlist completion: query half, retrieve other half.

    Exact full-catalog ranking; excludes query tracks and untrained tracks.
    Reports micro Recall@K. This evaluates retrieval, not cold-start.
    """
    rng = np.random.default_rng(seed)
    subset = rng.choice(len(playlists), min(max_playlists, len(playlists)), replace=False)
    vectors = F.normalize(model.target.weight + model.context.weight, dim=1).detach().cpu()
    seen_mask = torch.from_numpy(seen)
    hits = total = 0
    for j in subset:
        songs = playlists[j]
        songs = songs[seen[songs]]
        if len(songs) < 2:
            continue
        songs = rng.permutation(songs)
        n_query = max(1, len(songs) // 2)
        query = torch.as_tensor(songs[:n_query], dtype=torch.long)
        heldout = songs[n_query:]
        q = F.normalize(vectors[query].mean(0), dim=0)
        scores = vectors @ q
        scores[~seen_mask] = -float('inf')
        scores[query] = -float('inf')
        top = torch.topk(scores, min(k, int(seen.sum()) - len(query))).indices.numpy()
        hits += np.isin(heldout, top).sum()
        total += len(heldout)
    return float(hits / total) if total else None, int(total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog-dir', type=Path, default=Path('data/processed/catalog'))
    parser.add_argument('--output-dir', type=Path, default=Path('models/collaborative'))
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--pairs-per-epoch', type=int, default=2_000_000)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--negatives', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-playlists', type=int, default=200)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    args = parser.parse_args()
    if args.epochs < 1 or args.pairs_per_epoch < 1 or args.negatives < 1:
        parser.error('epochs, pairs-per-epoch, and negatives must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    print('Loading canonical catalog...', flush=True)
    track_ids, pid, tid = read_catalog(args.catalog_dir)
    train = group_playlists(pid, tid, split=None)
    val = group_playlists(pid, tid, split=8)
    flat, offsets, counts = training_arrays(train, len(track_ids))
    seen = counts > 0
    # Sample negatives from training-observed tracks only.
    # This distribution is fixed across training epochs.
    weights = counts.astype(np.float64) ** 0.75
    weights /= weights.sum()
    print(f'tracks={len(track_ids):,} train_playlists={len(train):,} val_playlists={len(val):,} seen_train_tracks={seen.sum():,} device={device}', flush=True)
    model = Item2Vec(len(track_ids), args.dim).to(device)
    # AdamW dense embedding updates are straightforward but memory intensive.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        anchors, positives = sample_pairs(flat, offsets, args.pairs_per_epoch, rng)
        permutation = rng.permutation(args.pairs_per_epoch)
        total_loss = 0.0
        steps = 0
        for idx in range(0, len(permutation), args.batch_size):
            batch = permutation[idx:idx + args.batch_size]
            a = torch.as_tensor(anchors[batch].astype(np.int64), device=device)
            p = torch.as_tensor(positives[batch].astype(np.int64), device=device)
            negatives = rng.choice(len(track_ids), size=(len(batch), args.negatives), p=weights)
            n = torch.as_tensor(negatives, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(a, p, n)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
        model.eval()
        recall, relevant = evaluate(model, val, seen, k=50, max_playlists=args.eval_playlists, seed=args.seed)
        row = {'epoch': epoch, 'loss': total_loss / steps, 'val_recall_at_50': recall, 'val_relevant_tracks': relevant}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save({'model': model.state_dict(), 'dim': args.dim, 'epoch': epoch}, args.output_dir / 'checkpoint.pt')
    # Keep untrained rows zero so they are never mistaken for meaningful vectors.
    with torch.no_grad():
        emb = F.normalize(model.target.weight + model.context.weight, dim=1).cpu().numpy()
    emb[~seen] = 0.0
    np.save(args.output_dir / 'embeddings.npy', emb)
    np.save(args.output_dir / 'track_ids.npy', track_ids)
    np.save(args.output_dir / 'seen_in_train.npy', seen)
    (args.output_dir / 'metrics.json').write_text(json.dumps({'config': vars(args) | {'catalog_dir': str(args.catalog_dir), 'output_dir': str(args.output_dir)}, 'history': history}, indent=2))
    print(f'Saved embeddings and metadata to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
