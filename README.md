# Semantic Spotify

**Learning song representations from playlist behavior — and building toward multimodal, natural-language music retrieval.**

Semantic Spotify is an end-to-end machine learning engineering project for music discovery. It transforms large, heterogeneous music datasets into a canonical song catalog, trains collaborative track embeddings with PyTorch, and evaluates retrieval against reproducible playlist-completion benchmarks. The longer-term objective is to align collaborative signals, numerical audio features, and text metadata in a shared retrieval system that can recommend music from songs, playlists, or natural-language descriptions.

**Current status:** The data pipeline, collaborative Item2Vec training, evaluation harness, and model-selectable nearest-neighbor CLI are implemented. Last.fm API access has been verified; bulk tag enrichment, audio/text encoders, and multimodal retrieval are **planned, not yet implemented**.

## Results at a glance

**Held-out playlist completion, 996 evaluated validation playlists, seed 42, top 50 recommendations**

| Metric | Popularity baseline | Item2Vec v2 (selected) | Item2Vec v3 |
|:--|--:|--:|--:|
| **Warm micro Recall@50** | 2.37% | **7.62%** | 5.86% |
| Warm macro Recall@50 | 2.52% | **9.38%** | 7.75% |
| Warm macro NDCG@50 | 0.0231 | **0.0844** | 0.0720 |
| Warm micro Recall@10 | 0.64% | **2.19%** | 1.99% |
| Full micro Recall@50 | 2.08% | **6.71%** | 5.15% |

**Item2Vec v2 delivers 3.22× the warm Recall@50 of a global-popularity recommender** on the same validation queries and candidate eligibility rules. The benchmark ranks against approximately **600K training-observed tracks**, not a prefiltered shortlist.

> **Interpretation:** Recall measures recovery of songs actually present in a held-out playlist, not subjective musical quality. The v2/v3 comparison is a controlled evaluation on the same sampled playlists; the results are not a claim of production recommendation performance. The test split remains reserved for final model selection and reporting.

### Data scale

| Asset | Size / coverage |
|:--|--:|
| Source audio-feature records | 56,277,664 rows |
| Unique IDs in source audio-feature dataset | 39,283,820 |
| Ingested playlists | **100,000** |
| Playlist–track occurrences | **6,685,101** |
| Unique canonical tracks | **679,889** |
| Tracks with matched audio features | **360,639 (53.04%)** |
| Playlist–track occurrences with matched audio features | **67.91%** |
| Tracks observed in collaborative training | **599,933 (88.24% of catalog)** |
| Collaborative embedding dimension | **128** |

**Dataset scope matters:** Although the source is named the *Spotify Million Playlist Dataset Challenge*, the Hugging Face mirror ingested for this experiment contains **100,000 playlists**, not the complete one-million-playlist corpus. All counts and results here refer to the ingested subset.

## Why this project

Music similarity is not a single signal. Songs may sound alike, be played by similar audiences, or share descriptors such as *dreamy*, *experimental*, or *upbeat*. These relationships overlap but are not interchangeable.

Semantic Spotify is designed around three complementary representations:

1. **Collaborative:** Which tracks appear together in user-curated playlists?
2. **Audio features:** What measurable characteristics does a track have, and can those features generalize to songs absent from training playlists?
3. **Text semantics:** Which genres, moods, styles, and natural-language concepts describe the music?

The implemented system focuses on the first representation while establishing the data contracts and evaluation methodology for the others.

## System architecture

```text
Hugging Face: Spotify track analysis         Hugging Face: MPD JSON slices
    ~56.3M source rows                              100K playlists
               |                                           |
               v                                           v
      Parquet ingestion                            Playlist ingestion
      DuckDB / PyArrow                             PyArrow / Parquet
               |                                           |
               +--------------------+----------------------+
                                    |
                                    v
                         Canonical catalog builder
                     ID joins • metadata • null handling
                       referential-integrity checks
                                    |
                                    v
                          data/processed/catalog/
                  tracks • artists • playlists • edges
                                    |
                     playlist-level train / val / test
                                    |
                                    v
                         Item2Vec / SGNS training
                   sampled co-occurrences + negatives
                       PyTorch on Apple MPS / CPU
                                    |
                                    v
                         128-dimensional vectors
                         + eligibility / ID mapping
                                    |
                   +----------------+----------------+
                   |                                 |
                   v                                 v
          Evaluation harness                  Model-selectable CLI
       Recall@K • NDCG@K                    cosine nearest neighbors
       popularity comparison                 song / multi-song queries

    NEXT: Last.fm tags → text encoder; audio features → audio encoder
                     → multimodal alignment / retrieval
```

## Engineering decisions

### 1. Reproducible, disk-aware data processing

- **Columnar Parquet artifacts** separate raw ingestion, intermediate data, and model-ready catalog tables.
- **DuckDB** performs large joins and aggregations without loading the full 56M-row audio dataset into a pandas DataFrame.
- Catalog construction uses bounded DuckDB settings (including a configurable memory limit, thread count, and temporary spill directory), supporting development on a memory-constrained laptop.
- The canonical catalog preserves **all MPD track IDs**, even when no matching audio features are available; missing modalities are represented explicitly rather than silently dropping tracks.
- Referential-integrity checks validate playlist–track relationships and unique playlist positions.
- Playlist-level splits prevent songs from the same playlist being divided across training and validation examples.

### 2. Collaborative representation learning

The implemented baseline is **Item2Vec-style skip-gram with negative sampling (SGNS)**. Each training example samples two distinct songs from the same training playlist as a positive pair. The model learns separate target and context embedding tables and optimizes:

\[
\mathcal{L} = -\log\sigma(\mathbf{u}_i^\top\mathbf{v}_j)
- \sum_{n=1}^{K}\log\sigma(-\mathbf{u}_i^\top\mathbf{v}_{n})
\]

Current runs use **128-dimensional vectors**, **five negative samples per positive**, and the **Adam** optimizer. Playlist sampling is weighted by playlist length rather than the number of all possible within-playlist pairs, limiting domination by unusually long playlists. Inference exports normalized **target + context** embeddings, with untrained tracks explicitly masked out.

This is a **transductive** collaborative model: it cannot produce meaningful embeddings for songs never observed in training playlists. Cold-start support is a planned role for the audio/text encoders.

### 3. Evaluation designed around retrieval

The evaluation harness holds out **entire playlists**, then splits eligible tracks within each validation playlist into a visible query set and a hidden target set. It averages the visible track embeddings, scores candidate tracks by cosine similarity, excludes visible tracks, and measures recovery of hidden songs.

It reports:

- **Micro Recall@10/@50:** total recovered hidden occurrences divided by total eligible hidden occurrences.
- **Macro Recall@10/@50:** mean per-playlist recall.
- **Macro NDCG@10/@50:** ranking quality with greater credit for relevant songs near the top.
- **Full Recall:** counts hidden tracks absent from collaborative training as misses, exposing coverage limitations.
- **Popularity baseline:** recommends globally frequent training tracks under the same evaluation setup.

Validation uses deterministic playlist sampling and query/holdout construction with a fixed seed. **Warm metrics exclude tracks unseen in training; full metrics account for them as unretrievable.** The held-out test split is not used for routine iteration.

### 4. Model-independent retrieval interface

`recommend.py` accepts a **model directory name** instead of hardcoding a checkpoint. Models that export the standard track-embedding artifact contract can be loaded without modifying the command-line workflow:

```text
models/<model_name>/
├── embeddings.npy       # [num_tracks, embedding_dim]
├── track_ids.npy        # Spotify track IDs aligned with embedding rows
└── seen_in_train.npy    # optional eligibility mask
```

The current adapter supports exact cosine retrieval, query-track exclusion, chunked scoring, catalog metadata resolution, JSON output, and mean-pooled multi-track queries. A model-adapter registry provides an extension point for future architectures. **Natural-language encoding is not implemented yet.**

## Reproducing the pipeline

**Environment:** Python 3.11; development and training have been performed on an Apple Silicon Mac using PyTorch's MPS backend. CPU and CUDA device options are also exposed by the training script; these have not been benchmarked here.

From the repository root, create an environment and install the project plus required libraries:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install duckdb numpy pandas pyarrow huggingface_hub torch requests
```

The editable-install command assumes the repository has Python packaging metadata (for example, `pyproject.toml`). Dependencies are listed explicitly to make the core workflow clear; refer to the repository's dependency configuration if it differs.

### A. Ingest datasets

```bash
# MPD JSON slices: the available mirror currently yields 100K playlists.
python -m semantic_spotify.data.ingest_mpd \
    --output-dir data/interim/mpd

# Large audio-feature source; choose paths explicitly for reproducibility.
python -m semantic_spotify.data.ingest_songs \
    --raw-dir data/raw/track_analysis \
    --output-dir data/interim/tracks_test
```

> **Filename note:** The audio ingestion entry point has appeared as `ingest_songs.py` in the working project and `ingest_tracks.py` in supplied source files. Use the module name present in your checkout (`semantic_spotify.data.ingest_songs` or `semantic_spotify.data.ingest_tracks`). The source's CLI supports `--raw-dir`, `--output-dir`, and `--repo-id`.

### B. Build the canonical catalog

```bash
python -m semantic_spotify.data.build_catalog \
    --mpd-dir data/interim/mpd \
    --audio-candidates data/interim/tracks_test/tracks.parquet \
    --output-dir data/processed/catalog \
    --memory-limit 4GB \
    --threads 2
```

The catalog builder emits the following main artifacts:

```text
data/processed/catalog/
├── tracks.parquet               # one row per Spotify track ID
├── track_artists.parquet        # track-to-artist relationships
├── playlists.parquet            # playlist metadata and split
├── playlist_tracks.parquet      # playlist membership and position
├── mpd_track_metadata.parquet   # consolidated MPD metadata
└── catalog_report.json         # counts and integrity checks
```

### C. Train Item2Vec

The strongest **completed** collaborative model in the reported 1,000-playlist comparison is `collaborative_v2`:

```bash
python -m semantic_spotify.models.train_collaborative \
    --catalog-dir data/processed/catalog \
    --output-dir models/collaborative_v2 \
    --epochs 5 \
    --pairs-per-epoch 2000000 \
    --batch-size 2048 \
    --eval-playlists 200
```

Default settings include `--dim 128`, `--negatives 5`, `--lr 0.002`, and `--seed 42`. The training script automatically selects MPS, CUDA, or CPU where available.

**Checkpoint caveat:** The current implementation overwrites `checkpoint.pt` at each epoch and exports `embeddings.npy` **only after the final epoch**. Best-validation checkpoint retention, optimizer-state resumption, and early stopping are planned improvements; they should not be assumed to exist today.

### D. Evaluate against popularity

```bash
python -m semantic_spotify.models.evaluate_collaborative \
    --catalog-dir data/processed/catalog \
    --model-dir models/collaborative_v2 \
    --eval-playlists 1000 \
    --seed 42
```

Outputs a JSON report at `models/collaborative_v2/evaluation.json`. Use the **same playlist count, seed, catalog, and candidate rules** when comparing models. The 1,000-playlist sample yielded 996 evaluable playlists in the reported run.

### E. Inspect recommendations

```bash
# Similar songs; exact catalog title and artist matching.
python -m semantic_spotify.models.recommend \
    --model collaborative_v2 \
    --track "Everything In Its Right Place" \
    --artist "Radiohead" \
    --k 20

# Disambiguate releases using an exact Spotify track ID.
python -m semantic_spotify.models.recommend \
    --model collaborative_v2 \
    --track-id 0ug5NqcwcFR2xrfTkc7k8e \
    --k 20

# Mean-pooled query from multiple tracks.
python -m semantic_spotify.models.recommend \
    --model collaborative_v2 \
    --track-ids TRACK_ID_1 TRACK_ID_2 TRACK_ID_3 \
    --k 20

# Discover compatible checkpoints or emit JSON.
python -m semantic_spotify.models.recommend --list-models
python -m semantic_spotify.models.recommend \
    --model collaborative_v2 --track-id TRACK_ID --json
```

Exact title/artist lookup may return multiple Spotify IDs or no match due to punctuation and release variants. Use `--track-id` for unambiguous queries; fuzzy lookup is a potential future enhancement.

## Experiment history and lessons learned

| Experiment | Training configuration | Warm micro Recall@50 | Evaluation sample |
|:--|:--|--:|:--|
| Item2Vec v1 | 2 × 200K positive pairs | 2.15% | 200 playlists |
| **Item2Vec v2** | **5 × 2M positive pairs** | **7.62%** | **996 playlists** |
| Item2Vec v3 | 10 × 5M positive pairs | 5.86% | 996 playlists |
| Popularity baseline | No learned embeddings | 2.37% | 996 playlists |

The initial 200-playlist evaluation for v2 yielded **7.04%** Recall@50; the larger fixed sample yielded **7.62%**. Numbers from different playlist samples should not be compared as though they were paired experiments.

**A useful negative result:** Scaling v3 to 50M positive pairs reduced retrieval quality. Its training loss decreased from **2.9126** to **1.5697** over 10 epochs, while its 200-playlist validation Recall@50 fell from **7.67% after epoch 1** to **5.14% after epoch 10**. The final v3 checkpoint was worse than v2 on the fixed 996-playlist evaluation.

This motivates validation-based checkpoint selection, early stopping, and investigation of the mismatch between the SGNS objective and mean-pooled cosine retrieval. It is **consistent with overfitting or objective mismatch**, but the logs do not independently establish which mechanism dominates.

### Qualitative findings

- A Radiohead query retrieved indie/alternative neighbors including alt-J, Wilco, Modest Mouse, and other Radiohead tracks.
- A Taylor Swift query retrieved mainstream-pop and same-artist neighbors.
- Two Spotify IDs for *Style* by Taylor Swift produced different neighborhoods and had a measured embedding cosine similarity of **0.6220**, highlighting the distinction between **recording identity**, **playlist context**, and **acoustic similarity**.

These are sanity checks, not substitutes for quantitative evaluation.

## Repository layout

```text
semantic-spotify/
├── src/semantic_spotify/
│   ├── data/
│   │   ├── ingest_mpd.py
│   │   ├── ingest_songs.py        # may be named ingest_tracks.py
│   │   └── build_catalog.py
│   └── models/
│       ├── train_collaborative.py
│       ├── evaluate_collaborative.py
│       └── recommend.py
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       └── catalog/
├── models/
│   ├── collaborative/
│   ├── collaborative_v2/
│   └── collaborative_v3/
└── README.md
```

Large downloaded datasets, DuckDB working files, and trained weight artifacts may not be appropriate for version control. Keep credentials and generated data out of commits; document how to regenerate them.

## Roadmap

| Stage | Status | Next engineering milestone |
|:--|:--|:--|
| Source ingestion and canonical catalog | **Implemented** | Data-quality regression tests and provenance tracking |
| Collaborative Item2Vec | **Implemented** | Best-checkpoint selection and controlled training experiments |
| Playlist-completion evaluation | **Implemented** | Playlist-length and popularity-sliced metrics; final held-out test |
| Modular song retrieval CLI | **Implemented** | Fuzzy metadata matching and faster indexed retrieval |
| Last.fm tag enrichment | **API verified; pipeline pending** | Rate-limited, resumable, cached `track.getTopTags` ingestion |
| Audio-feature encoder | **Planned** | Predict collaborative-space representations for audio-covered tracks |
| Text encoder | **Planned** | Encode Last.fm tags and playlist titles for semantic search |
| Multimodal retrieval | **Planned** | Compare alignment, late fusion, and cold-start retrieval |
| Serving / deployment | **Planned** | ANN index, service API, latency and memory benchmarks |

### Proposed multimodal evaluation

The next models should be evaluated **against the same retrieval task and candidate set**, with additional tests for cold-start tracks, tag/text retrieval, and modality coverage. Shared-space alignment alone is not evidence of improved recommendations; each modality must demonstrate incremental value through ablations and held-out metrics.

## Limitations and responsible use

- **Playlist data is observational:** co-occurrence captures listener/curator behavior, not necessarily acoustic resemblance or causal preference.
- **Catalog coverage is incomplete:** only 53.04% of canonical tracks currently have matched numerical audio features, and the collaborative model cannot embed training-unseen tracks.
- **Track IDs are not unique compositions:** different Spotify IDs may refer to different releases or closely related recordings. The current model treats them as distinct items.
- **Popularity and selection bias:** curated playlists may overrepresent particular artists, genres, and listening communities.
- **Evaluation proxy:** held-out playlist membership is imperfect ground truth for subjective music discovery.
- **Last.fm tags:** public user-generated labels may be sparse, ambiguous, or noisy. The API key has been tested successfully, but no large-scale enrichment results are claimed here.
- **Data rights:** this project is an independent research/portfolio effort. Review the applicable dataset licenses, Spotify/Last.fm terms, and redistribution restrictions before sharing raw datasets, metadata, or trained artifacts.

## Tech stack

**Python 3.11 · PyTorch · NumPy · DuckDB · PyArrow/Parquet · Hugging Face Hub · Apple Metal Performance Shaders (MPS)**

## Project scope

This repository documents a working collaborative recommendation baseline and the engineering infrastructure required to expand it into a multimodal retrieval system. The emphasis is on **large-scale data processing, explicit artifact contracts, reproducible experiments, retrieval-oriented evaluation, and honest analysis of failure modes**—not on presenting unimplemented features as finished products.
