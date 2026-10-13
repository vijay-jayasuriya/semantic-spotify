import time
from datasets import load_dataset

print("Initializing dataset...", flush=True)
start = time.perf_counter()

ds = load_dataset(
    "devmaxjasper/spotify_public_playlists",
    split="train",
    streaming=True,
    token=True,
)

print(
    f"Dataset initialized in {time.perf_counter() - start:.2f}s",
    flush=True,
)

print("Requesting first record...", flush=True)

start = time.perf_counter()

first = next(iter(ds))

print(
    f"First record received in {time.perf_counter() - start:.2f}s",
    flush=True,
)

print("Playlist:", first.get("name"))