import csv
import io
import time

import requests
from huggingface_hub import get_token

URL = (
    "https://huggingface.co/datasets/"
    "devmaxjasper/spotify_public_playlists/"
    "resolve/main/playlists.csv"
)

token = get_token()

headers = {
    "Authorization": f"Bearer {token}"
} if token else {}

start = time.perf_counter()

print("Connecting to Hugging Face...", flush=True)

with requests.get(
    URL,
    headers=headers,
    stream=True,
    timeout=(30, 120),
) as response:

    response.raise_for_status()

    print(
        f"Connected in {time.perf_counter() - start:.2f}s",
        flush=True,
    )

    response.raw.decode_content = True

    text_stream = io.TextIOWrapper(
        response.raw,
        encoding="utf-8",
        newline="",
    )

    reader = csv.DictReader(text_stream)

    print("Reading first playlist...", flush=True)

    first = next(reader)

    print(
        f"First playlist received in "
        f"{time.perf_counter() - start:.2f}s",
        flush=True,
    )

    print("Playlist:", first["name"])
    print("Playlist ID:", first["playlist_id"])