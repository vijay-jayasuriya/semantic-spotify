import json
from huggingface_hub import HfApi, hf_hub_download

DATASET_ID = "jaxliu/Spotify_Million_Playlist_Dataset_Challenge"

# List JSON files in the dataset repository
api = HfApi()
files = api.list_repo_files(
    repo_id=DATASET_ID,
    repo_type="dataset",
)

json_files = [f for f in files if f.endswith(".json")]

print(f"Found {len(json_files)} JSON files")

if not json_files:
    raise RuntimeError("No JSON files found")

# Download just the first JSON file
filename = json_files[0]
print(f"\nInspecting: {filename}")

local_path = hf_hub_download(
    repo_id=DATASET_ID,
    filename=filename,
    repo_type="dataset",
)

# Load JSON
with open(local_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Inspect structure
print("\nTop-level keys:")
print(list(data.keys()))

# Spotify Million Playlist Dataset files typically contain
# a list of playlists under the "playlists" key.
playlists = data.get("playlists", [])

print(f"\nPlaylists in file: {len(playlists)}")

if playlists:
    playlist = playlists[0]

    print("\nFirst playlist metadata:")
    for key, value in playlist.items():
        if key != "tracks":
            print(f"{key}: {value}")

    print("\nFirst 3 tracks:")
    for track in playlist.get("tracks", [])[:3]:
        print(json.dumps(track, indent=2))