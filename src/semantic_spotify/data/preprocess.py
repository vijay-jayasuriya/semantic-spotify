import json


def parse_playlist(row):
    raw = row["raw_data"]

    playlist = json.loads(raw)

    # Handle double-encoded JSON
    if isinstance(playlist, str):
        playlist = json.loads(playlist)

    if not isinstance(playlist, dict):
        raise ValueError(
            f"Expected playlist dictionary, got {type(playlist)}"
        )

    playlist_id = playlist["id"]
    tracks_data = playlist.get("tracks") or {}
    items = tracks_data.get("items") or []

    total_tracks = tracks_data.get("total", 0)
    available_tracks = len(items)

    playlist_record = {
        "playlist_id": playlist_id,
        "name": playlist.get("name"),
        "description": playlist.get("description"),
        "followers_count": (
            playlist.get("followers") or {}
        ).get("total", 0),
        "total_tracks": total_tracks,
        "available_tracks": available_tracks,
        "coverage": (
            available_tracks / total_tracks
            if total_tracks > 0
            else 1.0
        ),
        "is_complete": (
            available_tracks == total_tracks
            and tracks_data.get("offset", 0) == 0
        ),
    }
    
    track_records = []

    for position, item in enumerate(items):
        track = item.get("track") or {}

        track_id = track.get("id")

        if not track_id:
            continue

        track_records.append({
            "playlist_id": playlist_id,
            "track_id": track_id,
            "position": tracks_data.get("offset", 0) + position,
            "added_at": item.get("added_at"),
        })

    return playlist_record, track_records