import duckdb

con = duckdb.connect()

result = con.execute("""
    SELECT
        track_id,
        artist_name,
        track_name,
        album_name,
        track_popularity,
        danceability,
        energy
    FROM read_parquet(
        'data/interim/tracks_test/tracks.parquet'
    )
    WHERE track_id = '3XAzZZoJeLmIGAGYH7rTBj'
    LIMIT 50
""").fetchdf()

print(result.to_string(index=False))