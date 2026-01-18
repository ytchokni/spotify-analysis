"""
Data Extraction Script
----------------------
Pre-computes all growth metrics and trending data for the webapp.
Run this before launching the webapp to eliminate loading times.

Usage: python extract_data.py
"""

import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'spotify.db')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'analysis_results')
MIN_STREAMS = 100000  # Minimum streams to be included
MIN_GROWTH_PCT = 3.0  # Minimum growth % per 24h to be included

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_track_growth():
    """Calculate track growth metrics from database snapshots."""
    print("Extracting track growth data...")
    print(f"  Filter: minimum {MIN_STREAMS:,} streams")
    conn = sqlite3.connect(DB_PATH)

    query = f"""
    WITH track_snapshots AS (
        SELECT
            id,
            name,
            artist_id,
            plays,
            fetched_at,
            ROW_NUMBER() OVER (PARTITION BY id ORDER BY fetched_at ASC) as snapshot_num,
            ROW_NUMBER() OVER (PARTITION BY id ORDER BY fetched_at DESC) as rev_snapshot_num
        FROM tracks
        WHERE plays IS NOT NULL AND plays > 0
    ),
    first_last AS (
        SELECT
            t1.id,
            t1.name as track_name,
            t1.artist_id,
            t1.plays as first_plays,
            t1.fetched_at as first_fetched,
            t2.plays as last_plays,
            t2.fetched_at as last_fetched
        FROM track_snapshots t1
        JOIN track_snapshots t2 ON t1.id = t2.id
        WHERE t1.snapshot_num = 1 AND t2.rev_snapshot_num = 1
        AND t1.fetched_at != t2.fetched_at
        AND t1.plays >= {MIN_STREAMS}
    ),
    with_artist AS (
        SELECT
            fl.id as track_id,
            fl.track_name,
            a.name as artist_name,
            fl.first_plays,
            fl.last_plays,
            fl.first_fetched,
            fl.last_fetched,
            (fl.last_plays - fl.first_plays) as total_growth,
            CAST((julianday(fl.last_fetched) - julianday(fl.first_fetched)) * 24 AS REAL) as hours_elapsed,
            CASE
                WHEN fl.first_plays > 0
                THEN ((fl.last_plays - fl.first_plays) * 100.0 / fl.first_plays)
                ELSE 0
            END as growth_pct,
            ROW_NUMBER() OVER (PARTITION BY fl.id ORDER BY a.name) as artist_rn
        FROM first_last fl
        JOIN artists a ON fl.artist_id = a.id
        WHERE fl.last_plays > fl.first_plays
    )
    SELECT
        track_id,
        track_name,
        artist_name,
        first_plays,
        last_plays,
        first_fetched,
        last_fetched,
        total_growth,
        hours_elapsed,
        growth_pct,
        CASE
            WHEN hours_elapsed > 0
            THEN growth_pct / hours_elapsed * 24
            ELSE 0
        END as growth_pct_per_24h
    FROM with_artist
    WHERE artist_rn = 1
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Calculate log ratio for trending detection
    # log_ratio = log(last_plays / first_plays) normalized per 24h
    df['log_ratio'] = np.log(df['last_plays'] / df['first_plays'])
    df['log_ratio_per_24h'] = df.apply(
        lambda row: row['log_ratio'] / row['hours_elapsed'] * 24 if row['hours_elapsed'] > 0 else 0,
        axis=1
    )

    # Verify no duplicates
    duplicates = df['track_id'].duplicated().sum()
    if duplicates > 0:
        print(f"  Warning: Found {duplicates} duplicate track_ids, removing...")
        df = df.drop_duplicates(subset=['track_id'], keep='first')

    print(f"  Found {len(df):,} unique tracks with growth data (>= {MIN_STREAMS:,} streams)")
    
    # Filter to only tracks with significant growth
    before_filter = len(df)
    df = df[df['growth_pct_per_24h'] >= MIN_GROWTH_PCT]
    print(f"  After filtering (>= {MIN_GROWTH_PCT}% growth): {len(df):,} tracks ({before_filter - len(df):,} removed)")
    
    return df


def extract_playlist_growth():
    """Calculate playlist growth metrics from database snapshots."""
    print("Extracting playlist growth data...")
    conn = sqlite3.connect(DB_PATH)

    query = """
    WITH playlist_snapshots AS (
        SELECT
            id,
            name,
            followers,
            fetched_at,
            ROW_NUMBER() OVER (PARTITION BY id ORDER BY fetched_at ASC) as snapshot_num,
            ROW_NUMBER() OVER (PARTITION BY id ORDER BY fetched_at DESC) as rev_snapshot_num
        FROM playlists
        WHERE followers IS NOT NULL AND followers > 0
    ),
    first_last AS (
        SELECT
            p1.id,
            p1.name as playlist_name,
            p1.followers as first_followers,
            p1.fetched_at as first_fetched,
            p2.followers as last_followers,
            p2.fetched_at as last_fetched
        FROM playlist_snapshots p1
        JOIN playlist_snapshots p2 ON p1.id = p2.id
        WHERE p1.snapshot_num = 1 AND p2.rev_snapshot_num = 1
        AND p1.fetched_at != p2.fetched_at
    )
    SELECT
        id as playlist_id,
        playlist_name,
        first_followers,
        last_followers,
        first_fetched,
        last_fetched,
        (last_followers - first_followers) as total_growth,
        CAST((julianday(last_fetched) - julianday(first_fetched)) * 24 AS REAL) as hours_elapsed,
        CASE
            WHEN first_followers > 0
            THEN ((last_followers - first_followers) * 100.0 / first_followers)
            ELSE 0
        END as growth_pct,
        CASE
            WHEN first_followers > 0 AND julianday(last_fetched) - julianday(first_fetched) > 0
            THEN ((last_followers - first_followers) * 100.0 / first_followers) / ((julianday(last_fetched) - julianday(first_fetched)) * 24) * 24
            ELSE 0
        END as growth_pct_per_24h
    FROM first_last
    WHERE last_followers > first_followers
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Remove duplicates
    duplicates = df['playlist_id'].duplicated().sum()
    if duplicates > 0:
        print(f"  Warning: Found {duplicates} duplicate playlist_ids, removing...")
        df = df.drop_duplicates(subset=['playlist_id'], keep='first')

    # Calculate log ratio for trending detection
    # log_ratio = log(last_followers / first_followers) normalized per 24h
    df['log_ratio'] = np.log(df['last_followers'] / df['first_followers'])
    df['log_ratio_per_24h'] = df.apply(
        lambda row: row['log_ratio'] / row['hours_elapsed'] * 24 if row['hours_elapsed'] > 0 else 0,
        axis=1
    )

    print(f"  Found {len(df):,} unique playlists with growth data")
    return df


def extract_track_plays():
    """Extract latest play counts for all tracks."""
    print("Extracting track play counts...")
    conn = sqlite3.connect(DB_PATH)

    query = """
    WITH latest_tracks AS (
        SELECT
            id,
            name,
            plays,
            artist_id,
            ROW_NUMBER() OVER(PARTITION BY id ORDER BY fetched_at DESC) as rn
        FROM tracks
        WHERE plays IS NOT NULL
    )
    SELECT
        id as track_id,
        name as track_name,
        plays,
        artist_id
    FROM latest_tracks
    WHERE rn = 1
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"  Found {len(df):,} tracks with play counts")
    return df


def extract_playlists():
    """Extract latest snapshot of all playlists."""
    print("Extracting playlist data...")
    conn = sqlite3.connect(DB_PATH)

    query = """
    WITH latest_playlists AS (
        SELECT
            id,
            name,
            total_tracks,
            followers,
            ROW_NUMBER() OVER(PARTITION BY id ORDER BY fetched_at DESC) as rn
        FROM playlists
    )
    SELECT
        id as playlist_id,
        name as playlist_name,
        total_tracks,
        followers
    FROM latest_playlists
    WHERE rn = 1
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"  Found {len(df):,} playlists")
    return df


def extract_playlist_tracks():
    """Extract distinct playlist-track relationships."""
    print("Extracting playlist-track relationships...")
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT DISTINCT playlist_id, track_id
    FROM playlist_tracks
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"  Found {len(df):,} playlist-track relationships")
    return df


def calculate_trending(track_growth_df, playlist_growth_df, percentile=90):
    """Mark top 10% as trending based on log ratio per 24h (internal), but keep % growth for display."""
    print(f"Calculating trending (top {100-percentile}% by log ratio)...")

    # Track trending - use log_ratio_per_24h for detection
    if len(track_growth_df) > 0:
        threshold = track_growth_df['log_ratio_per_24h'].quantile(percentile / 100)
        track_growth_df['is_trending'] = track_growth_df['log_ratio_per_24h'] >= threshold
        trending_tracks = track_growth_df['is_trending'].sum()
        # Show both log ratio threshold and corresponding % growth for context
        avg_pct_growth = track_growth_df[track_growth_df['is_trending']]['growth_pct_per_24h'].mean()
        print(f"  Trending tracks: {trending_tracks:,} (log ratio threshold: {threshold:.4f}, avg % growth: {avg_pct_growth:.2f}%/24h)")
    else:
        track_growth_df['is_trending'] = False

    # Playlist trending - use log_ratio_per_24h for detection
    if len(playlist_growth_df) > 0:
        threshold = playlist_growth_df['log_ratio_per_24h'].quantile(percentile / 100)
        playlist_growth_df['is_trending'] = playlist_growth_df['log_ratio_per_24h'] >= threshold
        trending_playlists = playlist_growth_df['is_trending'].sum()
        avg_pct_growth = playlist_growth_df[playlist_growth_df['is_trending']]['growth_pct_per_24h'].mean()
        print(f"  Trending playlists: {trending_playlists:,} (log ratio threshold: {threshold:.4f}, avg % growth: {avg_pct_growth:.2f}%/24h)")
    else:
        playlist_growth_df['is_trending'] = False

    return track_growth_df, playlist_growth_df


def main():
    """Run full data extraction."""
    print("=" * 60)
    print("DATA EXTRACTION FOR WEBAPP")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"Min streams filter: {MIN_STREAMS:,}")
    print(f"Min growth filter: {MIN_GROWTH_PCT}%/24h")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract growth data (already filtered by MIN_GROWTH_PCT)
    track_growth_df = extract_track_growth()
    playlist_growth_df = extract_playlist_growth()

    # Get the set of track IDs that meet the growth threshold
    growing_track_ids = set(track_growth_df['track_id'])
    print(f"\nFiltering other data to {len(growing_track_ids):,} growing tracks...")

    # Extract static data for webapp (no DB connection needed at runtime)
    track_plays_df = extract_track_plays()
    playlists_df = extract_playlists()
    playlist_tracks_df = extract_playlist_tracks()

    # Filter track_plays to only include growing tracks
    before_plays = len(track_plays_df)
    track_plays_df = track_plays_df[track_plays_df['track_id'].isin(growing_track_ids)]
    print(f"  Track plays: {before_plays:,} -> {len(track_plays_df):,}")

    # Filter playlist_tracks to only include growing tracks
    before_pt = len(playlist_tracks_df)
    playlist_tracks_df = playlist_tracks_df[playlist_tracks_df['track_id'].isin(growing_track_ids)]
    print(f"  Playlist-tracks: {before_pt:,} -> {len(playlist_tracks_df):,}")

    # Filter playlists to only those containing growing tracks
    playlists_with_growing = set(playlist_tracks_df['playlist_id'])
    before_playlists = len(playlists_df)
    playlists_df = playlists_df[playlists_df['playlist_id'].isin(playlists_with_growing)]
    print(f"  Playlists: {before_playlists:,} -> {len(playlists_df):,}")

    # Calculate trending
    track_growth_df, playlist_growth_df = calculate_trending(
        track_growth_df, playlist_growth_df
    )

    # Save files
    print("\nSaving extracted data...")

    track_file = os.path.join(OUTPUT_DIR, f'track_growth_{timestamp}.csv')
    track_growth_df.to_csv(track_file, index=False)
    print(f"  Saved: {track_file}")

    playlist_file = os.path.join(OUTPUT_DIR, f'playlist_growth_{timestamp}.csv')
    playlist_growth_df.to_csv(playlist_file, index=False)
    print(f"  Saved: {playlist_file}")

    # Also save as "latest" for easy loading
    track_growth_df.to_csv(os.path.join(OUTPUT_DIR, 'track_growth_latest.csv'), index=False)
    playlist_growth_df.to_csv(os.path.join(OUTPUT_DIR, 'playlist_growth_latest.csv'), index=False)
    print(f"  Saved: track_growth_latest.csv, playlist_growth_latest.csv")

    # Save static data files
    track_plays_df.to_csv(os.path.join(OUTPUT_DIR, 'track_plays_latest.csv'), index=False)
    playlists_df.to_csv(os.path.join(OUTPUT_DIR, 'playlists_latest.csv'), index=False)
    playlist_tracks_df.to_csv(os.path.join(OUTPUT_DIR, 'playlist_tracks_latest.csv'), index=False)
    print(f"  Saved: track_plays_latest.csv, playlists_latest.csv, playlist_tracks_latest.csv")

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  - Tracks with >= {MIN_GROWTH_PCT}% growth: {len(track_growth_df):,}")
    print(f"  - Trending tracks (top 10%): {track_growth_df['is_trending'].sum():,}")
    print(f"  - Playlists with growth: {len(playlist_growth_df):,}")
    print(f"  - Trending playlists (top 10%): {playlist_growth_df['is_trending'].sum():,}")
    print(f"  - Track plays extracted: {len(track_plays_df):,}")
    print(f"  - Playlists extracted: {len(playlists_df):,}")
    print(f"  - Playlist-track relationships: {len(playlist_tracks_df):,}")

    # Show top growing tracks
    print(f"\nTop 5 fastest growing tracks (% growth/24h):")
    top5 = track_growth_df.nlargest(5, 'growth_pct_per_24h')
    for _, row in top5.iterrows():
        print(f"  {row['track_name'][:40]} - {row['artist_name'][:20]}: +{row['growth_pct_per_24h']:.2f}%/24h")

    print(f"\nNow run: streamlit run webapp.py")


if __name__ == '__main__':
    main()
