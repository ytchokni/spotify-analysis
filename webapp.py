"""
Spotify Community Detection - Web Explorer
-------------------------------------------
Interactive web application for exploring music communities.
Two modes: Trending Songs Analysis vs All Songs Analysis.

Run: streamlit run webapp.py
Requires: python genre_analysis.py (run first to generate data)
"""

import streamlit as st
import pandas as pd
import json
import os
from glob import glob
import plotly.express as px
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Spotify Community Explorer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1DB954;
        text-align: center;
        margin-bottom: 2rem;
    }
    .community-header {
        background: linear-gradient(90deg, #1DB954 0%, #191414 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'analysis_results')


@st.cache_data
def load_track_growth():
    """Load pre-computed track growth data."""
    filepath = os.path.join(RESULTS_DIR, 'track_growth_latest.csv')
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame()


@st.cache_data
def load_track_plays():
    """Load pre-computed track play counts."""
    filepath = os.path.join(RESULTS_DIR, 'track_plays_latest.csv')
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame()


@st.cache_data
def load_playlists():
    """Load pre-computed playlists data."""
    filepath = os.path.join(RESULTS_DIR, 'playlists_latest.csv')
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame()


@st.cache_data
def load_playlist_tracks():
    """Load pre-computed playlist-track relationships."""
    filepath = os.path.join(RESULTS_DIR, 'playlist_tracks_latest.csv')
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame()


def load_analysis_data(analysis_type='trending'):
    """Load community analysis results for specified type (trending or all).
    
    Note: No caching - analysis_type changes should reload data immediately.
    """
    suffix = f"_{analysis_type}_"

    # Find matching JSON files
    pattern = f'community_analysis{suffix}*.json'
    json_files = glob(os.path.join(RESULTS_DIR, pattern))

    if not json_files:
        return None, None, None

    latest_json = max(json_files, key=os.path.getmtime)

    with open(latest_json, 'r') as f:
        data = json.load(f)

    # Find corresponding files with same suffix
    mapping_pattern = f'track_community_mapping{suffix}*.csv'
    summary_pattern = f'community_summary{suffix}*.csv'

    mapping_files = glob(os.path.join(RESULTS_DIR, mapping_pattern))
    summary_files = glob(os.path.join(RESULTS_DIR, summary_pattern))

    if not mapping_files or not summary_files:
        return None, None, None

    mapping_df = pd.read_csv(max(mapping_files, key=os.path.getmtime))
    summary_df = pd.read_csv(max(summary_files, key=os.path.getmtime))

    return data, mapping_df, summary_df


def load_umap_embedding(analysis_type='trending'):
    """Load pre-computed UMAP embedding for specified type.
    
    Note: No caching - analysis_type changes should reload data immediately.
    """
    suffix = f"_{analysis_type}_"
    pattern = f'umap_embedding{suffix}*.csv'
    umap_files = glob(os.path.join(RESULTS_DIR, pattern))

    if not umap_files:
        return None

    latest_umap = max(umap_files, key=os.path.getmtime)
    return pd.read_csv(latest_umap)


@st.cache_data
def merge_growth_with_mapping(_mapping_df, _track_growth_df):
    """Merge growth data with community mapping.
    
    Only merges growth metrics, not track names (to avoid column conflicts).
    """
    if len(_track_growth_df) == 0:
        mapping_df = _mapping_df.copy()
        mapping_df['growth_pct_per_24h'] = 0
        mapping_df['is_trending'] = False
        return mapping_df
    
    # Only select growth columns to avoid duplicates
    growth_cols = _track_growth_df[['track_id', 'growth_pct_per_24h', 'is_trending']].copy()
    
    mapping_df = _mapping_df.merge(growth_cols, on='track_id', how='left')
    mapping_df['growth_pct_per_24h'] = mapping_df['growth_pct_per_24h'].fillna(0)
    mapping_df['is_trending'] = mapping_df['is_trending'].fillna(False)

    return mapping_df


def calculate_top_playlists_for_community(community_id, mapping_df, track_growth_df, playlists_df, playlist_tracks_df):
    """Calculate which playlists have the highest overlap with a given community.

    Uses pre-loaded data to find playlists containing tracks from this community,
    and ranks them by overlap percentage. Also includes growth metrics.
    """
    # Get track IDs in this community
    community_tracks = set(mapping_df[mapping_df['community_id'] == community_id]['track_id'])

    if len(community_tracks) == 0:
        return None

    if len(playlists_df) == 0 or len(playlist_tracks_df) == 0:
        return None

    # Filter playlist_tracks to only tracks in this community
    community_pt = playlist_tracks_df[playlist_tracks_df['track_id'].isin(community_tracks)]

    # Count tracks from community per playlist
    tracks_per_playlist = community_pt.groupby('playlist_id')['track_id'].nunique().reset_index()
    tracks_per_playlist.columns = ['playlist_id', 'tracks_from_community']

    # Filter to playlists with at least 2 tracks from community
    tracks_per_playlist = tracks_per_playlist[tracks_per_playlist['tracks_from_community'] >= 2]

    if len(tracks_per_playlist) == 0:
        return None

    # Merge with playlist info
    df = tracks_per_playlist.merge(playlists_df, on='playlist_id', how='inner')

    # Filter by total_tracks
    df = df[(df['total_tracks'] >= 5) & (df['total_tracks'] <= 500)]

    if len(df) == 0:
        return None

    # Calculate overlap percentage
    df['overlap_pct'] = (df['tracks_from_community'] / df['total_tracks'] * 100)

    # Sort and limit
    df = df.sort_values(['overlap_pct', 'tracks_from_community'], ascending=[False, False]).head(20)

    # Calculate average growth per playlist
    if len(track_growth_df) > 0:
        # Get all tracks in these playlists
        result_playlist_ids = set(df['playlist_id'])
        result_pt = playlist_tracks_df[playlist_tracks_df['playlist_id'].isin(result_playlist_ids)]

        # Merge with growth data
        result_pt = result_pt.merge(
            track_growth_df[['track_id', 'growth_pct_per_24h']],
            on='track_id',
            how='left'
        )
        result_pt['growth_pct_per_24h'] = result_pt['growth_pct_per_24h'].fillna(0)

        # Calculate average growth per playlist
        playlist_growth = result_pt.groupby('playlist_id')['growth_pct_per_24h'].mean().reset_index()
        playlist_growth.columns = ['playlist_id', 'avg_growth_pct']

        df = df.merge(playlist_growth, on='playlist_id', how='left')
        df['avg_growth_pct'] = df['avg_growth_pct'].fillna(0)
    else:
        df['avg_growth_pct'] = 0

    return df


def render_header():
    st.markdown('<div class="main-header">🎵 Spotify Music Trends</div>', unsafe_allow_html=True)
    st.markdown("---")


def render_track_landscape(data, mapping_df, track_growth_df, analysis_type='trending'):
    """Render track-based 2D visualization with bubble sizes."""
    st.header("🗺️ Track Landscape")

    mode_text = "Trending Songs" if analysis_type == 'trending' else "All Songs"
    st.markdown(f"**Analysis Mode:** {mode_text}")

    umap_df = load_umap_embedding(analysis_type)

    if umap_df is None or len(umap_df) == 0:
        st.error(f"Track visualization not found for '{analysis_type}' mode. Run `python genre_analysis.py` first.")
        return

    # Merge with track growth data if available
    if 'track_id' in umap_df.columns and len(track_growth_df) > 0:
        growth_cols = track_growth_df[['track_id', 'growth_pct_per_24h', 'is_trending']].copy()
        umap_df = umap_df.merge(growth_cols, on='track_id', how='left')
        umap_df['growth_pct_per_24h'] = umap_df['growth_pct_per_24h'].fillna(0)
        umap_df['is_trending'] = umap_df['is_trending'].fillna(False)
    else:
        umap_df['growth_pct_per_24h'] = 0
        umap_df['is_trending'] = False

    plot_df = umap_df.copy()

    # Get community info
    community_map = {c['community_id']: c for c in data['communities']}
    plot_df['community_name'] = plot_df['community_id'].apply(
        lambda x: f"#{community_map[x]['rank']}: {community_map[x].get('name', 'Unknown')[:30]}"
        if x in community_map and x != -1 else "Uncategorized"
    )

    # Controls
    col1, col2 = st.columns(2)

    with col1:
        color_by = st.selectbox("Color by:", ["Community", "Growth Rate (%)", "Track Popularity"])

    with col2:
        max_communities = min(30, len(data['communities']))
        show_top_n = st.slider("Show top N communities:", 3, max_communities, min(15, max_communities))

    # Filter to top N communities
    top_communities = sorted(
        [c['community_id'] for c in data['communities']],
        key=lambda x: community_map[x]['rank']
    )[:show_top_n]

    plot_df = plot_df[plot_df['community_id'].isin(top_communities)].copy()

    st.subheader(f"Showing {len(plot_df):,} tracks in {show_top_n} communities")

    # Use log_playlist_count for bubble size if available
    size_col = 'log_playlist_count' if 'log_playlist_count' in plot_df.columns else None

    if color_by == "Community":
        fig = px.scatter(
            plot_df, x='umap_x', y='umap_y', color='community_name',
            size=size_col,
            hover_data=['track_name', 'artist_name', 'playlist_count', 'growth_pct_per_24h'],
            title=f"Track Landscape ({mode_text}) - By Community",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    elif color_by == "Growth Rate (%)":
        fig = px.scatter(
            plot_df, x='umap_x', y='umap_y', color='growth_pct_per_24h',
            size=size_col,
            hover_data=['track_name', 'artist_name', 'playlist_count', 'community_name'],
            title=f"Track Landscape ({mode_text}) - By Growth Rate (% per 24h)",
            color_continuous_scale='RdYlGn'
        )
    else:  # Track Popularity
        fig = px.scatter(
            plot_df, x='umap_x', y='umap_y', color='playlist_count',
            size=size_col,
            hover_data=['track_name', 'artist_name', 'growth_pct_per_24h', 'community_name'],
            title=f"Track Landscape ({mode_text}) - By Popularity (Playlist Count)",
            color_continuous_scale='Blues'
        )

    fig.update_layout(height=700, xaxis_title="", yaxis_title="", showlegend=True)
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)

    st.plotly_chart(fig, width="stretch")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tracks Shown", f"{len(plot_df):,}")
    with col2:
        st.metric("Communities", show_top_n)
    with col3:
        avg_growth = plot_df['growth_pct_per_24h'].mean() if 'growth_pct_per_24h' in plot_df.columns else 0
        st.metric("Avg Growth/24h", f"+{avg_growth:.2f}%")


def render_community_explorer(data, mapping_df, summary_df, track_growth_df, track_plays_df, playlists_df, playlist_tracks_df, analysis_type='trending'):
    """Render community explorer with track details and representative playlists."""
    st.header("🔍 Community Explorer")

    mode_text = "Trending Songs" if analysis_type == 'trending' else "All Songs"
    st.markdown(f"**Analysis Mode:** {mode_text}")

    # Merge growth data
    mapping_with_growth = merge_growth_with_mapping(mapping_df, track_growth_df)

    # Community growth stats
    community_growth = mapping_with_growth.groupby('community_id').agg({
        'growth_pct_per_24h': ['sum', 'mean', 'count'],
        'is_trending': 'sum'
    }).reset_index()
    community_growth.columns = ['community_id', 'total_growth_pct', 'avg_growth_pct', 'track_count', 'trending_count']
    community_growth['trending_pct'] = (community_growth['trending_count'] / community_growth['track_count'] * 100).round(1)

    # Merge with summary
    summary_with_growth = summary_df.merge(community_growth, on='community_id', how='left')
    summary_with_growth = summary_with_growth.sort_values('avg_growth_pct', ascending=False)

    st.subheader("🚀 Communities Ranked by Average Growth")

    cols_to_show = ['rank', 'name', 'size', 'avg_growth_pct', 'top_artist']
    if all(c in summary_with_growth.columns for c in cols_to_show):
        growth_display = summary_with_growth.head(20)[cols_to_show].copy()
        growth_display.columns = ['Rank', 'Theme', 'Tracks', 'Avg Growth %/24h', 'Top Artist']
        growth_display['Theme'] = growth_display['Theme'].str[:40]

        # Use column_config to format the percentage column while keeping numeric for sorting
        st.dataframe(
            growth_display,
            hide_index=True,
            width="stretch",
            column_config={
                'Avg Growth %/24h': st.column_config.NumberColumn(
                    'Avg Growth %/24h',
                    format='%.2f%%'
                )
            }
        )
    else:
        st.warning("Summary data format issue")

    st.markdown("---")

    # Community selector - sort by rank number
    communities_sorted_by_rank = sorted(
        data['communities'],
        key=lambda c: c['rank']
    )

    community_options = [
        f"#{c['rank']}: {c.get('name', 'Unknown')[:40]} ({c['size']} tracks)"
        for c in communities_sorted_by_rank
    ]

    selected = st.selectbox("Select a community to explore:",
                           options=range(len(community_options)),
                           format_func=lambda x: community_options[x])

    community = communities_sorted_by_rank[selected]
    community_id = community['community_id']

    # Get community growth info
    comm_growth_info = community_growth[community_growth['community_id'] == community_id]
    avg_growth = comm_growth_info['avg_growth_pct'].values[0] if len(comm_growth_info) > 0 else 0
    trending_count = int(comm_growth_info['trending_count'].values[0]) if len(comm_growth_info) > 0 else 0

    st.markdown(f"""
    <div class="community-header">
        <h2>Community #{community['rank']}: {community.get('name', 'Unknown')[:50]}</h2>
        <p>Size: {community['size']:,} tracks | Avg Growth: +{avg_growth:.2f}%/24h | Trending: {trending_count} tracks</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tracks", f"{community['size']:,}")
    with col2:
        st.metric("Avg Growth/24h", f"+{avg_growth:.2f}%")
    with col3:
        st.metric("Trending Tracks", trending_count)
    with col4:
        st.metric("Density", f"{community['density']:.4f}")

    # Get community tracks with growth
    community_tracks = mapping_with_growth[mapping_with_growth['community_id'] == community_id].copy()

    # Calculate growth percentiles for this community
    if 'growth_pct_per_24h' in community_tracks.columns and len(community_tracks) > 0:
        community_tracks['growth_percentile'] = community_tracks['growth_pct_per_24h'].rank(pct=True) * 100
    else:
        community_tracks['growth_percentile'] = 0

    # Calculate top playlists by overlap with this community (using pre-loaded data)
    top_playlists_df = calculate_top_playlists_for_community(
        community_id, mapping_df, track_growth_df, playlists_df, playlist_tracks_df
    )

    # Get plays for community tracks from pre-loaded data
    if len(track_plays_df) > 0:
        community_tracks = community_tracks.merge(
            track_plays_df[['track_id', 'plays']],
            on='track_id',
            how='left'
        )
        community_tracks['plays'] = community_tracks['plays'].fillna(0)
    else:
        community_tracks['plays'] = 0

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚀 Top Growing Tracks")
        if len(community_tracks) > 0 and 'track_name' in community_tracks.columns and 'artist_name' in community_tracks.columns:
            top_growing = community_tracks.nlargest(10, 'growth_pct_per_24h')[
                ['track_name', 'artist_name', 'growth_pct_per_24h', 'growth_percentile']
            ].drop_duplicates(subset=['track_name', 'artist_name']).head(10).copy()
            top_growing.columns = ['Track', 'Artist', 'Growth %/24h', 'Percentile']

            # Keep numeric for sorting
            st.dataframe(
                top_growing,
                hide_index=True,
                width="stretch",
                column_config={
                    'Growth %/24h': st.column_config.NumberColumn(
                        'Growth %/24h',
                        format='%.2f%%'
                    ),
                    'Percentile': st.column_config.NumberColumn(
                        'Percentile',
                        format='%.0f'
                    )
                }
            )
        else:
            st.info("No tracks in this community")

    with col2:
        st.subheader("🎤 Top Songs by Absolute Plays")
        if len(community_tracks) > 0 and 'track_name' in community_tracks.columns and 'artist_name' in community_tracks.columns and 'plays' in community_tracks.columns:
            top_songs = community_tracks.nlargest(10, 'plays')[
                ['track_name', 'artist_name', 'plays']
            ].drop_duplicates(subset=['track_name', 'artist_name']).head(10).copy()
            top_songs.columns = ['Track', 'Artist', 'Plays']

            st.dataframe(
                top_songs,
                hide_index=True,
                width="stretch",
                column_config={
                    'Plays': st.column_config.NumberColumn(
                        'Plays',
                        format=',.0f'
                    )
                }
            )
        else:
            st.info("No tracks with play counts in this community")

    st.markdown("---")

    # Add Top Growing Playlists with 5k+ followers section
    st.subheader("🔥 Top Growing Playlists (5k+ followers)")
    if top_playlists_df is not None and len(top_playlists_df) > 0:
        # Filter for playlists with 5k+ followers
        high_follower_playlists = top_playlists_df[top_playlists_df['followers'] >= 5000].copy()

        if len(high_follower_playlists) > 0:
            # Sort by average growth rate
            high_follower_playlists = high_follower_playlists.sort_values('avg_growth_pct', ascending=False)

            # Calculate growth percentiles
            high_follower_playlists['growth_percentile'] = high_follower_playlists['avg_growth_pct'].rank(pct=True) * 100

            # Select top 15 playlists
            top_growing_playlists = high_follower_playlists.head(15)[
                ['playlist_name', 'followers', 'tracks_from_community', 'avg_growth_pct', 'growth_percentile']
            ].copy()
            top_growing_playlists.columns = ['Playlist', 'Followers', 'Tracks from Community', 'Avg Growth %/24h', 'Growth Percentile']

            st.dataframe(
                top_growing_playlists,
                hide_index=True,
                width="stretch",
                column_config={
                    'Followers': st.column_config.NumberColumn(
                        'Followers',
                        format=',.0f'
                    ),
                    'Tracks from Community': st.column_config.NumberColumn(
                        'Tracks from Community',
                        format=',.0f'
                    ),
                    'Avg Growth %/24h': st.column_config.NumberColumn(
                        'Avg Growth %/24h',
                        format='%.2f%%'
                    ),
                    'Growth Percentile': st.column_config.NumberColumn(
                        'Growth Percentile',
                        format='%.0f'
                    )
                }
            )
        else:
            st.info("No playlists with 5k+ followers in this community")
    else:
        st.info("No playlist data available")

    # Add representative playlists section
    if top_playlists_df is not None and len(top_playlists_df) > 0:
        st.markdown("---")
        st.subheader("📋 Most Representative Playlists")
        st.caption("Playlists with the highest proportion of tracks from this community")

        # Remove duplicate playlists by name, keeping the first occurrence
        display_playlists = top_playlists_df.drop_duplicates(subset=['playlist_name'], keep='first').head(15).copy()

        # Calculate growth percentiles
        if 'avg_growth_pct' in display_playlists.columns:
            display_playlists['growth_percentile'] = display_playlists['avg_growth_pct'].rank(pct=True) * 100
        else:
            display_playlists['growth_percentile'] = 0

        # Select and rename columns
        display_playlists = display_playlists[['playlist_name', 'tracks_from_community', 'total_tracks',
                                                'overlap_pct', 'avg_growth_pct', 'growth_percentile', 'followers']].copy()
        display_playlists.columns = ['Playlist', 'Tracks from Community', 'Total Tracks', 'Overlap %',
                                     'Avg Growth %/24h', 'Growth Percentile', 'Followers']
        display_playlists['Playlist'] = display_playlists['Playlist'].str[:50]

        # Keep numeric for proper sorting
        st.dataframe(
            display_playlists,
            hide_index=True,
            width="stretch",
            height=400,
            column_config={
                'Overlap %': st.column_config.NumberColumn(
                    'Overlap %',
                    format='%.1f%%'
                ),
                'Avg Growth %/24h': st.column_config.NumberColumn(
                    'Avg Growth %/24h',
                    format='%.2f%%'
                ),
                'Growth Percentile': st.column_config.NumberColumn(
                    'Growth Percentile',
                    format='%.0f'
                ),
                'Followers': st.column_config.NumberColumn(
                    'Followers',
                    format=',.0f'
                ),
                'Tracks from Community': st.column_config.NumberColumn(
                    'Tracks from Community',
                    format=',.0f'
                ),
                'Total Tracks': st.column_config.NumberColumn(
                    'Total Tracks',
                    format=',.0f'
                )
            }
        )


def main():
    render_header()

    # Sidebar
    st.sidebar.title("🎛️ Controls")

    # Analysis mode selector (this is the main toggle)
    st.sidebar.markdown("### Analysis Mode")
    analysis_type = st.sidebar.radio(
        "View communities from:",
        ["trending", "all"],
        format_func=lambda x: "🔥 Trending Songs" if x == "trending" else "📊 All Songs",
        help="Switch between trending-only analysis and all-songs analysis"
    )

    # Load data based on selection
    data, mapping_df, summary_df = load_analysis_data(analysis_type)
    track_growth_df = load_track_growth()

    # Load pre-calculated static data
    track_plays_df = load_track_plays()
    playlists_df = load_playlists()
    playlist_tracks_df = load_playlist_tracks()

    if data is None:
        st.error(f"No analysis results found for '{analysis_type}' mode. Run `python genre_analysis.py` first.")
        return

    # Check if pre-calculated data is available
    if len(playlists_df) == 0 or len(playlist_tracks_df) == 0:
        st.warning("Pre-calculated data not found. Run `python extract_data.py` first for best performance.")

    # Get counts from the analysis metadata
    metadata = data.get('metadata', {})
    total_tracks = metadata.get('total_tracks', len(mapping_df) if mapping_df is not None else 0)
    total_playlists = metadata.get('total_playlists', 0)
    total_communities = len(data.get('communities', []))

    # Get initial (pre-pruning) counts
    initial_tracks = metadata.get('initial_tracks', total_tracks)
    initial_playlists = metadata.get('initial_playlists', total_playlists)

    # Show analysis stats in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Analysis Stats")

    mode_label = "Trending Songs" if analysis_type == "trending" else "All Songs"
    st.sidebar.info(f"""
**{mode_label} Analysis:**

*Database totals (before filtering):*
- {initial_tracks:,} total tracks
- {initial_playlists:,} total playlists

*After filtering/clustering:*
- {total_tracks:,} tracks clustered
- {total_playlists:,} playlists analyzed
- {total_communities} communities found
    """)

    st.sidebar.markdown("---")

    # Page navigation
    page = st.sidebar.radio("Go to:", ["Track Landscape", "Community Explorer"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Info")
    st.sidebar.caption(f"""
    **Current mode:** {mode_label}
    
    Toggle the analysis mode above to switch between:
    - **Trending:** Top 10% fastest growing songs
    - **All:** Complete dataset analysis
    """)

    # Render selected page
    if page == "Track Landscape":
        render_track_landscape(data, mapping_df, track_growth_df, analysis_type)
    elif page == "Community Explorer":
        render_community_explorer(data, mapping_df, summary_df, track_growth_df, track_plays_df, playlists_df, playlist_tracks_df, analysis_type)


if __name__ == "__main__":
    main()
