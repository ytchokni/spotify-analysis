"""
Interactive Results Explorer
-----------------------------
Simple script to explore the community detection results.
"""

import json
import pandas as pd
import os
from glob import glob

def load_latest_results():
    """Load the most recent analysis results."""
    results_dir = 'analysis_results'
    
    # Find the latest JSON file
    json_files = glob(os.path.join(results_dir, 'community_analysis_*.json'))
    if not json_files:
        print("No analysis results found. Please run genre_analysis.py first.")
        return None, None, None
    
    latest_json = max(json_files, key=os.path.getmtime)
    
    # Load corresponding CSV files
    # Extract timestamp: community_analysis_20260115_104057.json -> 20260115_104057
    filename = os.path.basename(latest_json)
    timestamp = filename.replace('community_analysis_', '').replace('.json', '')
    mapping_file = os.path.join(results_dir, f'track_community_mapping_{timestamp}.csv')
    summary_file = os.path.join(results_dir, f'community_summary_{timestamp}.csv')
    
    with open(latest_json, 'r') as f:
        data = json.load(f)
    
    mapping_df = pd.read_csv(mapping_file)
    summary_df = pd.read_csv(summary_file)
    
    return data, mapping_df, summary_df

def print_overview(data, summary_df):
    """Print overview of the analysis."""
    print("="*80)
    print("SPOTIFY COMMUNITY DETECTION RESULTS")
    print("="*80)
    print(f"\nAnalysis Timestamp: {data['metadata']['timestamp']}")
    print(f"Total Tracks Analyzed: {data['metadata']['total_tracks']:,}")
    print(f"Total Playlists: {data['metadata']['total_playlists']:,}")
    print(f"Communities Discovered: {data['metadata']['total_communities']}")
    print(f"\nCommunity Size Range: {summary_df['size'].min()} - {summary_df['size'].max():,} tracks")
    print(f"Average Community Size: {summary_df['size'].mean():.0f} tracks")

def explore_community(data, community_rank):
    """Explore a specific community in detail."""
    if community_rank < 1 or community_rank > len(data['communities']):
        print(f"Invalid rank. Please choose between 1 and {len(data['communities'])}")
        return
    
    community = data['communities'][community_rank - 1]
    
    print("\n" + "="*80)
    print(f"COMMUNITY #{community['rank']}")
    print("="*80)
    print(f"Community ID: {community['community_id']}")
    print(f"Size: {community['size']:,} tracks")
    print(f"Unique Playlists: {community['unique_playlists']}")
    print(f"Density: {community['density']:.4f}")
    print(f"Average Shared Playlists: {community['avg_shared_playlists']:.2f}")
    
    print(f"\n{'Top Artists':-^80}")
    for i, artist in enumerate(community['top_artists'][:10], 1):
        print(f"{i:2}. {artist['artist_name']:40} ({artist['track_count']} tracks)")
    
    print(f"\n{'Top Tracks':-^80}")
    for i, track in enumerate(community['top_tracks'][:10], 1):
        print(f"{i:2}. {track['track_name'][:35]:35} - {track['artist_name'][:25]:25}")
        print(f"    ({track['appearances']} playlist appearances)")
    
    print(f"\n{'Playlist Themes':-^80}")
    themes = [t['word'] for t in community['playlist_themes']]
    print(f"{', '.join(themes)}")

def search_track(mapping_df, search_term):
    """Search for a track and show its community."""
    results = mapping_df[
        mapping_df['track_name'].str.contains(search_term, case=False, na=False) |
        mapping_df['artist_name'].str.contains(search_term, case=False, na=False)
    ]
    
    if len(results) == 0:
        print(f"\nNo tracks found matching '{search_term}'")
        return
    
    print(f"\n{'Search Results':-^80}")
    print(f"Found {len(results)} matching track(s):\n")
    
    for idx, row in results.head(20).iterrows():
        print(f"Track: {row['track_name']}")
        print(f"Artist: {row['artist_name']}")
        print(f"Community ID: {row['community_id']}")
        print(f"Track ID: {row['track_id']}")
        print("-" * 80)

def show_menu():
    """Display interactive menu."""
    print("\n" + "="*80)
    print("MENU")
    print("="*80)
    print("1. Show overview")
    print("2. Explore a specific community (by rank)")
    print("3. Search for a track/artist")
    print("4. Show top 10 communities")
    print("5. Export community to CSV")
    print("6. Quit")
    print("="*80)

def export_community(data, mapping_df, community_rank):
    """Export a specific community to CSV."""
    if community_rank < 1 or community_rank > len(data['communities']):
        print(f"Invalid rank. Please choose between 1 and {len(data['communities'])}")
        return
    
    community = data['communities'][community_rank - 1]
    community_id = community['community_id']
    
    # Filter tracks in this community
    community_tracks = mapping_df[mapping_df['community_id'] == community_id]
    
    # Add rank info
    output_file = f"analysis_results/community_{community_rank}_tracks.csv"
    community_tracks.to_csv(output_file, index=False)
    
    print(f"\n✓ Exported {len(community_tracks)} tracks to: {output_file}")

def show_top_communities(summary_df):
    """Show summary of top 10 communities."""
    print("\n" + "="*80)
    print("TOP 10 COMMUNITIES BY SIZE")
    print("="*80)
    print(f"\n{'Rank':<6}{'Size':<8}{'Top Artist':<30}{'Top Themes'}")
    print("-" * 80)
    
    for _, row in summary_df.head(10).iterrows():
        themes = row['top_themes'][:50] if len(row['top_themes']) > 50 else row['top_themes']
        print(f"{row['rank']:<6}{row['size']:<8}{row['top_artist'][:30]:<30}{themes}")

def main():
    """Main interactive loop."""
    print("Loading analysis results...")
    data, mapping_df, summary_df = load_latest_results()
    
    if data is None:
        return
    
    print("✓ Results loaded successfully!")
    
    while True:
        show_menu()
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            print_overview(data, summary_df)
        
        elif choice == '2':
            try:
                rank = int(input("\nEnter community rank (1-" + str(len(data['communities'])) + "): "))
                explore_community(data, rank)
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        elif choice == '3':
            search_term = input("\nEnter search term (track or artist name): ").strip()
            if search_term:
                search_track(mapping_df, search_term)
        
        elif choice == '4':
            show_top_communities(summary_df)
        
        elif choice == '5':
            try:
                rank = int(input("\nEnter community rank to export (1-" + str(len(data['communities'])) + "): "))
                export_community(data, mapping_df, rank)
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        elif choice == '6':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()

