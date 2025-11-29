import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
import re

# Load your merged IRA data
df = pd.read_parquet('data/processed/ira_merged_tweets.parquet')

print(f"Loaded {len(df)} tweets from {df['external_author_id'].nunique()} users")

# Extract hashtags from content column
def extract_hashtags(text):
    """Extract hashtags from tweet text using regex"""
    if pd.isna(text):
        return []
    return re.findall(r'#(\w+)', text.lower())

df['hashtag_list'] = df['content'].apply(extract_hashtags)

# Filter to only tweets with hashtags
df_hashtags = df[df['hashtag_list'].apply(len) > 0].copy()
print(f"Tweets with hashtags: {len(df_hashtags)}")

# Create time windows (e.g., 6-hour windows)
df_hashtags['time_window'] = pd.to_datetime(df_hashtags['publish_date']).dt.floor('6H')

# Build co-occurrence edges
# For each (hashtag, time_window), find all users who used it
# Connect all pairs of users who used same hashtag in same window

print("\nBuilding co-occurrence network...")

edge_counts = defaultdict(int)

# Group by hashtag and time window
for (hashtag, time_window), group in df_hashtags.explode('hashtag_list').groupby(['hashtag_list', 'time_window']):
    users = group['external_author_id'].unique()
    
    # Only create edges if 2+ users used this hashtag in this window
    if len(users) >= 2:
        # Create all pairs of users
        for user1, user2 in combinations(users, 2):
            if user1 != user2:  # No self-loops
                # Sort to make undirected (user1, user2) same as (user2, user1)
                edge = tuple(sorted([user1, user2]))
                edge_counts[edge] += 1

print(f"Found {len(edge_counts)} unique user pairs with coordinated hashtag use")

# Convert to dataframe with edge weights
cooccurrence_edges = pd.DataFrame([
    {'user1': edge[0], 'user2': edge[1], 'weight': count}
    for edge, count in edge_counts.items()
])

# Filter to edges with minimum coordination (e.g., 3+ shared hashtag uses)
min_coordination = 3
cooccurrence_edges = cooccurrence_edges[cooccurrence_edges['weight'] >= min_coordination]

print(f"\n=== CO-OCCURRENCE NETWORK STATS ===")
print(f"Total edges (min {min_coordination} coordinated actions): {len(cooccurrence_edges)}")
print(f"Unique users with coordination: {len(set(cooccurrence_edges['user1']) | set(cooccurrence_edges['user2']))}")
print(f"Average coordination per edge: {cooccurrence_edges['weight'].mean():.1f}")
print(f"Max coordination: {cooccurrence_edges['weight'].max()}")

# Save
output_path = 'data/processed/ira_cooccurrence_edges.csv'
cooccurrence_edges.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

# Optional: Combine with retweet edges
try:
    retweet_edges = pd.read_csv('data/processed/ira_retweet_edges.csv')
    
    # Convert retweet edges to same format (undirected, with weight)
    retweet_edges['edge'] = retweet_edges.apply(
        lambda row: tuple(sorted([row['retweeter_id'], row['original_author_id']])), 
        axis=1
    )
    retweet_edge_counts = retweet_edges.groupby('edge').size().reset_index(name='weight')
    retweet_edge_counts[['user1', 'user2']] = pd.DataFrame(
        retweet_edge_counts['edge'].tolist(), 
        index=retweet_edge_counts.index
    )
    retweet_edge_counts = retweet_edge_counts[['user1', 'user2', 'weight']]
    retweet_edge_counts['edge_type'] = 'retweet'
    
    # Add edge type to cooccurrence
    cooccurrence_edges['edge_type'] = 'cooccurrence'
    
    # Combine
    combined_edges = pd.concat([retweet_edge_counts, cooccurrence_edges], ignore_index=True)
    
    # Aggregate weights if same edge appears in both
    combined_edges = combined_edges.groupby(['user1', 'user2']).agg({
        'weight': 'sum',
        'edge_type': lambda x: ','.join(sorted(set(x)))
    }).reset_index()
    
    combined_path = 'data/processed/ira_combined_edges.csv'
    combined_edges.to_csv(combined_path, index=False)
    
    print(f"\n=== COMBINED NETWORK (RETWEET + CO-OCCURRENCE) ===")
    print(f"Total edges: {len(combined_edges)}")
    print(f"Unique users: {len(set(combined_edges['user1']) | set(combined_edges['user2']))}")
    print(f"Saved to: {combined_path}")
    
except FileNotFoundError:
    print("\nNo retweet edges found, using co-occurrence only")