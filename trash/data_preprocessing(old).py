import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_user_features(df):
    df = pd.DataFrame(df)
    if df.empty:
        logger.error("extract_user_features received empty DataFrame")
        return df
    
    if 'external_author_id' not in df.columns:
        logger.error("extract_user_features: 'external_author_id' column missing")
        return pd.DataFrame()
    
    # Aggregate by user
    user_features = df.groupby('external_author_id').agg({
        'tweet_id': 'count',  # total_tweets
        'retweet': 'mean',    # avg_retweet_count
        'followers': 'mean',  # avg_followers
        'account_type': 'first',
        'publish_date': ['min', 'max']
    }).reset_index()
    
    # Flatten multi-level column names
    user_features.columns = ['external_author_id', 'total_tweets', 'avg_retweet_count', 
                             'avg_followers', 'account_type', 'first_tweet', 'last_tweet']
    
    # Calculate derived metrics
    user_features['account_age_days'] = (user_features['last_tweet'] - user_features['first_tweet']).dt.days
    user_features['tweet_frequency'] = user_features['total_tweets'] / (user_features['account_age_days'] + 1)  # +1 to avoid division by 0
    user_features['retweet_ratio'] = (df.groupby('external_author_id')['retweet'].apply(lambda x: (x > 0).sum() / len(x))).values
    
    logger.info(f"Extracted features for {len(user_features)} users")
    return user_features


def identify_core_coordinated_accounts(user_features, retweet_threshold=10, frequency_threshold=5):
    """
    TODO: Identify the "core" coordinated network
    
    Core accounts should have:
    - High average retweet counts (> retweet_threshold)
    - High tweet frequency (> frequency_threshold per day)
    
    Return: 
    - core_user_ids (list)
    - user_features with new 'is_core' boolean column
    """
    user_features = pd.DataFrame(user_features)
    if user_features.empty:
        logger.error("identify_core_coordinated_accounts received empty DataFrame")
        return [], user_features

    # Define core criteria
    user_features['is_core'] = (
        (user_features['avg_retweet_count'] > retweet_threshold) &
        (user_features['tweet_frequency'] > frequency_threshold) 
    )
    core_user_ids = user_features.loc[user_features['is_core'], 'external_author_id'].tolist()
    return core_user_ids, user_features
    


def sample_organic_network(df, coordinated_user_ids, target_size=None):
    """
    TODO: Sample organic (non-IRA) users for comparison
    
    If target_size is None, use len(coordinated_user_ids) for 1:1 comparison
    
    Requirements:
    - Users NOT in coordinated_user_ids
    - Match follower distribution to coordinated users
    - Same time period as coordinated tweets
    - Target sample size: target_size users
    
    Hint: Use follower quantiles to match distributions
    
    Return: 
    - organic_user_ids (list)
    - organic_df (filtered dataframe)
    """

    df = pd.DataFrame(df)
    if df.empty:
        logger.error("sample_organic_network received empty DataFrame")
        return [], pd.DataFrame()
    
    if target_size is None:
        target_size = len(coordinated_user_ids)
    
    organic_candidates = df[~df['external_author_id'].isin(coordinated_user_ids)]
    coordinated_followers = df[df['external_author_id'].isin(coordinated_user_ids)]['followers']
    quantiles = coordinated_followers.quantile([0.25, 0.5, 0.75]).values

        # Get follower counts for organic candidates (mean per user)
    organic_user_followers = organic_candidates.groupby('external_author_id')['followers'].mean()

    # Define bins based on coordinated quantiles
    bins = [0] + list(quantiles) + [float('inf')]
    bin_labels = ['low', 'medium_low', 'medium_high', 'high']

    # Assign each organic user to a follower bin
    organic_user_followers_binned = pd.cut(organic_user_followers, bins=bins, labels=bin_labels)

    # Calculate coordinated user distribution across bins
    coordinated_user_followers = df[df['external_author_id'].isin(coordinated_user_ids)].groupby('external_author_id')['followers'].mean()
    coordinated_binned = pd.cut(coordinated_user_followers, bins=bins, labels=bin_labels)
    coordinated_bin_counts = coordinated_binned.value_counts()

    # Sample proportionally from each bin
    sampled_user_ids = []
    for bin_label in bin_labels:
        num_in_bin = coordinated_bin_counts.get(bin_label, 0)
        proportion = num_in_bin / len(coordinated_user_ids)
        target_from_bin = int(proportion * target_size)
        
        organic_in_bin = organic_user_followers_binned[organic_user_followers_binned == bin_label].index.tolist()
        
        if len(organic_in_bin) >= target_from_bin:
            sampled_from_bin = np.random.choice(organic_in_bin, size=target_from_bin, replace=False)
        else:
            sampled_from_bin = organic_in_bin
            logger.warning(f"Not enough organic users in {bin_label} bin, taking all {len(organic_in_bin)}")
        
        sampled_user_ids.extend(sampled_from_bin)

    # Filter by same time period as coordinated tweets
    coord_date_range = df[df['external_author_id'].isin(coordinated_user_ids)]['publish_date']
    min_date, max_date = coord_date_range.min(), coord_date_range.max()
    organic_df = df[(df['external_author_id'].isin(sampled_user_ids)) & 
                    (df['publish_date'] >= min_date) & 
                    (df['publish_date'] <= max_date)]

    organic_user_ids = list(set(sampled_user_ids))
    logger.info(f"Sampled {len(organic_user_ids)} organic users matching coordinated distribution")

    return organic_user_ids, organic_df

    
    


def prepare_network_data(df, coordinated_ids, organic_ids):
    """
    TODO: Prepare final datasets for network analysis
    
    Steps:
    1. Create coordinated_df: filter to coordinated_ids only
    2. Create organic_df: filter to organic_ids only
    3. Add 'network_type' column to each ('coordinated' or 'organic')
    4. For NETWORK ANALYSIS, we need:
       - userid/author
       - followers
       - retweet (influence metric)
       - account info
    
    Return:
    - coordinated_df
    - organic_df
    - user_summary (aggregated metrics per user for both networks)
    """
    df = pd.DataFrame(df)
    if df.empty:
        logger.error("prepare_network_data received empty DataFrame")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # coordinated and organic dataframes
    coordinated_df = df[df['external_author_id'].isin(coordinated_ids)].copy()
    coordinated_df['network_type'] = 'coordinated'
    organic_df = df[df['external_author_id'].isin(organic_ids)].copy()
    organic_df['network_type'] = 'organic'

    # Create user summary
    combined_df = pd.concat([coordinated_df, organic_df], ignore_index=True)
    user_summary = combined_df.groupby(['external_author_id', 'network_type']).agg({
        'tweet_id': 'count',
        'retweet': 'mean',
        'followers': 'mean',
        'account_type': 'first'
    }).reset_index()
    user_summary.columns = ['external_author_id', 'network_type', 'total_tweets', 'avg_retweet_count', 'avg_followers', 'account_type']
    return coordinated_df, organic_df, user_summary




def validate_network_samples(coordinated_df, organic_df):
    """
    TODO: Check that samples are comparable
    
    Print/verify:
    - Sample sizes match expectations
    - Follower distributions are similar (compare means, medians)
    - Date ranges overlap
    - Both have retweet data
    
    Return: Boolean (True if validation passes)
    """
    coordinated_df = pd.DataFrame(coordinated_df)
    organic_df = pd.DataFrame(organic_df)
    if coordinated_df.empty or organic_df.empty:
        logger.error("validate_network_samples received empty DataFrame(s)")
        return False
    
    valid = True
    
    # Sample sizes
    coord_size = coordinated_df['external_author_id'].nunique()
    organ_size = organic_df['external_author_id'].nunique()
    logger.info(f"Coordinated users: {coord_size}, Organic users: {organ_size}")
    if coord_size != organ_size:
        logger.warning("Sample sizes do not match")
        valid = False
    
    # Follower distributions
    coord_followers = coordinated_df.groupby('external_author_id')['followers'].mean()
    organ_followers = organic_df.groupby('external_author_id')['followers'].mean()
    logger.info(f"Coordinated followers - Mean: {coord_followers.mean()}, Median: {coord_followers.median()}")
    logger.info(f"Organic followers - Mean: {organ_followers.mean()}, Median: {organ_followers.median()}")
    
    # Date ranges
    coord_dates = coordinated_df['publish_date']
    organ_dates = organic_df['publish_date']
    coord_min, coord_max = coord_dates.min(), coord_dates.max()
    organ_min, organ_max = organ_dates.min(), organ_dates.max()
    logger.info(f"Coordinated date range: {coord_min} to {coord_max}")
    logger.info(f"Organic date range: {organ_min} to {organ_max}")
    
    if coord_max < organ_min or organ_max < coord_min:
        logger.warning("Date ranges do not overlap")
        valid = False
    
    # Retweet data presence
    if coordinated_df['retweet'].isnull().all():
        logger.warning("Coordinated data has no retweet information")
        valid = False
    if organic_df['retweet'].isnull().all():
        logger.warning("Organic data has no retweet information")
        valid = False
    
    return valid


def save_preprocessed_data(coordinated_df, organic_df, user_summary, output_dir='data/processed/'):
    """
    TODO: Save all preprocessed data for network analysis
    
    Save:
    - coordinated_df.parquet
    - organic_df.parquet
    - user_summary.parquet
    """
    pass


if __name__ == "__main__":
    # Load from data_loading.py output
    df = pd.read_parquet('data/processed/ira_merged_tweets.parquet')
    
    # TODO: Extract user features
    user_features = extract_user_features(df)
    
    # TODO: Identify core coordinated accounts
    core_ids, user_features = identify_core_coordinated_accounts(user_features)
    
    # TODO: Sample organic network for comparison
    organic_ids = sample_organic_network(df, core_ids, target_size=15000)
    
    # TODO: Prepare both networks
    coordinated_df, organic_df, user_summary = prepare_network_data(df, core_ids, organic_ids)
    
    # TODO: Validate
    is_valid = validate_network_samples(coordinated_df, organic_df)
    
    if is_valid:
        # TODO: Save for next step
        save_preprocessed_data(coordinated_df, organic_df, user_summary)
        print("✓ Preprocessing complete")
    else:
        print("✗ Validation failed")









