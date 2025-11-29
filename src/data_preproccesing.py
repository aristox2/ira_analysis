import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os 


#data_preproccesing.py#

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


# def identify_core_coordinated_accounts(user_features, retweet_threshold=10, frequency_threshold=5):
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
    


# def sample_organic_network(df, coordinated_user_ids, target_size=None):
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

    
    


def prepare_network_data(df, user_features, influence_threshold=None):
    """
    TODO:
    1. If influence_threshold is None, set it to median(user_features['avg_retweet_count']).
    2. Label users:
       - 'high_influence' if avg_retweet_count > threshold
       - 'low_influence' otherwise
    3. Merge labels back into df as 'network_type'.
    4. Build retweet or mention graph for each group.
    5. Compute eigenvector centrality per user.
    6. Merge centrality scores into user_features.
    7. Summarize correlations:
       - High-influence: centrality strongly predicts retweet count.
       - Low-influence: weak or no correlation.
    8. Return high_influence_df, low_influence_df, user_summary.
    """
    df = pd.DataFrame(df)
    user_features = pd.DataFrame(user_features)
    if df.empty or user_features.empty:
        logger.error("prepare_network_data received empty DataFrame(s)")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    if influence_threshold is None:
        influence_threshold = user_features['avg_retweet_count'].median()
    
    # Label users
    user_features['network_type'] = np.where(
        user_features['avg_retweet_count'] > influence_threshold,
        'high_influence',
        'low_influence'
    )
    
    # Merge labels back into df
    df = df.merge(user_features[['external_author_id', 'network_type']], on='external_author_id', how='left')
    
    # Split dataframes
    high_influence_df = df[df['network_type'] == 'high_influence']
    low_influence_df = df[df['network_type'] == 'low_influence']
    
    
    
    user_summary = user_features.copy()  # Placeholder for summary with centrality scores
    
    logger.info(f"Prepared network data: {len(high_influence_df)} high-influence tweets, {len(low_influence_df)} low-influence tweets")
    
    return high_influence_df, low_influence_df, user_summary

    



def validate_network_samples(high_df, low_df):
    """
    TODO:
    1. Check both DataFrames are non-empty.
    2. Compare sample sizes (high vs low influence).
    3. Compare follower distributions (mean, median).
    4. Check date ranges overlap.
    5. Verify retweet data exists in both sets.
    6. Compute correlation between centrality and avg_retweet_count:
       - High-influence: expect strong correlation.
       - Low-influence: expect weak/no correlation.
    7. Log summary stats for inspection.
    8. Return True if all checks pass, else False.
    """

    
    if high_df.empty or low_df.empty:
        logger.error("One of the network samples is empty")
        return False    

    # Sample size comparison
    size_diff = abs(len(high_df) - len(low_df))
    logger.info(f"High-influence sample size: {len(high_df)}, Low-influence sample size: {len(low_df)}, Difference: {size_diff}")
    # Sample size comparison
    size_diff = abs(len(high_df) - len(low_df))
    logger.info(f"High-influence sample size: {len(high_df)}, Low-influence sample size: {len(low_df)}, Difference: {size_diff}")

    # Comparring unique users
    high_users = high_df['external_author_id'].nunique()
    low_users = low_df['external_author_id'].nunique()
    logger.info(f"High-influence unique users: {high_users}, Low-influence unique users: {low_users}")
    
    # Follower distribution comparison
    high_followers = high_df['followers']
    low_followers = low_df['followers']
    logger.info(f"High-influence followers - Mean: {high_followers.mean()}, Median: {high_followers.median()}")
    logger.info(f"Low-influence followers - Mean: {low_followers.mean()}, Median: {low_followers.median()}")

    # Date range overlap
    high_dates = pd.to_datetime(high_df['publish_date'], errors='coerce')
    low_dates = pd.to_datetime(low_df['publish_date'], errors='coerce')
    overlap_start = max(high_dates.min(), low_dates.min())
    overlap_end = min(high_dates.max(), low_dates.max())
    if overlap_start >= overlap_end:
        logger.error("Date ranges do not overlap")
        return False
    logger.info(f"Date range overlap: {overlap_start} to {overlap_end}")

    # Retweet data check
    if high_df['retweet'].isnull().all() or low_df['retweet'].isnull().all():
        logger.error("Retweet data missing in one of the samples")
        return False

    # Placeholder for correlation check
    logger.info("Correlation between centrality and avg_retweet_count should be computed here")

    logger.info("Network samples validated successfully")
    return True






def save_preprocessed_data(high_influence_df, low_influence_df, user_summary, output_dir='data/processed/'):
    """
    TODO:
    1. Ensure DataFrames are non-empty.
    2. Ensure 'network_type' and centrality columns exist in user_summary.
    3. Save high_influence_df -> output_dir + 'high_influence_network.parquet'
    4. Save low_influence_df  -> output_dir + 'low_influence_network.parquet'
    5. Save user_summary      -> output_dir + 'user_summary_with_centrality.parquet'
    6. Log confirmation of saved files.
    """
    # df = pd.DataFrame(df)
    if high_influence_df.empty or low_influence_df.empty or user_summary.empty:
        logger.error("save_preprocessed_data received empty DataFrame(s)")
        return None
    
    #Save outputs
    os.makedirs(output_dir, exist_ok=True)
    high_path = os.path.join(output_dir, 'high_influence_network.parquet')
    low_path = os.path.join(output_dir, 'low_influence_network.parquet')
    summary_path = os.path.join(output_dir, 'user_summary_with_centrality.parquet')
    high_influence_df.to_parquet(high_path)
    low_influence_df.to_parquet(low_path)
    user_summary.to_parquet(summary_path)
    logger.info(f"Saved high-influence data to {high_path}")
    logger.info(f"Saved low-influence data to {low_path}")
    logger.info(f"Saved user summary data to {summary_path}")





if __name__ == "__main__":
    import pandas as pd
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load data
    df = pd.read_parquet("data/processed/ira_merged_tweets.parquet")

    # Extract user features
    user_features = extract_user_features(df)

    # Prepare high vs low influence networks
    high_influence_df, low_influence_df, user_summary = prepare_network_data(df, user_features)

    # Validate network splits
    is_valid = validate_network_samples(high_influence_df, low_influence_df)
    logger.info(f"Network validation: {'PASSED' if is_valid else 'FAILED'}")

    # Save preprocessed data
    save_preprocessed_data(high_influence_df, low_influence_df, user_summary)

