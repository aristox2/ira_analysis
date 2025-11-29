import pandas as pd
import os
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#data_loading.py#


# TODO: Load all 13 IRA CSV files and merge into one dataframe
# Expected output: merged_df with all tweets from coordinated IRA accounts

def load_ira_data(repo_path='data/raw/russian-troll-tweets'):
    """
    Load all IRA CSV files from the cloned repository
    """
    
    # Find all CSV files in the repo
    csv_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(csv_files)} CSV files in {repo_path!r}")
    if not csv_files:
        logger.error("No CSV files found. Check repo_path and that files exist.")
        return pd.DataFrame()  # return empty DF instead of letting concat fail
    
    # Load each CSV and append to list
    dfs = []
    for csv_file in sorted(csv_files):
        try:
            # be a bit more robust with encoding/low_memory
            df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
            
            if df is None or df.shape[0] == 0:
                logger.warning(f"File {csv_file} loaded but is empty, skipping.")
                continue
            dfs.append(df)
            logger.info(f"Loaded {csv_file}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")    
    if not dfs:
        logger.error("No CSV files were successfully read. Exiting load step.")
        return pd.DataFrame()
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total merged rows: {len(merged_df):,}")
    
    return merged_df


def clean_ira_data(df):
    df = pd.DataFrame(df)
    if df.empty:
        logger.error("clean_ira_data received empty DataFrame")
        return df
    
    # Convert date columns to datetime
    if 'publish_date' in df.columns:
        df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
    
    # TODO: Convert numeric columns from object to proper types
    numeric_cols = ['external_author_id', 'tweet_id', 'retweet', 'followers', 'following', 'alt_external_id']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove duplicates by tweet_id
    initial_rows = len(df)
    if 'tweet_id' in df.columns:
        df = df.drop_duplicates(subset=['tweet_id'], keep='first')
        logger.info(f"Removed {initial_rows - len(df)} duplicate tweets")
    
    # Drop rows with missing critical data
    drop_cols = [c for c in ['external_author_id', 'tweet_id'] if c in df.columns]
    if drop_cols:
        df = df.dropna(subset=drop_cols)
    
    return df


def save_processed_data(df, output_path='data/processed/ira_merged_tweets.parquet'):
    """
    TODO: Save cleaned data to efficient format for next steps
    """
    df = pd.DataFrame(df)  # ensure it's a DataFrame
    if df.empty:
        logger.error("No data to save; DataFrame is empty.")
        return None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    logger.info(f"Saved processed data to {output_path}")
    return output_path


if __name__ == "__main__":
    # Load raw data
    df = load_ira_data()
    
    if df.empty:
        logger.error("No data loaded. Exiting.")
        sys.exit(1)
    
    # Clean data
    df = clean_ira_data(df)
    # Save for next pipeline step
    save_processed_data(df)    
    # Print summary stats (guarded)
    print(f"\n=== DATA SUMMARY ===")
    print(f"Total tweets: {len(df):,}")
    if 'external_author_id' in df.columns:
        print(f"Unique users: {df['external_author_id'].nunique():,}")
    if 'publish_date' in df.columns:
        print(f"Date range: {df['publish_date'].min()} to {df['publish_date'].max()}")
    if 'followers' in df.columns:
        print(f"Avg followers per user: {df['followers'].mean():.0f}")
    if 'retweet' in df.columns:
        print(f"Avg retweets per tweet: {df['retweet'].mean():.2f}")