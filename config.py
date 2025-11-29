"""
Configuration file for IRA Tweets Network Analysis

This file contains all the parameters and settings for the analysis pipeline.
Modify these values to adjust the analysis according to your needs.

Author: [Your Name]
Date: [Current Date]
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data paths
DATA_PATHS = {
    'raw': DATA_DIR / "raw",
    'processed': DATA_DIR / "processed", 
    'sampled': DATA_DIR / "sampled",
    'results': RESULTS_DIR
}

# Sampling parameters
SAMPLING = {
    'sample_size': 10000,  # Number of tweets to sample
    'method': 'random',    # Sampling method: 'random', 'stratified', 'temporal'
    'seed': 42,           # Random seed for reproducibility
    'min_activity': 5,     # Minimum number of tweets per user
    'max_users': 1000,    # Maximum number of users to include
}

# Network construction parameters
NETWORK_PARAMS = {
    'min_retweets': 2,        # Minimum retweets for edge inclusion
    'min_mentions': 2,        # Minimum mentions for edge inclusion
    'min_hashtag_cooccur': 3, # Minimum co-occurrence for hashtag edges
    'time_window_days': 30,   # Time window for temporal networks
    'weight_threshold': 0.1,  # Minimum edge weight threshold
}

# Analysis parameters
ANALYSIS_PARAMS = {
    'centrality_measures': [
        'degree_centrality',
        'betweenness_centrality', 
        'closeness_centrality',
        'eigenvector_centrality',
        'pagerank'
    ],
    'community_detection': {
        'algorithm': 'louvain',  # 'louvain', 'label_propagation', 'girvan_newman'
        'resolution': 1.0,      # Resolution parameter for modularity
        'random_state': 42
    },
    'temporal_analysis': {
        'time_bins': 10,         # Number of time bins for temporal analysis
        'min_nodes_per_bin': 10, # Minimum nodes per time bin
    }
}

# Visualization parameters
VISUALIZATION_PARAMS = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'husl',
    'node_size': 50,
    'edge_width': 0.5,
    'alpha': 0.7,
    'layout': 'spring',  # 'spring', 'circular', 'random', 'hierarchical'
}

# File formats and naming
FILE_FORMATS = {
    'data_format': 'csv',           # Format for data files
    'graph_format': 'gexf',         # Format for network files
    'figure_format': 'png',         # Format for figures
    'table_format': 'csv',          # Format for tables
    'model_format': 'pkl',          # Format for saved models
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',                # Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'analysis.log',         # Log file name
    'max_size': 10 * 1024 * 1024,  # Max log file size (10MB)
    'backup_count': 5,              # Number of backup log files
}

# Twitter API parameters (if using live data)
TWITTER_API = {
    'bearer_token': os.getenv('TWITTER_BEARER_TOKEN', ''),
    'api_key': os.getenv('TWITTER_API_KEY', ''),
    'api_secret': os.getenv('TWITTER_API_SECRET', ''),
    'access_token': os.getenv('TWITTER_ACCESS_TOKEN', ''),
    'access_secret': os.getenv('TWITTER_ACCESS_SECRET', ''),
    'rate_limit': 300,              # Requests per 15 minutes
    'retry_attempts': 3,            # Number of retry attempts
}

# Data quality thresholds
DATA_QUALITY = {
    'min_tweet_length': 10,         # Minimum tweet character length
    'max_tweet_length': 280,        # Maximum tweet character length
    'min_hashtags': 1,              # Minimum number of hashtags
    'max_hashtags': 10,             # Maximum number of hashtags
    'language_filter': ['en'],      # Languages to include
    'exclude_bots': True,           # Whether to exclude bot accounts
    'bot_keywords': ['bot', 'automated', 'spam'],  # Keywords to identify bots
}

# Statistical analysis parameters
STATISTICAL_PARAMS = {
    'significance_level': 0.05,     # Alpha level for statistical tests
    'confidence_interval': 0.95,    # Confidence interval level
    'bootstrap_samples': 1000,      # Number of bootstrap samples
    'correlation_threshold': 0.3,   # Minimum correlation for reporting
    'effect_size_threshold': 0.1,   # Minimum effect size for reporting
}

# Performance and memory settings
PERFORMANCE = {
    'max_memory_gb': 8,             # Maximum memory usage in GB
    'chunk_size': 10000,            # Chunk size for processing large datasets
    'parallel_workers': 4,         # Number of parallel workers
    'cache_results': True,          # Whether to cache intermediate results
    'cache_dir': PROJECT_ROOT / "cache",
}

# Output settings
OUTPUT = {
    'save_intermediate': True,      # Save intermediate results
    'compress_outputs': True,       # Compress output files
    'include_metadata': True,       # Include metadata in outputs
    'timestamp_files': True,        # Add timestamps to output files
}

# Validation settings
VALIDATION = {
    'validate_inputs': True,        # Validate input data
    'check_data_types': True,       # Check data types
    'verify_network_properties': True,  # Verify network properties
    'run_unit_tests': True,        # Run unit tests
}

# Create directories if they don't exist
def create_directories():
    """Create necessary directories for the project."""
    for path in DATA_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "models").mkdir(parents=True, exist_ok=True)
    
    # Create cache directory
    if PERFORMANCE['cache_dir']:
        PERFORMANCE['cache_dir'].mkdir(parents=True, exist_ok=True)

# Initialize directories
create_directories()

# Export configuration for easy access
CONFIG = {
    'data_paths': DATA_PATHS,
    'sampling': SAMPLING,
    'network_params': NETWORK_PARAMS,
    'analysis_params': ANALYSIS_PARAMS,
    'visualization_params': VISUALIZATION_PARAMS,
    'file_formats': FILE_FORMATS,
    'logging_config': LOGGING_CONFIG,
    'twitter_api': TWITTER_API,
    'data_quality': DATA_QUALITY,
    'statistical_params': STATISTICAL_PARAMS,
    'performance': PERFORMANCE,
    'output': OUTPUT,
    'validation': VALIDATION,
}

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")

