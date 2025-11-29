import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def load_analysis_data():
    """
    TODO: Load user_summary with all centrality scores
    
    Return: DataFrame with columns including:
    - external_author_id
    - avg_followers
    - avg_retweet_count (this is your influence metric)
    - network_type
    - eigenvector_centrality_high_influence
    - eigenvector_centrality_low_influence
    - degree_centrality_high_influence
    - degree_centrality_low_influence
    """
    pass
    df = pd.read_parquet('data/processed/user_summary_with_centrality.parquet')

    # Log what you loaded
    logger.info(f"Loaded {len(df)} users")
    logger.info(f"High-influence users: {(df['network_type'] == 'high_influence').sum()}")
    logger.info(f"Low-influence users: {(df['network_type'] == 'low_influence').sum()}")

    # Validate expected columns
    required_columns = ['external_author_id', 'avg_followers', 'avg_retweet_count', 'network_type',
                        'eigenvector_centrality_high_influence', 'eigenvector_centrality_low_influence',
                        'degree_centrality_high_influence', 'degree_centrality_low_influence']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df





def test_hypothesis_single_network(df, network_type):
    """
    TODO: Test if eigenvector centrality adds predictive value beyond followers
    
    Steps:
    1. Filter to users in this network_type (high_influence or low_influence)
    2. Get relevant columns:
       - X_baseline = avg_followers only
       - X_full = avg_followers + eigenvector_centrality
       - y = avg_retweet_count (influence metric)
    3. Fit Model 1 (baseline): y ~ avg_followers
       - Calculate R² for baseline
    4. Fit Model 2 (full): y ~ avg_followers + eigenvector_centrality
       - Calculate R² for full model
    5. Calculate ΔR² = R²_full - R²_baseline
    6. Run F-test to check if improvement is significant
       - F = ((RSS_baseline - RSS_full) / 1) / (RSS_full / (n - 3))
       - p-value from F-distribution
    7. Log all results
    
    Return: dict with {
        'network_type': network_type,
        'n_users': int,
        'r2_baseline': float,
        'r2_full': float,
        'delta_r2': float,
        'f_statistic': float,
        'p_value': float,
        'coefficients_baseline': dict,
        'coefficients_full': dict
    }
    """
    pass

    df = df[df['network_type'] == network_type].copy()
    x_baseline = df[['avg_followers']].values
    x_full = df[['avg_followers', f'eigenvector_centrality_{network_type}']].values
    y = df['avg_retweet_count'].values
    n = len(df)
    #Fit Model 1 (baseline)
    model_baseline = LinearRegression().fit(x_baseline, y) 
    r2_baseline = model_baseline.score(x_baseline, y) # how much influence can be predicted by follower count alone

    #Fit Model 2 (full)
    model_full = LinearRegression().fit(x_full, y)
    r2_full = model_full.score(x_full, y) #Test if adding network position improves prediction
    
    #Calculate ΔR² = R²_full - R²_baseline
    delta_r2 = r2_full - r2_baseline #Change in R squared t
    
    
    #Establish residual sum of squares for both models
    rss_baseline = np.sum((y - model_baseline.predict(x_baseline))**2)
    rss_full = np.sum((y - model_full.predict(x_full))**2) #Measure the improvement from adding eigenvector centrality
    #Run F-test to check if improvement is significant
    f_statistic = ((rss_baseline - rss_full) / 1) / (rss_full / (n - 3))
    p_value = 1 - stats.f.cdf(f_statistic, 1, n - 3)
    logger.info(f"Network Type: {network_type}")
    logger.info(f"Number of Users: {n}")

    logger.info(f"R² Baseline Model: {r2_baseline:.4f}")
    logger.info(f"R² Full Model: {r2_full:.4f}")
    logger.info(f"ΔR²: {delta_r2:.4f}")
    logger.info(f"F-statistic: {f_statistic:.4f}")
    logger.info(f"p-value: {p_value:.4f}")
    
    # Extract coefficients for reporting
    coeffs_baseline = {'intercept': float(model_baseline.intercept_), 'avg_followers': float(model_baseline.coef_[0])}
    coeffs_full = {'intercept': float(model_full.intercept_), 'avg_followers': float(model_full.coef_[0])}
    # if second coefficient (eigenvector) exists, include it
    if model_full.coef_.shape[0] > 1:
        coeffs_full[f'eigenvector_{network_type}'] = float(model_full.coef_[1])

    # Defensive: if n is too small, avoid invalid F calculation
    if n <= 3:
        f_statistic = None
        p_value = None

    return {
        'network_type': network_type,
        'n_users': int(n),
        'r2_baseline': float(r2_baseline),
        'r2_full': float(r2_full),
        'delta_r2': float(delta_r2),
        'f_statistic': float(f_statistic) if f_statistic is not None else None,
        'p_value': float(p_value) if p_value is not None else None,
        'coefficients_baseline': coeffs_baseline,
        'coefficients_full': coeffs_full
    }
    

    


def compare_networks(results_high, results_low):
    """
    TODO: Compare results between high and low influence networks
    
    Steps:
    1. Calculate relative improvement: (ΔR²_high - ΔR²_low) / ΔR²_low
    2. Log comparison summary
    3. Determine if hypothesis is supported:
       - Is ΔR²_high significantly larger than ΔR²_low?
       - Are both p-values < 0.05?
    
    Return: dict with comparison metrics
    """
    pass

    delta_r2_high = results_high['delta_r2']
    delta_r2_low = results_low['delta_r2']
    relative_improvement = (delta_r2_high - delta_r2_low) / delta_r2_low if delta_r2_low != 0 else np.inf
    logger.info(f"Relative Improvement in ΔR² (High vs Low): {relative_improvement:.4f}")

    hypothesis_supported = (results_high['p_value'] < 0.05) and (results_low['p_value'] < 0.05) and (delta_r2_high > delta_r2_low)
    if hypothesis_supported:
        logger.info("Hypothesis Supported: Eigenvector centrality adds more predictive value in high-influence network.")
    else:
        logger.info("Hypothesis Not Supported.")
    return {
        'relative_improvement': relative_improvement,
        'hypothesis_supported': hypothesis_supported
    }


def plot_baseline_vs_full_comparison(results_high, results_low, output_dir='results/figures/'):
    """
    TODO: Create bar chart comparing R² improvement
    
    Create grouped bar chart:
    - X-axis: Model type (Baseline, Full)
    - Y-axis: R² value
    - Two groups: High-influence, Low-influence
    - Show ΔR² as annotation
    
    Save to output_dir
    """
    pass


    x= ['Baseline', 'Full']
    r2_high = [results_high['r2_baseline'], results_high['r2_full']]
    r2_low = [results_low['r2_baseline'], results_low['r2_full']]
    
    x_indexes = np.arange(len(x))
    width = 0.35
    
    plt.bar(x_indexes - width/2, r2_high, width=width, label='High-Influence Network')
    plt.bar(x_indexes + width/2, r2_low, width=width, label='Low-Influence Network')
    plt.xlabel('Model Type')
    plt.ylabel('R² Value')
    plt.title('Comparison of R² Values: Baseline vs Full Model')
    plt.xticks(ticks=x_indexes, labels=x)
    
    
    plt.ylim(0, 1)
    plt.legend()
    delta_r2_high = results_high['delta_r2']
    delta_r2_low = results_low['delta_r2']
    plt.text(0.5 - width/2, max(r2_high)+0.02, f'ΔR²: {delta_r2_high:.4f}', ha='center', color='black')
    plt.text(0.5 + width/2, max(r2_low)+0.02, f'ΔR²: {delta_r2_low:.4f}', ha='center', color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_comparison_baseline_vs_full.png'))
    


def plot_scatter_eigenvector_vs_influence(df, network_type, output_dir='results/figures/'):
    """
    TODO: Create scatter plot showing eigenvector centrality vs influence
    
    For specified network_type:
    1. Filter to users in that network
    2. X-axis: eigenvector_centrality
    3. Y-axis: avg_retweet_count
    4. Color points by avg_followers (gradient)
    5. Add regression line
    6. Show R² and equation on plot
    7. Save to output_dir
    
    This visualizes if eigenvector centrality correlates with influence
    """
    pass

    df = df[df['network_type'] == network_type].copy()
    x = df[f'eigenvector_centrality_{network_type}']
    y = df['avg_retweet_count']
    followers = df['avg_followers']
    plt.figure()

    scatter = plt.scatter(x, y, c=followers, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Average Followers')
    plt.xlabel('Eigenvector Centrality')
    plt.ylabel('Average Retweet Count (Influence)')
    plt.title(f'Eigenvector Centrality vs Influence ({network_type.replace("_", " ").title()})')

    # Fit regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, intercept + slope * x, 'r', label=f'Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_value**2:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'eigenvector_vs_influence_{network_type}.png'))



def plot_followers_vs_influence(df, network_type, output_dir='results/figures/'):
    """
    TODO: Create scatter plot showing followers vs influence (baseline)
    
    For comparison with eigenvector centrality plot
    Same style as above but X-axis = avg_followers
    """
    pass

    df = df[df['network_type'] == network_type].copy()
    x = df['avg_followers']
    y = df['avg_retweet_count']
    plt.figure()
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel('Average Followers')
    plt.ylabel('Average Retweet Count (Influence)')
    plt.title(f'Average Followers vs Influence ({network_type.replace("_", " ").title()})')
    # Fit regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, intercept + slope * x, 'r', label=f'Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_value**2:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'followers_vs_influence_{network_type}.png'))



def plot_network_structure_comparison(stats_high, stats_low, output_dir='results/figures/'):
    """
    TODO: Create comparison visualization of network structures
    
    Multi-panel figure showing:
    1. Density comparison (bar chart)
    2. Diameter comparison (bar chart)
    3. Average path length (bar chart)
    4. Clustering coefficient (bar chart)
    
    This explains WHY eigenvector centrality matters more in high-influence
    """
    pass

    # Helper to try multiple possible keys and return a numeric value or NaN
    def get_metric(d, keys):
        for k in keys:
            if k in d:
                try:
                    return float(d[k])
                except Exception:
                    return np.nan
        return np.nan

    df = pd.DataFrame({
        'Metric': ['Density', 'Diameter', 'Average Path Length', 'Clustering Coefficient'],
        'High-Influence': [
            get_metric(stats_high, ['density', 'Density']),
            get_metric(stats_high, ['diameter', 'Diameter']),
            get_metric(stats_high, ['average_path_length', 'avg_path_length', 'avg_path', 'average_path_length']),
            get_metric(stats_high, ['clustering_coefficient', 'avg_clustering', 'clustering'])
        ],
        'Low-Influence': [
            get_metric(stats_low, ['density', 'Density']),
            get_metric(stats_low, ['diameter', 'Diameter']),
            get_metric(stats_low, ['average_path_length', 'avg_path_length', 'avg_path', 'average_path_length']),
            get_metric(stats_low, ['clustering_coefficient', 'avg_clustering', 'clustering'])
        ]
    })

    df_melted = df.melt(id_vars='Metric', var_name='Network Type', value_name='Value')
    plt.figure()
    # seaborn will handle NaN by skipping those bars; ensure numeric dtype
    df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')
    sns.barplot(data=df_melted, x='Metric', y='Value', hue='Network Type')
    plt.title('Network Structure Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'network_structure_comparison.png'))



def create_correlation_heatmap(df, network_type, output_dir='results/figures/'):
    """
    TODO: Create correlation heatmap
    
    Show correlations between:
    - avg_followers
    - eigenvector_centrality
    - degree_centrality
    - pagerank
    - avg_retweet_count (influence)
    
    Separate heatmaps for high vs low influence
    Shows if metrics are redundant or complementary
    """
    pass

    df = df[df['network_type'] == network_type].copy()
    corr_matrix = df[['avg_followers',
                        f'eigenvector_centrality_{network_type}',
                        f'degree_centrality_{network_type}',
                        f'pagerank_{network_type}',
                        'avg_retweet_count']].corr()
    plt.figure()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Heatmap ({network_type.replace("_", " ").title()})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'correlation_heatmap_{network_type}.png'))



def generate_summary_table(results_high, results_low, output_dir='results/tables/'):
    """
    TODO: Create publication-ready summary table
    
    Table with columns:
    - Network Type
    - N Users
    - R² Baseline
    - R² Full
    - ΔR²
    - F-statistic
    - p-value
    - Conclusion
    
    Save as CSV and formatted text
    """
    pass

    results = [results_high, results_low]
    summary_data = []
    for res in results:
        conclusion = "Significant" if res['p_value'] < 0.05 else "Not Significant"
        summary_data.append({
            'Network Type': res['network_type'].replace('_', ' ').title(),
            'N Users': res['n_users'],
            'R² Baseline': f"{res['r2_baseline']:.4f}",
            'R² Full': f"{res['r2_full']:.4f}",
            'ΔR²': f"{res['delta_r2']:.4f}",
            'F-statistic': f"{res['f_statistic']:.4f}",
            'p-value': f"{res['p_value']:.4f}",
            'Conclusion': conclusion
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'summary_table.csv'), index=False)
    with open(os.path.join(output_dir, 'summary_table.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_df.to_string(index=False))


def generate_regression_table(results_high, results_low, output_dir='results/tables/'):
    """
    TODO: Create detailed regression coefficients table
    
    Show coefficients, standard errors, t-statistics, p-values
    for both baseline and full models in both networks
    
    Publication-ready format
    """
    pass

    results = [results_high, results_low]
    regression_data = []
    for res in results:
        for model_type in ['baseline', 'full']:
            coeffs = res[f'coefficients_{model_type}']
            for var, coef in coeffs.items():
                regression_data.append({
                    'Network Type': res['network_type'].replace('_', ' ').title(),
                    'Model Type': model_type.title(),
                    'Variable': var,
                    'Coefficient': f"{coef:.4f}",
                    # Placeholder values for standard error, t-statistic, p-value
                    'Std. Error': 'N/A',
                    't-Statistic': 'N/A',
                    'p-Value': 'N/A'
                })
    regression_df = pd.DataFrame(regression_data)
    regression_df.to_csv(os.path.join(output_dir, 'regression_table.csv'), index=False)
    with open(os.path.join(output_dir, 'regression_table.txt'), 'w', encoding='utf-8') as f:
        f.write(regression_df.to_string(index=False))



def save_all_results(results_high, results_low, comparison, output_dir='results/'):
    """
    TODO: Save all analysis results
    
    Save as JSON for easy loading later:
    - results_high
    - results_low  
    - comparison metrics
    - Summary statistics
    """
    pass

    all_results = {
        'results_high': results_high,
        'results_low': results_low,
        'comparison': comparison
    }
    pd.Series(all_results).to_json(os.path.join(output_dir, 'all_results.json'))
    


if __name__ == "__main__":
    # Create output directories
    os.makedirs('results/figures/', exist_ok=True)
    os.makedirs('results/tables/', exist_ok=True)
    
    # Load data with centrality scores
    logger.info("Loading analysis data...")
    df = load_analysis_data()
    
    # Load network stats for structure comparison
    stats_high = pd.read_json('data/processed/high_influence_network_stats.json', typ='series').to_dict()
    stats_low = pd.read_json('data/processed/low_influence_network_stats.json', typ='series').to_dict()
    
    # Test hypothesis for high-influence network
    logger.info("\n=== TESTING HIGH-INFLUENCE NETWORK ===")
    results_high = test_hypothesis_single_network(df, 'high_influence')
    
    # Test hypothesis for low-influence network
    logger.info("\n=== TESTING LOW-INFLUENCE NETWORK ===")
    results_low = test_hypothesis_single_network(df, 'low_influence')
    
    # Compare networks
    logger.info("\n=== COMPARING NETWORKS ===")
    comparison = compare_networks(results_high, results_low)
    
    # Generate visualizations
    logger.info("\n=== GENERATING VISUALIZATIONS ===")
    plot_baseline_vs_full_comparison(results_high, results_low)
    plot_scatter_eigenvector_vs_influence(df, 'high_influence')
    plot_scatter_eigenvector_vs_influence(df, 'low_influence')
    plot_followers_vs_influence(df, 'high_influence')
    plot_followers_vs_influence(df, 'low_influence')
    plot_network_structure_comparison(stats_high, stats_low)
    create_correlation_heatmap(df, 'high_influence')
    create_correlation_heatmap(df, 'low_influence')
    
    # Generate tables
    logger.info("\n=== GENERATING TABLES ===")
    generate_summary_table(results_high, results_low)
    generate_regression_table(results_high, results_low)
    
    # Save everything
    save_all_results(results_high, results_low, comparison)
    
    logger.info("\n✓ Results generation complete!")
    logger.info(f"Figures saved to: results/figures/")
    logger.info(f"Tables saved to: results/tables/")