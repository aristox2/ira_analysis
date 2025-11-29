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
    Load user_summary with all centrality scores
    """
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
    Test if eigenvector centrality adds predictive value beyond followers
    """
    df = df[df[f'eigenvector_centrality_{network_type}'] > 0].copy()  # Only users actually in network    
    x_baseline = df[['avg_followers']].values
    x_full = df[['avg_followers', f'eigenvector_centrality_{network_type}']].values
    y = df['avg_retweet_count'].values
    n = len(df)
    
    # Fit Model 1 (baseline)
    model_baseline = LinearRegression().fit(x_baseline, y) 
    r2_baseline = model_baseline.score(x_baseline, y)

    # Fit Model 2 (full)
    model_full = LinearRegression().fit(x_full, y)
    r2_full = model_full.score(x_full, y)
    
    # Calculate ΔR²
    delta_r2 = r2_full - r2_baseline
    
    # Establish residual sum of squares for both models
    rss_baseline = np.sum((y - model_baseline.predict(x_baseline))**2)
    rss_full = np.sum((y - model_full.predict(x_full))**2)
    
    # Run F-test to check if improvement is significant
    if n > 3:
        f_statistic = ((rss_baseline - rss_full) / 1) / (rss_full / (n - 3))
        p_value = 1 - stats.f.cdf(f_statistic, 1, n - 3)
    else:
        f_statistic = None
        p_value = None

    logger.info(f"Network Type: {network_type}")
    logger.info(f"Number of Users: {n}")
    logger.info(f"R² Baseline Model: {r2_baseline:.4f}")
    logger.info(f"R² Full Model: {r2_full:.4f}")
    logger.info(f"ΔR²: {delta_r2:.4f}")
    logger.info(f"F-statistic: {f_statistic:.4f}" if f_statistic else "F-statistic: N/A")
    logger.info(f"p-value: {p_value:.4f}" if p_value else "p-value: N/A")
    
    # Extract coefficients for reporting
    coeffs_baseline = {
        'intercept': float(model_baseline.intercept_), 
        'avg_followers': float(model_baseline.coef_[0])
    }
    coeffs_full = {
        'intercept': float(model_full.intercept_), 
        'avg_followers': float(model_full.coef_[0]),
        f'eigenvector_{network_type}': float(model_full.coef_[1])
    }

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
    Compare results between high and low influence networks
    """
    delta_r2_high = results_high['delta_r2']
    delta_r2_low = results_low['delta_r2']
    relative_improvement = (delta_r2_high - delta_r2_low) / delta_r2_low if delta_r2_low != 0 else np.inf
    
    logger.info(f"Relative Improvement in ΔR² (High vs Low): {relative_improvement:.4f}")

    # Hypothesis supported if high-influence improvement is significant and substantially larger
    hypothesis_supported = (
        results_high['p_value'] < 0.05 and  # High network improvement is significant
        delta_r2_high > delta_r2_low and     # High network has larger improvement
        relative_improvement > 0.2           # At least 20% better
    )
    
    if hypothesis_supported:
        logger.info("✓ Hypothesis Supported: Eigenvector centrality adds more predictive value in high-influence network.")
    else:
        logger.info("✗ Hypothesis Not Supported.")
    
    return {
        'relative_improvement': relative_improvement,
        'hypothesis_supported': hypothesis_supported
    }


def plot_baseline_vs_full_comparison(results_high, results_low, output_dir='results/figures/'):
    """
    Create bar chart comparing R² improvement
    """
    x = ['Baseline', 'Full']
    r2_high = [results_high['r2_baseline'], results_high['r2_full']]
    r2_low = [results_low['r2_baseline'], results_low['r2_full']]
    
    x_indexes = np.arange(len(x))
    width = 0.35
    
    plt.figure()
    plt.bar(x_indexes - width/2, r2_high, width=width, label='High-Influence Network')
    plt.bar(x_indexes + width/2, r2_low, width=width, label='Low-Influence Network')
    plt.xlabel('Model Type')
    plt.ylabel('R² Value')
    plt.title('Comparison of R² Values: Baseline vs Full Model')
    plt.xticks(ticks=x_indexes, labels=x)
    plt.ylim(0, max(max(r2_high), max(r2_low)) + 0.15)
    plt.legend()
    
    delta_r2_high = results_high['delta_r2']
    delta_r2_low = results_low['delta_r2']
    plt.text(1 - width/2, max(r2_high)+0.02, f'ΔR²: {delta_r2_high:.4f}', ha='center', color='black')
    plt.text(1 + width/2, max(r2_low)+0.02, f'ΔR²: {delta_r2_low:.4f}', ha='center', color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_comparison_baseline_vs_full.png'))
    plt.close()


def plot_scatter_eigenvector_vs_influence(df, network_type, output_dir='results/figures/'):
    """
    Create scatter plot showing eigenvector centrality vs influence
    """
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
    plt.close()


def plot_followers_vs_influence(df, network_type, output_dir='results/figures/'):
    """
    Create scatter plot showing followers vs influence (baseline)
    """
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
    plt.close()


def plot_network_structure_comparison(stats_high, stats_low, output_dir='results/figures/'):
    """
    Create comparison visualization of network structures
    """
    def get_metric(d, keys):
        for k in keys:
            if k in d:
                try:
                    return float(d[k])
                except Exception:
                    return np.nan
        return np.nan

    df = pd.DataFrame({
        'Metric': ['Density', 'Diameter', 'Avg Path Length', 'Clustering Coeff'],
        'High-Influence': [
            get_metric(stats_high, ['density']),
            get_metric(stats_high, ['diameter']),
            get_metric(stats_high, ['avg_path_length', 'average_path_length']),
            get_metric(stats_high, ['avg_clustering', 'clustering_coefficient'])
        ],
        'Low-Influence': [
            get_metric(stats_low, ['density']),
            get_metric(stats_low, ['diameter']),
            get_metric(stats_low, ['avg_path_length', 'average_path_length']),
            get_metric(stats_low, ['avg_clustering', 'clustering_coefficient'])
        ]
    })

    df_melted = df.melt(id_vars='Metric', var_name='Network Type', value_name='Value')
    df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')
    
    plt.figure()
    sns.barplot(data=df_melted, x='Metric', y='Value', hue='Network Type')
    plt.title('Network Structure Comparison')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'network_structure_comparison.png'))
    plt.close()


def create_correlation_heatmap(df, network_type, output_dir='results/figures/'):
    """
    Create correlation heatmap
    """
    df_high = df[df['network_type'] == 'high_influence']
    print(f"Users with centrality > 0: {(df_high['eigenvector_centrality_high_influence'] > 0).sum()}")
    print(f"Total high-influence users: {len(df_high)}")



    corr_matrix = df[[
        'avg_followers',
        f'eigenvector_centrality_{network_type}',
        f'degree_centrality_{network_type}',
        f'pagerank_{network_type}',
        'avg_retweet_count'
    ]].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(f'Correlation Heatmap ({network_type.replace("_", " ").title()})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'correlation_heatmap_{network_type}.png'))
    plt.close()


def generate_summary_table(results_high, results_low, output_dir='results/tables/'):
    """
    Create publication-ready summary table
    """
    results = [results_high, results_low]
    summary_data = []
    for res in results:
        conclusion = "Significant" if res['p_value'] and res['p_value'] < 0.05 else "Not Significant"
        summary_data.append({
            'Network Type': res['network_type'].replace('_', ' ').title(),
            'N Users': res['n_users'],
            'R² Baseline': f"{res['r2_baseline']:.4f}",
            'R² Full': f"{res['r2_full']:.4f}",
            'ΔR²': f"{res['delta_r2']:.4f}",
            'F-statistic': f"{res['f_statistic']:.4f}" if res['f_statistic'] else "N/A",
            'p-value': f"{res['p_value']:.4f}" if res['p_value'] else "N/A",
            'Conclusion': conclusion
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'summary_table.csv'), index=False)
    with open(os.path.join(output_dir, 'summary_table.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_df.to_string(index=False))
    logger.info(f"Summary table saved to {output_dir}")


def generate_regression_table(results_high, results_low, output_dir='results/tables/'):
    """
    Create detailed regression coefficients table
    """
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
                    'Coefficient': f"{coef:.4f}"
                })
    regression_df = pd.DataFrame(regression_data)
    regression_df.to_csv(os.path.join(output_dir, 'regression_table.csv'), index=False)
    with open(os.path.join(output_dir, 'regression_table.txt'), 'w', encoding='utf-8') as f:
        f.write(regression_df.to_string(index=False))
    logger.info(f"Regression table saved to {output_dir}")


def save_all_results(results_high, results_low, comparison, output_dir='results/'):
    """
    Save all analysis results
    """
    all_results = {
        'results_high': results_high,
        'results_low': results_low,
        'comparison': comparison
    }
    import json
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"All results saved to {output_dir}")


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