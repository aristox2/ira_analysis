# Eigenvector Centrality in Coordinated Disinformation Networks

## Project Overview
Analysis of how eigenvector centrality predicts influence differently across high-influence vs low-influence tiers within the Internet Research Agency's coordinated disinformation campaign.

## Hypothesis
In low-influence nodes, eigenvector centrality adds little predictive value beyond follower count. In high-influence coordination hubs, eigenvector centrality becomes a stronger predictor, reflecting concentrated network structures where central positioning drives disproportionate amplification.

## Dataset
- **Source**: Internet Research Agency Twitter dataset (FiveThirtyEight)
- **Period**: 2012-2018
- **Tweets**: 2,944,811
- **Unique Users**: 2,483 IRA accounts

## Methodology
1. User feature extraction and influence classification
2. Retweet network construction via TF-IDF text matching
3. Network graph building (high-influence vs low-influence)
4. Eigenvector centrality calculation
5. Regression analysis comparing predictive power

## Current Status
- [x] Data loading and cleaning
- [x] User feature extraction
- [x] Network splitting (high vs low influence)
- [ ] Retweet edge extraction (in progress)
- [ ] Network graph construction
- [ ] Centrality calculation
- [ ] Statistical analysis