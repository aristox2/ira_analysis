# Eigenvector Centrality in Coordinated Disinformation Networks  
### Using the 3M IRA Tweet Dataset Published by FiveThirtyEight  
Dataset source: **“3 Million Russian Troll Tweets” (FiveThirtyEight, 2018)**  
https://fivethirtyeight.com/features/why-were-sharing-3-million-russian-troll-tweets

---

## Purpose  
This project investigates whether **eigenvector centrality** is an effective predictor of account influence **within a coordinated disinformation network**, using the publicly released IRA dataset.  
Because the dataset contains *full tweet text*, it avoids the Tweet-ID hydration barriers that prevent analysis in modern Twitter/X datasets.

The study reframes the research question to compare **high-influence vs. low-influence IRA accounts**, based on median retweet performance.

---

## Repository Overview  
### 1. **Data Loading**
- Loads all 13 raw IRA CSV files.  
- Fixes inconsistent ID types (e.g., `external_author_id`).  
- Produces a unified Parquet dataset of ~2.9M tweets and 2,483 unique accounts.

### 2. **User-Level Feature Extraction**
- Computes activity statistics (tweet count, avg retweets).  
- Derives influence labels via **median split**.  
- Generates `user_summary.parquet`.

### 3. **Network Construction**
Two network types are generated:

#### **a. Retweet Reconstruction (directed)**
- Detects retweet patterns (`RT @username:`).  
- Uses TF-IDF similarity search to find the most likely original tweet.  
- Resulting network too sparse (LCC ≈ 47 nodes) for robust analysis.

#### **b. Hashtag Co-occurrence Network (undirected)**
- Users connected if they share hashtags within **6-hour windows**.  
- Edge created only if users co-occur **≥ 3 times**.  
- Produces a dense and analyzable coordination graph.

#### **Hybrid Network**
- Retweet edges + hashtag co-occurrence edges.  
- Used as the primary structure for centrality-based analysis.

### 4. **Network Analysis**
- Eigenvector centrality  
- Degree centrality  
- PageRank  
- Largest Connected Component (LCC) extraction  
- Path length, clustering, density  
- Saved as Parquet + JSON for reproducibility

### 5. **Results Generation**
- Produces analysis-ready datasets for:
  - Influence vs. centrality regression  
  - High- vs. low-influence comparisons  
  - Structural characterization of the network  

---

## Research Significance  
The IRA dataset represents a coordinated state-sponsored operation with minimal organic user activity.  
This provides a controlled environment to examine how influence operates **within a coordinated network**, isolating structural mechanisms from typical social media noise.

**Why eigenvector centrality?**
- Coordinated operations rely on mutual amplification.  
- Eigenvector centrality captures *embeddedness in reinforcement structures*, not raw popularity.  
- If high-influence IRA accounts occupy central structural positions, this supports the idea that coordinated influence is shaped by internal network architecture rather than external audience response.

This analysis offers empirical insight into the mechanics of coordinated amplification.

## Outputs Used in the Paper  
- Hybrid coordination graph  
- Eigenvector centrality per user  
- Regression results comparing centrality and influence  
- Network-wide structural measures  
- Group comparisons (high vs. low influence)

---
