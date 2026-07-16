# Eigenvector Centrality in Coordinated Disinformation Networks - Project Cardinal  
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
## Known IRA Operatives
# The following individuals were identified and indicted by the U.S. Department of Justice for their roles in IRA operations (Mueller Report, 2019):

- Mikhail Ivanovich Bystrov
- Igor Vladimirovich Nesterov
- Robert Sergeyevich Bovda
- Dzheykhun Nasimi Ogly Aslanov
- Vladimir Dmitriyevich Venkov
- Mikhail Leonidovich Burchik
- Anna Vladislavovna Bogacheva
- Aleksandra Yuryevna Krylova
- Irina Viktorovna Kaverzina
- Vadim Vladimirovich Podkopaev
- Sergey Pavlovich Polozov
- Taras Kirillovich Pribyshin
- Gleb Igorevich Vasilchenko
- Maria Anatolyevna Bovda
- Denis Igorevich Kuzmin

*Note: These individuals were charged with conspiracy to defraud the United States through their coordination of the IRA's social media influence operations during the 2016 U.S. presidential election.*

---
## Operation Sites
<img width="883" height="652" alt="image" src="https://github.com/user-attachments/assets/58846006-1e4a-481a-be55-ee01824eebbe" />

## Findings

The analysis characterizes the **structural organization** of the IRA network — specifically, whether eigenvector centrality distinguishes between accounts that primarily *originate* content and accounts that primarily *amplify* others.

### Setup

Two OLS regressions per subgroup, predicting `retweet_behavior_proportion` (the fraction of a user's tweets flagged as retweets in the FiveThirtyEight dataset — see Limitations for why this is the outcome variable):

- **Baseline:** `behavior ~ avg_followers`
- **Full:** `behavior ~ avg_followers + eigenvector_centrality`

Users were split at the median of retweet behavior into two subgroups: **amplifier-leaning** (high retweet proportion) and **originator-leaning** (low retweet proportion).

### Results

| Subgroup | n | R² (followers only) | R² (+ eigenvector) | ΔR² | p |
|---|---|---|---|---|---|
| Amplifier-leaning | 1,096 | 0.004 | 0.034 | 0.0295 | 1.0e-8 |
| Originator-leaning | 965 | 0.047 | 0.067 | 0.0208 | 4.2e-6 |

Eigenvector centrality adds **41.9% more incremental explanatory power** in the amplifier-leaning subgroup than in the originator-leaning subgroup. Both improvements are highly statistically significant.

### Interpretation

The IRA network can be decomposed into two structural roles: **content originators** and **amplifiers**. Eigenvector centrality distinguishes these roles — amplifier-leaning accounts occupy more centrally-embedded positions in the hashtag co-occurrence graph than originator-leaning accounts do. Follower count barely distinguishes the two roles at the top of the amplification tier (baseline R² = 0.004 for amplifier-leaning accounts); network position does.

This is consistent with amplification being the structural function of a coordinated influence network. Originators — the accounts producing the material to be amplified — are distributed across the periphery. Amplifiers cluster centrally because that's what makes them structurally effective at their job.

The eigenvector coefficient is negative in both subgroups, which fits this interpretation: within each subgroup, higher centrality is associated with slightly *lower* retweet-behavior proportion, meaning the most centrally-embedded accounts are more likely to be doing original posting rather than pure amplification. This suggests the coordination hub is not a monolithic layer of retweeters but includes structurally central originators driving the material.

Full regression outputs are in `results/all_results.json`. Figures in `results/figures/`.

## Limitations & Future Work

**The outcome variable measures posting behavior, not received engagement.** The FiveThirtyEight dataset provides a boolean `retweet` flag on each tweet (was this post a retweet of someone else) but does not provide per-tweet received-retweet counts. Aggregating the flag per user therefore produces a *retweet behavior proportion* — how much of the user's activity is amplification vs. origination — not a measure of how much the user *was retweeted by others*.

This constrains the finding. The current analysis characterizes the **structural role** of accounts in the coordinated network (amplifier vs. originator, and how centrality maps to that role). It does **not** measure whether high-centrality accounts received more audience engagement or drove more downstream propagation. Those are related but distinct questions.

**Planned follow-up work:**

1. **Received-engagement outcome variable.** Reconstruct per-user received-retweet counts by extracting retweet targets from `article_url` and shortened-URL fields in the raw data where available. Coverage will be partial — FiveThirtyEight's collection normalized the `RT @username:` prefix out of tweet text — but even partial reconstruction enables a proper received-engagement regression on the subset of tweets with recoverable targets.
2. **Replication on a comparable dataset with native engagement metrics.** A follow-up analysis using a Twitter data release that includes per-tweet `retweet_count` and `favorite_count` (e.g., post-2022 archive releases) would allow the eigenvector-vs-follower comparison to run against the outcome variable this project originally intended.
3. **Robustness across coordination-network types.** The structural role finding is currently established on one operation (IRA 2015–2017). Extending it to other documented coordinated networks would test whether the amplifier-cluster-in-central-positions pattern is a general property of state-directed influence operations or specific to this one.
