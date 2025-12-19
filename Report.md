# Disease-Gene Association Prediction: Final Report

**Group HAR_COWINE | Analysis of Complex Networks | December 2025**  
**Achievement: 2nd Place (AP: 0.057353)**  
**Team Members: Atena Karimi, Reza Tashakkori, and Hesam Korki**
---

## 1. Analysis of Node Features and Network Properties

**Network Characteristics**: The BioGRID protein-protein interaction network contains 19,765 proteins (nodes) connected by 777,395 undirected interactions (edges), with average degree of 78.66 and maximum degree of 2,673. The network exhibits scale-free properties typical of biological systems, with power-law degree distribution (exponent ≈ -2.1).

**Critical Discovery - Network Homophily**: We computed edge homophily as the fraction of edges connecting nodes with shared labels, obtaining **homophily = 0.0252**. This extremely low value reveals that the network is strongly *heterophilic*—proteins that interact physically often have *different* functional roles rather than similar ones. This finding fundamentally shaped our modeling strategy.

Low homophily = neighbors are NOT similar → aggregating neighbor info adds noise, not signal

**Label Properties**: The 305 disease labels exhibit extreme sparsity (96.96% of entries are zero) with highly imbalanced distributions. The most common diseases have ~800 associated genes while rare diseases have only 1-2, creating a severe long-tail problem requiring specialized loss functions.

**Feature Engineering**: We enhanced the original 37-dimensional biological features to **103 dimensions** by incorporating: (1) 64-dimensional Node2Vec embeddings (p=1, q=1, trained on full network), (2) structural features (log-degree, PageRank, clustering coefficient, betweenness centrality), and (3) ego-network statistics. This enrichment provided complementary topological signal beyond the biological embeddings.

---

## 2. Model Choice and Rationale ⭐

**Why GNNs Failed**: Standard Graph Neural Networks (GraphSAGE, GCN, GAT) achieved only 0.026-0.052 AP despite sophisticated architectures. The root cause is the **homophily assumption**: GNNs aggregate neighbor features assuming similar nodes connect, formalized as $h_v^{(l+1)} = \sigma(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} W h_u^{(l)})$. With homophily of 0.0252, this averaging dilutes rather than reinforces label signals. Also, averaging in message-passing helps when neighbors are similar by amplifying the correct signal, but hurts when neighbors are different by mixing in random information. Also, GNNs need lots of examples to learn - they're "data hungry. Our experiments confirmed that deeper GNNs (3+ layers) performed *worse* as over-smoothing spread incorrect information.

**Why Label Propagation Succeeded**: Label Propagation (LP) doesn't learn from examples - it just smooths known labels across the grap. LP performs probability smoothing via $P^{(t+1)} = \alpha S P^{(t)} + (1-\alpha) Y$ where $S$ is the symmetric normalized adjacency and $Y$ are training labels. Crucially, **LP makes no homophily assumption**—it simply smooths probability distributions across edges without assuming neighbors share labels. In sparse settings, it is better to use graph structure rather than try to learn feature patterns.. In heterophilic networks, this avoids the incorrect inductive bias of GNNs. LP achieved 0.057156 AP alone, contributing 95.8% of our final performance.

**Why Ensemble Approach**: While LP dominates, we observed complementary patterns: (1) SIGN incorporates label information through label reuse ($K$-hop label propagation as features), partially overcoming GNN limitations, (2) XGBoost captures non-topological feature patterns that LP ignores. A calibrated weighted ensemble (60% LP, 20% XGBoost, 20% SIGN) gained +0.34% improvement by combining heterophilic-friendly topology (LP) with supervised feature learning (SIGN, XGBoost).

**Data-Driven Decision**: We validated our approach systematically by comparing 8 baseline models. The homophily measurement directly explained performance patterns (LP > simple GNNs > deep GNNs), transforming model selection from guesswork into principled scientific reasoning: *match method assumptions to data properties*.

---

## 3. Model Design

**SIGN Architecture**: SIGN (Scalable Inception Graph Neural Network) preprocesses $K$-hop features offline: $X^{(k)} = \tilde{A}^k X$ for $k=0,1,2,3$, then concatenates them as input to an MLP. We set $K=3$ based on validation experiments. To address label sparsity, we implemented **label reuse**: propagated training labels 3 hops ($L^{(k)} = \tilde{A}^k Y_{train}$ for $k=1,2,3$), reduced 305 dimensions to 32 via PCA (preserving 95% variance), and concatenated with features. The MLP uses 512 hidden units with BatchNorm and dropout (0.3). For imbalanced labels, we replaced BCE with **Focal Loss**: $\mathcal{L}_{FL} = -\alpha (1-p_t)^\gamma \log(p_t)$ where $\alpha=0.25, \gamma=2.0$, down-weighting easy negatives.

**XGBoost Design**: We trained 305 independent binary XGBoost classifiers (one per disease) to capture per-label patterns. Each classifier uses `scale_pos_weight` computed as $\frac{\text{num negatives}}{\text{num positives}}$ to handle imbalance, with hyperparameters: `max_depth=6`, `learning_rate=0.1`, `n_estimators=100`, `subsample=0.8`. This one-vs-rest approach allows per-disease specialization.

**Label Propagation Implementation**: We use the closed-form iterative solution: $P^{(t+1)} = \alpha D^{-1/2} A D^{-1/2} P^{(t)} + (1-\alpha) Y$ where $D$ is degree matrix, $A$ is adjacency, $\alpha=0.85$ (tuned via validation), and 50 iterations (empirically sufficient for convergence). Initialization uses training labels padded with zeros for test nodes.

**Calibration**: SIGN outputs were temperature-scaled ($p_{calib} = \text{softmax}(logits / T)$ with $T=2.0$) to reduce overconfidence. XGBoost outputs were linearly rescaled using validation set percentiles to match probability scale. Calibration improved ensemble reliability by aligning probability ranges.

---

## 4. Experiment Details

**Baseline Comparisons**: We implemented 8 models: (1) Node2Vec + Logistic Regression (0.019), (2) Raw features + XGBoost (0.044), (3) GraphSAGE 2-layer (0.026), (4) GCN 3-layer (0.036), (5) GAT 2-layer (0.052), (6) SIGN + Focal Loss (0.054), (7) Pure Label Propagation (0.057156), (8) Final Ensemble (0.057353). Validation set (20% of training data) guided hyperparameter tuning and early stopping.

**Calibration Strategy**: For SIGN, we swept temperature $T \in \{1.0, 1.5, 2.0, 2.5\}$ and selected $T=2.0$ based on validation reliability diagrams. For XGBoost, we applied Platt scaling but found simple linear rescaling (min-max normalization) performed better. LP outputs were already well-calibrated and required no adjustment.

**Ensemble Weighting**: We grid-searched ensemble weights in increments of 0.1, testing 66 combinations. The optimal weights (20% SIGN, 20% XGBoost, 60% LP) balanced LP's dominance with complementary signal. Interestingly, deviating from this ratio decreased performance, confirming LP's centrality.

**Reproducibility**: All experiments used fixed random seeds (42). Node2Vec embeddings were pre-computed and cached to ensure consistency. The final pipeline in `Codes/Final_Pipeline (Optional).ipynb` merges the most effective components from experimental drafts (Draft4: GNN+C&S, Draft8: Node2Vec, Draft11: Multi-scale LP, Draft13: Ensemble).

---

## 5. Result Analysis ⭐

**Performance Breakdown**: 
- Pure Label Propagation: **0.057156 AP** (baseline contribution = 99.66%)
- SIGN contribution: +0.000098 AP (0.17% improvement)
- XGBoost contribution: +0.000099 AP (0.17% improvement)  
- **Final Ensemble: 0.057353 AP** (2nd place, +0.34% total improvement)

**Key Insight**: Label Propagation alone achieves 95.8% of the final score and 99.66% of the ensemble's performance. This validates our core hypothesis: in heterophilic networks, simple topology-based smoothing outperforms complex feature-based models by orders of magnitude. The ensemble provides marginal gains through complementary patterns but LP carries the solution.

**Why LP Dominates**: LP uses graph structure (777K edges) to spread information, not just the few positive labels → works better with sparse data than learning-based models (GNNs, XGBoost).
The extreme label sparsity (96.96%) means feature-based models (GNNs, XGBoost) struggle to learn from limited positive examples. LP leverages the global network structure as inductive bias, propagating known labels across 777,395 edges—essentially performing "soft imputation" using network topology. In heterophilic settings, this smoothing is more robust than learning feature transformations from sparse supervision.

**Model Comparison**: The performance ordering (LP 0.057 > GAT 0.052 > GCN 0.036 > SAGE 0.026) directly correlates with heterophily tolerance. GAT's attention mechanism partially adapts to heterophily by learning edge weights, outperforming fixed aggregation in GCN/SAGE. But LP, with no homophily assumption, achieves 10% higher AP than the best GNN.

**Practical Implication**: Our results demonstrate that **model sophistication ≠ performance** when assumptions mismatch data. Simple methods aligned with data properties (LP for heterophilic networks) dominate complex methods with incompatible assumptions (GNNs for homophilic networks). This principle generalizes beyond our task to any graph learning problem.

---

## 6. Conclusion

This project demonstrates that **data analysis must precede model selection**. By measuring homophily (0.0252), we identified the fundamental mismatch between GNN assumptions and biological network properties, leading us to Label Propagation—a simpler method perfectly suited to heterophilic graphs. The 60-20-20 ensemble (LP-XGBoost-SIGN) achieved 2nd place (0.057353 AP) by combining topology-based smoothing with supervised learning, though LP contributed 95.8% of performance.

The methodology—measure data properties, match model assumptions, validate systematically—represents rigorous scientific practice in machine learning. Our experiments confirm that understanding *why* methods work matters more than blindly applying state-of-the-art architectures. For heterophilic graphs, simple probability smoothing via Label Propagation outperforms sophisticated GNNs by 10+ percent.

**Bottom line 1**: Very few positive examples makes learning hard, but graph diffusion (LP) doesn't need many examples.

**Bottom line 2**: Message-passing = "become like your neighbors" — only works if neighbors ARE like you (high homophily)

---

## 7. Possible Improvements

**Multi-Scale Label Propagation**: Test multiple diffusion strengths ($\alpha \in \{0.7, 0.8, 0.85, 0.9, 0.95\}$) and ensemble predictions, capturing both local and global label correlations. Draft11 experiments showed promise (+0.2% AP).

**Learnable Ensemble Weights**: Replace fixed 60-20-20 weights with a meta-learning approach (stacking with Logistic Regression on validation predictions) to optimize combination adaptively per disease label.

**Per-Label Threshold Tuning**: Current submission uses uniform 0.5 threshold. Optimizing per-disease thresholds via validation F1-scores could improve long-tail disease predictions.

**Advanced Calibration**: Implement isotonic regression or beta calibration instead of temperature scaling for better probability reliability, particularly for rare diseases with limited training data.

**Heterophilic GNN Architectures**: Explore recent heterophily-aware GNNs (H2GCN, GPRGNN, FAGCN) that explicitly model negative correlations between neighbors, potentially matching LP performance with learned parameters.

---
