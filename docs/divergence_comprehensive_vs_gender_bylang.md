# Why the Comprehensive Pipeline Plots Diverge from `02_gender_bylang.py`

> Scope: Evidence and code-path analysis for the large differences between the plots produced by `notebooks/comprehensive_bias_analysis_plots.ipynb` and the earlier `notebooks/michele/02_gender_bylang.py` workflow. The latter file is not present in the current repository history, so the comparison below isolates the concrete behaviours of the comprehensive pipeline and highlights the code-level mechanisms that would diverge from a simpler per-language gender script.

## What the current comprehensive notebook actually computes

* **Input handling** – `comprehensive_bias_analysis_plots.ipynb` loads the Spotlight-weighted edges and metadata, drops self-loops, and calls `analysis.run_bias_analysis` (see notebook cells 4, 6, 9).  
* **Aggregation step** – `run_bias_analysis` **adds a synthetic `all` language** by summing weights across languages *before* any scoring (`analysis.py`, lines 817‑825). Every subsequent calculation is therefore executed both on per-language slices and on this pooled layer.  
* **Backbone score, not raw weight** – For each language slice, edges are re‑weighted with the **Noise-Corrected backbone** (binomial CDF–based p‑value) (`analysis.py`, line 857; `modules/backboning.py`, lines 131‑203). The score is the tail probability of observing \(nij\) given row/column totals, not the raw edge frequency.  
* **Attribute enrichment and transformation** – Node attributes are merged and transformed: `bigperiod_birth` is collapsed into five bins and `un_subregion` is collapsed into `Western`/`Non‑Western` (`analysis.py`, lines 864‑909). Rows with NaNs for the analysed attribute are dropped *per attribute* (`analysis.py`, lines 507‑516).  
* **Threshold grid and retention definitions** – Thresholds are **log-spaced extremely close to 1** (10^{-14}→1; `analysis.py`, lines 523‑525).  
  * Edge retention is the fraction of edges with `score >= t` per source→target attribute pair, and AUC is the trapezoid of that curve (`analysis.py`, lines 218‑255).  
  * Node retention is computed with the **max-edge approach**: a node is retained at threshold \(t\) if its *highest* incident backbone score exceeds \(t\) (`analysis.py`, lines 399‑455).  
* **Reliability gate** – Any pair/value with `< min_edges` (default 500) is masked for curves and for AUC heatmaps (`analysis.py`, lines 734‑770, 993‑1109). Sparse groups vanish entirely.

These choices together yield AUCs that are usually ~0.99 (visible in cached CSVs from `bias_analysis_pageview_filtered.ipynb`), because scores are already near 1 and the threshold grid is concentrated near 1. Small differences in backbone scores translate into visually tiny separations in curves but still integrate to large AUC differences.

## Likely points of divergence from `02_gender_bylang.py`

Although the historical `02_gender_bylang.py` file is not in the repository, the following concrete behaviours in the comprehensive pipeline would materially diverge from a simpler “gender by language” script and explain direction/scale changes in plots:

1. **Backbone vs. raw weight or proportions**  
   * Comprehensive: uses noise-corrected **p‑values** (`score`) driven by row/column marginals, not raw counts. This emphasises *unexpectedness* rather than magnitude.  
   * A basic gender-by-language script typically uses raw edge weights, degree-normalised proportions, or simple ratios of female↔male edges. Switching from p‑value significance to magnitude reverses which edges rank highest and will flip or flatten retention curves.  
   * Evidence: `analysis.py` line 857 calls `noise_corrected(..., calculate_p_value=True)`; AUC is then computed over `score` (`analysis.py` lines 218‑255).

2. **Pre-aggregation across languages**  
   * Comprehensive: synthesises an `all` layer by *summing weights across languages first* (`analysis.py` lines 817‑825). This boosts English-heavy or cross-language-popular nodes and changes both the marginals (ni., n.j, n..) and the derived p‑values for every edge.  
   * A per-language script would analyse each language independently with its native weight distribution, so the global “all” patterns (often male/Western-heavy) leak into every comparison in the comprehensive pipeline when plots mix aggregated and sliced outputs.

3. **Strict min-edge gating**  
   * Comprehensive: drops any attribute pair with <500 edges from both curves and heatmaps (e.g., Female→Other often disappears). This removes the long tail and leaves only the densest relations, inflating AUCs and reducing asymmetry.  
   * A simpler script might include all pairs, leading to noisier but more asymmetric curves. The absence of low-support pairs in the comprehensive plots can look like bias “vanishing.”

4. **Attribute recoding and NaN dropping**  
   * Comprehensive collapses `un_subregion` to Western/Non-Western and `bigperiod_birth` into 5 buckets, then drops rows where *either* endpoint has NaN for that attribute (`analysis.py` lines 507‑516).  
   * If `02_gender_bylang.py` worked on uncollapsed regions or did not drop NaNs per attribute, the analysed population differs, altering the base rates used in bias calculations.

5. **Node-retention metric choice**  
   * Comprehensive uses the **max-edge** method (retain node if any incident edge survives threshold; `analysis.py` lines 399‑455). This favours high-degree nodes and dampens attrition for dominant groups.  
   * A sweep-based or degree-weighted node retention (common in quick scripts) counts nodes that stay connected after removing sub-threshold edges overall, not just via their strongest tie, producing steeper drop-offs and different AUC ordering.

6. **Log-space thresholds near 1**  
   * The grid emphasises 0.999999… thresholds (`analysis.py` lines 523‑525). If `02_gender_bylang.py` used a linear grid over [0,1] or a fixed density cutoff, the resulting curves/AUCs would emphasise different parts of the distribution, easily reversing “which curve is above.”

7. **Filtering of zero weights and self-loops**  
   * Comprehensive explicitly removes zero weights before scoring (`analysis.py` line 812) and self-loops earlier in the notebook. Scripts that include them increase low-score mass, steepening curves in the opposite direction.

## Practical checks to align both analyses

To make an apples-to-apples comparison with the historical script, configure the comprehensive notebook (or a helper script) as follows:

1. **Disable the aggregated layer**: call `run_bias_analysis(..., add_aggregated_all=False)` to prevent “all” leakage into language slices.  
2. **Use raw weights**: replace the backbone call with `score = nij` (or set `calculate_p_value=False` and operate on `nij`) and re-compute retention on weights.  
3. **Match thresholds**: switch to a linear threshold grid over [0, max_weight] or reuse the grid from `02_gender_bylang.py` if available.  
4. **Lower or remove `min_edges`**: set `min_edges=1` to keep sparse gender pairs and replicate the earlier script’s coverage.  
5. **Keep original attribute categories**: skip `_transform_region` and `_transform_period` to mirror any uncollapsed categories used previously.  
6. **Choose the sweep-based node retention**: swap `_calculate_node_retention_by_max_edge` for `_calculate_node_retention_by_sweep` if the earlier script counted nodes differently.

Running the comprehensive pipeline with these toggles will isolate the exact mathematical levers responsible for the divergent plots and should bring the curves much closer to the historical outputs.
