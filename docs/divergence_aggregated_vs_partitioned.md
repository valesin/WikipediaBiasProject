# Why Aggregated (“all”) and Per-Language Plots Point in Different Directions

## Quick answer
The aggregated plots are built on a **different network with different denominators** than any single-language plot. The “all” layer sums weights across languages *before* scoring, shifts degree distributions, and applies filters differently. When bias is measured as “unexpected retention relative to marginals” (the Noise-Corrected backbone), pooling languages changes the null model enough to flip the sign or shrink the magnitude—classic Simpson’s paradox territory.

## Mechanisms in the current codebase that drive divergence

1. **Pre-sum of weights across languages**  
   * Code: `run_bias_analysis` creates an aggregated `all` DataFrame by summing weights across languages (`analysis.py`, lines 817‑825). This inflates degrees of high-visibility entities (often male, Western, English-visible) and redefines row/column totals (`ni.`, `n.j`, `n..`) used in the backbone p‑values.  
   * Effect: The baseline expectation for edges changes. Edges that look “over-retained” in a small language can look unremarkable, or even under-retained, once pooled.

2. **Noise-Corrected p‑value depends on marginals**  
   * Code: Backbone scores come from binomial CDF using row/column sums (`modules/backboning.py`, lines 131‑203). When degrees explode in the aggregated layer, the prior probability (`mean_prior_probability`) rises, reducing `score` for edges attached to dominant groups.  
   * Effect: Aggregation penalises hubs from over-represented groups, which can flatten or invert gender/region effects compared with language-specific plots where those hubs were extreme outliers.

3. **Pageview filters behave differently for “all”**  
   * Code: `restrict_by_pageviews_quantile` uses **per-language quantiles** but falls back to **sum of pageviews across languages** for the `all` layer (`filters.py`, lines 541‑554).  
   * Effect: The set of nodes that survive filtering in `all` is not the union of per-language survivors; it is biased toward cross-lingual celebrities, again skewing marginals.

4. **Minimum edge threshold masks asymmetries**  
   * Code: `min_edges` (default 500) removes sparse attribute pairs from curves and AUCs (`analysis.py`, lines 734‑770, 993‑1109). Languages with fewer edges for Female→Female may drop that pair entirely, while aggregation crosses the threshold and keeps it.  
   * Effect: Aggregated plots show symmetric matrices because all pairs clear the threshold; language plots drop low-support pairs, exaggerating asymmetry.

5. **Attribute-level NaN dropping is per attribute, not per dataset**  
   * Code: Rows with NaNs in the attribute under study are dropped before computing retention for that attribute (`analysis.py`, lines 507‑516). Metadata coverage varies by language.  
   * Effect: The analysed population for `gender` in a low-coverage language is smaller and differently composed than in the aggregated layer, changing base rates and retention curves.

6. **Simpson’s paradox from mixing languages with opposing trends**  
   * Research angle: Languages differ in gender balance, editorial norms, and coverage of regions/periods. Aggregating mixes them with weights proportional to edge volume. A small female advantage in a high-volume language can overwhelm a male advantage in low-volume languages, reversing the aggregate direction.

## How to interpret the differences

* **Aggregated “all” answers a different question**: “Given the multi-language network, how surprising is retention overall?” Per-language plots ask: “Given this language’s own structure, how surprising is retention?” The null model (expected weight given degrees) is different.
* **Directional flips are expected** when:  
  * Dominant languages differ in composition from smaller ones.  
  * Filters (pageviews, years) select different node sets per language than in aggregate.  
  * The backbone penalises high-degree hubs differently once edges are pooled.

## Suggestions to reconcile or co-report

1. **Report both and frame the question**: Clarify that “all” is a multilayer pooled backbone; per-language plots are layer-specific.  
2. **Weight languages equally**: Build an “all-balanced” layer by normalising weights per language before summing, or average AUCs across languages instead of pooling edges.  
3. **Mirror filters**: For pageviews, apply either per-language thresholds everywhere or create an aggregate threshold and apply it back to each language for symmetry.  
4. **Lower `min_edges` for small languages**: Avoid masking asymmetric pairs that only appear in aggregate.  
5. **Run backbone per language then average**: Compute NC scores separately and aggregate statistics (e.g., mean AUC) rather than aggregating edges first; this isolates Simpson’s paradox effects.

These adjustments make aggregated and partitioned views answer more comparable questions and reduce direction flips rooted in the underlying mathematics of the backbone and filtering steps.
