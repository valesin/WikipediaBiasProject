"""
Data filtering functions for Wikipedia bias analysis.

This module provides reusable filter functions that can be applied to enriched
edge DataFrames. Each filter function takes a DataFrame row and returns a boolean
indicating whether the row should be kept.

Usage:
    from src import filters

    # Create a year filter
    filter_1800s = filters.restrict_years(1800, 1900)

    # Apply to analysis
    results = analysis.run_bias_analysis(
        edge_csv="data.csv",
        meta_csv="meta.csv",
        pre_transform_filters=[filter_1800s]  # Applied to raw data
    )
"""

import pandas as pd


def restrict_years(min_year, max_year):
    """
    Factory function to create a year-based filter.

    Filters edges to only include those where BOTH source and target
    have birth years within the specified range (inclusive).

    Parameters:
    -----------
    min_year : int
        Minimum birth year (inclusive)
    max_year : int
        Maximum birth year (inclusive)

    Returns:
    --------
    function : A filter function that takes a DataFrame row and returns bool

    Example:
    --------
    >>> filter_modern = restrict_years(1750, 1950)
    >>> # Use with run_bias_analysis
    >>> results = run_bias_analysis(..., pre_transform_filters=[filter_modern])
    """

    def filter_func(row):
        birth_src = row.get("birth_source")
        birth_trg = row.get("birth_target")

        # Skip rows with missing birth data
        if pd.isna(birth_src) or pd.isna(birth_trg):
            return False

        # Convert to numeric in case they are stored as strings
        try:
            birth_src = float(birth_src)
            birth_trg = float(birth_trg)
        except (ValueError, TypeError):
            return False

        # Check if both source and target are within range
        return (min_year <= birth_src <= max_year) and (
            min_year <= birth_trg <= max_year
        )

    return filter_func


def modern_period_only(row):
    """
    Keep only modern period (post-1500) links.

    Filters edges where both source and target have birth years after 1500.

    Parameters:
    -----------
    row : pd.Series
        DataFrame row with 'birth_source' and 'birth_target' columns

    Returns:
    --------
    bool : True if both nodes are from modern period, False otherwise
    """
    birth_src = row.get("birth_source")
    birth_trg = row.get("birth_target")

    if pd.isna(birth_src) or pd.isna(birth_trg):
        return False

    return (birth_src > 1500) and (birth_trg > 1500)


def historic_period_only(row):
    """
    Keep only historic period (pre-1500) links.

    Filters edges where both source and target have birth years before or equal to 1500.

    Parameters:
    -----------
    row : pd.Series
        DataFrame row with 'birth_source' and 'birth_target' columns

    Returns:
    --------
    bool : True if both nodes are from historic period, False otherwise
    """
    birth_src = row.get("birth_source")
    birth_trg = row.get("birth_target")

    if pd.isna(birth_src) or pd.isna(birth_trg):
        return False

    return (birth_src <= 1500) and (birth_trg <= 1500)


def western_regions_only(row):
    """
    Keep only links between individuals from Western regions.

    Filters edges where both source and target are from Western Europe,
    Southern Europe, Northern Europe, or Northern America (per UN subregion classification).

    Parameters:
    -----------
    row : pd.Series
        DataFrame row with 'un_subregion_source' and 'un_subregion_target' columns

    Returns:
    --------
    bool : True if both nodes are from Western regions, False otherwise
    """
    western = {
        "Western Europe",
        "Southern Europe",
        "Northern Europe",
        "Northern America",
    }

    src = row.get("un_subregion_source")
    trg = row.get("un_subregion_target")

    if pd.isna(src) or pd.isna(trg):
        return False

    return src in western and trg in western


def same_gender_only(row):
    """
    Keep only same-gender links.

    Filters edges where source and target have the same gender.

    Parameters:
    -----------
    row : pd.Series
        DataFrame row with 'gender_source' and 'gender_target' columns

    Returns:
    --------
    bool : True if both nodes have the same gender, False otherwise
    """
    gender_src = row.get("gender_source")
    gender_trg = row.get("gender_target")

    if pd.isna(gender_src) or pd.isna(gender_trg):
        return False

    return gender_src == gender_trg


def cross_gender_only(row):
    """
    Keep only cross-gender links.

    Filters edges where source and target have different genders.

    Parameters:
    -----------
    row : pd.Series
        DataFrame row with 'gender_source' and 'gender_target' columns

    Returns:
    --------
    bool : True if nodes have different genders, False otherwise
    """
    gender_src = row.get("gender_source")
    gender_trg = row.get("gender_target")

    if pd.isna(gender_src) or pd.isna(gender_trg):
        return False

    return gender_src != gender_trg


def restrict_by_pageviews_quantile(pageviews_df, quantile):
    """
    Factory function to create a pageview-based filter using quantile thresholds.

    Filters edges to only include those where BOTH source and target
    have pageviews above the specified quantile threshold.

    Parameters:
    -----------
    pageviews_df : pd.DataFrame
        DataFrame with columns: wikidata_id, language_code, pageviews
        This should be the pageviews CSV loaded from the export
    quantile : float
        Quantile value between 0 and 1 (e.g., 0.8 means top 20%)

    Returns:
    --------
    function : A filter function that takes a DataFrame row and returns bool

    Example:
    --------
    >>> import pandas as pd
    >>> pageviews = pd.read_csv("data/pageviews_2023-01-01_to_2025-12-14.csv")
    >>> filter_top_20pct = restrict_by_pageviews_quantile(pageviews, 0.8)
    >>> # Use with run_bias_analysis
    >>> results = run_bias_analysis(..., pre_transform_filters=[filter_top_20pct])
    """
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be between 0 and 1")

    # Calculate the threshold value from the quantile
    threshold = pageviews_df["pageviews"].quantile(quantile)

    # Create a lookup dictionary for fast access
    # Key: (wikidata_id, language_code), Value: pageviews
    pageviews_lookup = {}
    for _, row in pageviews_df.iterrows():
        key = (row["wikidata_id"], row["language_code"])
        pageviews_lookup[key] = row["pageviews"]

    def filter_func(row):
        source_id = row.get("src")
        target_id = row.get("trg")
        language = row.get("language_code")

        # Skip rows with missing data
        if pd.isna(source_id) or pd.isna(target_id) or pd.isna(language):
            return False

        # Look up pageviews for source and target
        source_key = (source_id, language)
        target_key = (target_id, language)

        source_pageviews = pageviews_lookup.get(source_key, 0)
        target_pageviews = pageviews_lookup.get(target_key, 0)

        # Check if both are above threshold
        return (source_pageviews >= threshold) and (target_pageviews >= threshold)

    return filter_func


def restrict_by_pageviews_absolute(pageviews_df, min_pageviews):
    """
    Factory function to create a pageview-based filter using absolute threshold.

    Filters edges to only include those where BOTH source and target
    have pageviews >= min_pageviews.

    Parameters:
    -----------
    pageviews_df : pd.DataFrame
        DataFrame with columns: wikidata_id, language_code, pageviews
        This should be the pageviews CSV loaded from the export
    min_pageviews : float
        Minimum daily average pageviews required

    Returns:
    --------
    function : A filter function that takes a DataFrame row and returns bool

    Example:
    --------
    >>> import pandas as pd
    >>> pageviews = pd.read_csv("data/pageviews_2023-01-01_to_2025-12-14.csv")
    >>> filter_1000_views = restrict_by_pageviews_absolute(pageviews, 1000)
    >>> # Use with run_bias_analysis
    >>> results = run_bias_analysis(..., pre_transform_filters=[filter_1000_views])
    """
    if min_pageviews < 0:
        raise ValueError("min_pageviews must be non-negative")

    # Create a lookup dictionary for fast access
    # Key: (wikidata_id, language_code), Value: pageviews
    pageviews_lookup = {}
    for _, row in pageviews_df.iterrows():
        key = (row["wikidata_id"], row["language_code"])
        pageviews_lookup[key] = row["pageviews"]

    def filter_func(row):
        source_id = row.get("src")
        target_id = row.get("trg")
        language = row.get("language_code")

        # Skip rows with missing data
        if pd.isna(source_id) or pd.isna(target_id) or pd.isna(language):
            return False

        # Look up pageviews for source and target
        source_key = (source_id, language)
        target_key = (target_id, language)

        source_pageviews = pageviews_lookup.get(source_key, 0)
        target_pageviews = pageviews_lookup.get(target_key, 0)

        # Check if both are above threshold
        return (source_pageviews >= min_pageviews) and (
            target_pageviews >= min_pageviews
        )

    return filter_func


def exclude_edges_with_shared_names(custom_stopwords=None):
    """
    Factory function to exclude edges where titles share common words.

    This filter helps identify potential disambiguation errors where links
    might be spuriously inflated due to entities sharing surnames or common
    name components (e.g., "Charles Darwin" -> "Emma Darwin").

    Filters edges by checking if source and target titles share any
    non-stopword tokens. Returns False (exclude) when shared words are
    detected, True (keep) when titles don't share words.

    Parameters:
    -----------
    custom_stopwords : set, optional
        Custom set of stopwords to ignore when comparing titles.
        If None, uses a default set including common words, articles,
        and titles (king, emperor, saint, etc.).

    Returns:
    --------
    function : A filter function that takes a DataFrame row and returns bool

    Example:
    --------
    >>> # Exclude edges with shared surnames
    >>> filter_no_shared_names = exclude_edges_with_shared_names()
    >>> results = run_bias_analysis(..., pre_transform_filters=[filter_no_shared_names])
    >>>
    >>> # Use custom stopwords
    >>> my_stopwords = {'the', 'of', 'and'}
    >>> filter_custom = exclude_edges_with_shared_names(custom_stopwords=my_stopwords)
    """
    # Default stopwords: common words + titles/honorifics
    default_stopwords = {
        "the",
        "of",
        "and",
        "or",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "a",
        "an",
        "to",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "king",
        "emperor",
        "saint",
        "queen",
        "prince",
        "princess",
        "duke",
        "duchess",
        "sir",
        "lord",
        "lady",
        "mr",
        "mrs",
        "dr",
        "count",
        "baron",
        "earl",
        "marquis",
        "viscount",
        "pope",
        "father",
    }

    stopwords = custom_stopwords if custom_stopwords is not None else default_stopwords

    def filter_func(row):
        src_title = row.get("src_title")
        trg_title = row.get("trg_title")

        # Keep edges with missing title data (can't determine)
        if pd.isna(src_title) or pd.isna(trg_title):
            return True

        # Extract non-stopword tokens (length > 1) from both titles
        src_words = set(
            w
            for w in str(src_title).lower().split()
            if w not in stopwords and len(w) > 1
        )
        trg_words = set(
            w
            for w in str(trg_title).lower().split()
            if w not in stopwords and len(w) > 1
        )

        # Check for intersection of word sets
        shared_words = src_words & trg_words

        # Return False (exclude) if titles share words, True (keep) otherwise
        return len(shared_words) == 0

    return filter_func


def exclude_high_weight_edges(max_weight):
    """
    Factory function to exclude edges with abnormally high weights.

    This filter helps identify potential spurious edges where link counts
    might be inflated due to disambiguation errors or other data quality
    issues. Edges with weights at or above the threshold are excluded.

    Parameters:
    -----------
    max_weight : float
        Maximum acceptable edge weight (exclusive threshold).
        Edges with weight >= max_weight will be excluded.

    Returns:
    --------
    function : A filter function that takes a DataFrame row and returns bool

    Example:
    --------
    >>> # Exclude edges with weight >= 100
    >>> filter_weight_100 = exclude_high_weight_edges(100)
    >>> results = run_bias_analysis(..., pre_transform_filters=[filter_weight_100])
    >>>
    >>> # Combine with other filters
    >>> filter_weight_50 = exclude_high_weight_edges(50)
    >>> filter_shared_names = exclude_edges_with_shared_names()
    >>> results = run_bias_analysis(..., pre_transform_filters=[filter_weight_50, filter_shared_names])
    """
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")

    def filter_func(row):
        weight = row.get("nij")

        # Keep edges with missing weight data
        if pd.isna(weight):
            return True

        # Convert to numeric in case stored as string
        try:
            weight = float(weight)
        except (ValueError, TypeError):
            return True

        # Return False (exclude) if weight >= threshold, True (keep) otherwise
        return weight < max_weight

    return filter_func


def exclude_edges_with_relative_high_weight(
    edges_df, std_threshold=2.0, min_edges_for_stats=5
):
    """
    Factory function to exclude edges with anomalously high weights relative to node patterns.

    This filter performs per-node outlier detection by comparing each edge's weight
    to the statistical distribution of all edges for that node. This is more precise
    than absolute thresholds because it accounts for node-specific connectivity patterns.

    Use case: Identifying pseudo-self-loops (disambiguation errors where A mentions
    themselves but it's incorrectly recognized as A mentioning A', someone with a
    similar name). These often show as statistical outliers in the source node's
    edge weight distribution.

    Parameters:
    -----------
    edges_df : pd.DataFrame
        Full edges DataFrame with columns: 'source', 'target', 'nij'
        Must contain ALL edges to compute accurate per-node statistics
    std_threshold : float, default=2.0
        Number of standard deviations above the mean to consider an outlier.
        Common values: 2.0 (moderate), 2.5 (conservative), 1.5 (aggressive)
    min_edges_for_stats : int, default=5
        Minimum number of edges a node must have to compute statistics.
        Nodes with fewer edges will use global statistics instead.

    Returns:
    --------
    function : A filter function that takes a DataFrame row and returns bool

    Example:
    --------
    >>> import pandas as pd
    >>> from src import filters
    >>>
    >>> # Load full edge dataset
    >>> edges = pd.read_csv("weighted_links.csv")
    >>>
    >>> # Create filter with precomputed statistics
    >>> filter_relative = filters.exclude_edges_with_relative_high_weight(
    ...     edges_df=edges,
    ...     std_threshold=2.0,
    ...     min_edges_for_stats=5
    ... )
    >>>
    >>> # Use in analysis (on enriched dataset)
    >>> results = run_bias_analysis(
    ...     edges_data=enriched_edges,
    ...     meta_data=metadata,
    ...     pre_transform_filters=[filter_relative]
    ... )
    >>>
    >>> # Combine with title matching for targeted filtering
    >>> filter_names = filters.exclude_edges_with_shared_names()
    >>> results = run_bias_analysis(
    ...     edges_data=enriched_edges,
    ...     meta_data=metadata,
    ...     pre_transform_filters=[filter_names, filter_relative]
    ... )

    Notes:
    ------
    - Precomputes statistics during factory call (one-time cost)
    - Filter function lookups are O(1) per row
    - Conservative with missing data: keeps edges when stats unavailable
    - Uses source node statistics (outgoing edge weights from source)
    """
    import numpy as np

    if std_threshold <= 0:
        raise ValueError("std_threshold must be positive")
    if min_edges_for_stats < 2:
        raise ValueError("min_edges_for_stats must be at least 2")

    # Precompute per-node statistics for all source nodes
    node_stats = {}

    # Group by source node and calculate statistics
    for source_node, group in edges_df.groupby("source"):
        weights = group["nij"].values

        # Only compute stats if node has enough edges
        if len(weights) >= min_edges_for_stats:
            node_stats[source_node] = {
                "mean": np.mean(weights),
                "std": np.std(weights),
                "count": len(weights),
            }

    # Fallback: global statistics for nodes with insufficient data
    global_mean = edges_df["nij"].mean()
    global_std = edges_df["nij"].std()

    def filter_func(row):
        source = row.get("source")
        weight = row.get("nij")

        # Keep edges with missing data
        if pd.isna(source) or pd.isna(weight):
            return True

        # Convert weight to numeric
        try:
            weight = float(weight)
        except (ValueError, TypeError):
            return True

        # Get statistics for this source node
        if source in node_stats:
            stats = node_stats[source]
            mean = stats["mean"]
            std = stats["std"]

            # Avoid division by zero for constant-weight nodes
            if std == 0:
                return True

            # Calculate z-score
            z_score = (weight - mean) / std

            # Return False (exclude) if weight is outlier, True (keep) otherwise
            return z_score < std_threshold
        else:
            # Use global statistics for nodes with insufficient edges
            if global_std == 0:
                return True

            z_score = (weight - global_mean) / global_std
            return z_score < std_threshold

    return filter_func


def restrict_by_pageviews_quantile_global(pageviews_df, quantile):
    """
    Factory function for global (multilayer) pageview-based filtering using quantile.

    Aggregates pageviews across all languages and filters to nodes above the specified
    quantile threshold. This creates a consistent node set across all language editions
    (multilayer network).

    **Difference from language-specific filtering**:
    - Language-specific: Different threshold per language based on per-language quantile
    - Global: Single threshold based on aggregated pageviews across ALL languages

    Parameters:
    -----------
    pageviews_df : pd.DataFrame
        DataFrame with columns: wikidata_id, language_code, pageviews
        Will be aggregated by wikidata_id (summing pageviews across languages)
    quantile : float
        Quantile value between 0 and 1 (e.g., 0.8 means top 20%)

    Returns:
    --------
    function : A filter function that takes a DataFrame row and returns bool

    Example:
    --------
    >>> import pandas as pd
    >>> from src import filters
    >>>
    >>> # Load pageviews
    >>> pageviews = pd.read_csv("data/out/pageviews_2023-01-01_to_2025-12-14.csv")
    >>>
    >>> # Create global filter (top 20% by aggregated pageviews)
    >>> filter_top_20_global = filters.restrict_by_pageviews_quantile_global(pageviews, 0.8)
    >>>
    >>> # Use with run_bias_analysis
    >>> results = run_bias_analysis(..., pre_transform_filters=[filter_top_20_global])
    """
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be between 0 and 1")

    # Aggregate pageviews across all languages
    pageviews_agg = pageviews_df.groupby("wikidata_id")["pageviews"].sum().reset_index()
    pageviews_agg.columns = ["wikidata_id", "pageviews_total"]

    # Calculate the threshold value from the quantile
    threshold = pageviews_agg["pageviews_total"].quantile(quantile)

    # Create a lookup dictionary for fast access
    # Key: wikidata_id, Value: aggregated pageviews
    pageviews_lookup = dict(
        zip(pageviews_agg["wikidata_id"], pageviews_agg["pageviews_total"])
    )

    def filter_func(row):
        source_id = row.get("src")
        target_id = row.get("trg")

        # Skip rows with missing data
        if pd.isna(source_id) or pd.isna(target_id):
            return False

        # Look up aggregated pageviews for source and target (language-independent)
        source_pageviews = pageviews_lookup.get(source_id, 0)
        target_pageviews = pageviews_lookup.get(target_id, 0)

        # Check if both are above threshold
        return (source_pageviews >= threshold) and (target_pageviews >= threshold)

    return filter_func


def restrict_by_pageviews_absolute_global(pageviews_df, min_pageviews):
    """
    Factory function for global (multilayer) pageview-based filtering using absolute threshold.

    Aggregates pageviews across all languages and filters to nodes with aggregated
    pageviews >= min_pageviews. This creates a consistent node set across all language
    editions (multilayer network).

    **Difference from language-specific filtering**:
    - Language-specific: Per-language threshold (node must have >= min_pageviews in THAT language)
    - Global: Single threshold based on aggregated pageviews across ALL languages

    Parameters:
    -----------
    pageviews_df : pd.DataFrame
        DataFrame with columns: wikidata_id, language_code, pageviews
        Will be aggregated by wikidata_id (summing pageviews across languages)
    min_pageviews : float
        Minimum aggregated daily average pageviews required

    Returns:
    --------
    function : A filter function that takes a DataFrame row and returns bool

    Example:
    --------
    >>> import pandas as pd
    >>> from src import filters
    >>>
    >>> # Load pageviews
    >>> pageviews = pd.read_csv("data/out/pageviews_2023-01-01_to_2025-12-14.csv")
    >>>
    >>> # Create global filter (>= 5000 aggregated pageviews)
    >>> filter_5000_global = filters.restrict_by_pageviews_absolute_global(pageviews, 5000)
    >>>
    >>> # Use with run_bias_analysis
    >>> results = run_bias_analysis(..., pre_transform_filters=[filter_5000_global])
    """
    if min_pageviews < 0:
        raise ValueError("min_pageviews must be non-negative")

    # Aggregate pageviews across all languages
    pageviews_agg = pageviews_df.groupby("wikidata_id")["pageviews"].sum().reset_index()
    pageviews_agg.columns = ["wikidata_id", "pageviews_total"]

    # Create a lookup dictionary for fast access
    # Key: wikidata_id, Value: aggregated pageviews
    pageviews_lookup = dict(
        zip(pageviews_agg["wikidata_id"], pageviews_agg["pageviews_total"])
    )

    def filter_func(row):
        source_id = row.get("src")
        target_id = row.get("trg")

        # Skip rows with missing data
        if pd.isna(source_id) or pd.isna(target_id):
            return False

        # Look up aggregated pageviews for source and target (language-independent)
        source_pageviews = pageviews_lookup.get(source_id, 0)
        target_pageviews = pageviews_lookup.get(target_id, 0)

        # Check if both are above threshold
        return (source_pageviews >= min_pageviews) and (
            target_pageviews >= min_pageviews
        )

    return filter_func
