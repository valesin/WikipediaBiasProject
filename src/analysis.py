"""
Bias Analysis Pipeline for Wikipedia Network Data

This module provides a complete pipeline for analyzing retention bias in Wikipedia
link networks across different demographic attributes (gender, birth period, region).
"""

import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom
import itertools

# Import cache utilities for consistent cache key generation
from . import cache_utils


def _transform_period(period_value):
    """
    Transform bigperiod_birth values to cleaner period names.

    Removes year ranges and 'period' suffix for readability.

    Args:
        period_value: Raw bigperiod_birth value from dataset

    Returns:
        str or NaN: Simplified period name, or NaN if missing/unknown
    """
    # Return NaN for missing values
    if pd.isna(period_value) or period_value == "Missing":
        return np.nan

    period_mapping = {
        "5.Contemporary period 1901-2020AD": "Contemporary",
        "4.Mid Modern Period 1751-1900AD": "Mid Modern",
        "3.Early Modern Period 1501-1750AD": "Early Modern",
        "2.Post-Classical History 501-1500AD": "Post-Classical",
        "1.Ancient History Before 500AD": "Ancient",
    }
    # Return NaN for unknown values not in mapping
    return period_mapping.get(period_value, np.nan)


def _transform_region(region_value):
    """
    Transform un_subregion values into Western/Non-Western classification.

    Western regions include Europe and Northern America.
    All other regions are classified as Non-Western.

    Args:
        region_value: UN subregion value from dataset

    Returns:
        str or NaN: 'Western', 'Non-Western', or NaN if missing/unknown
    """
    # Return NaN for missing values
    if pd.isna(region_value):
        return np.nan

    western_regions = {
        "Western Europe",
        "Northern America",
        "Eastern Europe",
        "Southern Europe",
        "Northern Europe",
        "Oceania Western World",
    }

    non_western_regions = {
        "SouthEast Asia",
        "South America",
        "Central Africa",
        "Western Asia (Middle East Caucasus)",
        "Eastern Asia",
        "South Asia incl. Indian Peninsula",
        "North Africa",
        "Oceania not Aus Nze",
        "Caribbean",
        "West Africa",
        "Central America",
        "East Africa",
        "Central Asia",
        "Southern Africa",
    }

    if region_value in western_regions:
        return "Western"
    elif region_value in non_western_regions:
        return "Non-Western"
    else:
        # Return NaN for unexpected/unknown values
        return np.nan


def partition_edges_by_language(edges_df, language_col="language_code", languages=None):
    """
    Partition edges DataFrame by language.

    Parameters:
    -----------
    edges_df : pd.DataFrame
        Edge dataframe with language column
    language_col : str
        Name of the language column
    languages : list, optional
        List of languages to extract. If None, uses all available languages.

    Returns:
    --------
    dict : {language: DataFrame}
    """
    if languages is None:
        languages = edges_df[language_col].dropna().unique()

    return {
        lang: edges_df[edges_df[language_col] == lang].reset_index(drop=True)
        for lang in languages
        if lang in edges_df[language_col].values
    }


def merge_node_metadata(edges_df, meta_df, columns, meta_id_col="wikidata_id"):
    """
    Merge node metadata onto edges without applying transformations.

    Args:
        edges_df (pd.DataFrame): DataFrame with at least 'src' and 'trg' columns.
        meta_df (pd.DataFrame): DataFrame with node metadata.
        columns (list): List of column names to merge from metadata.
        meta_id_col (str): Column name for node IDs in metadata (default: 'wikidata_id').

    Returns:
        pd.DataFrame: edges_df with {colname}_source and {colname}_target columns.
    """
    meta = meta_df.copy()
    edges_df = edges_df.copy()

    # Set index once for efficient mapping
    meta_indexed = meta.set_index(meta_id_col)

    # Merge each column for source and target
    for col in columns:
        if col not in meta_indexed.columns:
            logger.warning(f"Column '{col}' not found in metadata, skipping")
            continue
        edges_df[f"{col}_source"] = edges_df["src"].map(meta_indexed[col])
        edges_df[f"{col}_target"] = edges_df["trg"].map(meta_indexed[col])

    return edges_df


def transform_node_attributes(edges_df, transformations):
    """
    Apply transformations to merged node attributes in-place.

    Transformations modify the existing {colname}_source and {colname}_target columns.
    For example, transforms raw birth year values into period categories.

    Args:
        edges_df (pd.DataFrame): DataFrame with merged metadata columns.
        transformations (list): List of (column_name, transform_function) tuples.

    Returns:
        pd.DataFrame: edges_df with transformed attribute columns.
    """
    edges_df = edges_df.copy()

    for col, transform_func in transformations:
        source_col = f"{col}_source"
        target_col = f"{col}_target"

        if source_col not in edges_df.columns:
            logger.warning(
                f"Column '{source_col}' not found in edges, skipping transformation for '{col}'"
            )
            continue

        # Apply transformation in-place
        edges_df[source_col] = edges_df[source_col].apply(transform_func)
        edges_df[target_col] = edges_df[target_col].apply(transform_func)

    return edges_df


def _calculate_edge_retention(s_table, attr, attr_values, thresholds, min_edges_len):
    """
    Calculate edge retention curves for all source-target pairs of an attribute.

    Parameters:
    -----------
    s_table : pd.DataFrame
        Table with backbone scores and enriched attributes
    attr : str
        Attribute name to analyze
    attr_values : list
        List of unique attribute values to analyze
    thresholds : np.ndarray
        Array of threshold values to evaluate
    min_edges_len : int
        Minimum number of edges required for a group to be analyzed

    Returns:
    --------
    tuple : (edge_results_list, auc_matrix)
        - edge_results_list: List of dicts with edge retention data for each pair
        - auc_matrix: DataFrame with AUC values for each source-target pair
    """
    src_col = f"{attr}_source"
    trg_col = f"{attr}_target"

    # Initialize AUC matrix for edges
    auc_matrix = pd.DataFrame(np.nan, index=attr_values, columns=attr_values)
    edge_results = []

    # Analyze each source-target pair for EDGE retention
    for src_attr, trg_attr in itertools.product(attr_values, repeat=2):
        edge_df = s_table[
            (s_table[src_col] == src_attr) & (s_table[trg_col] == trg_attr)
        ]

        if edge_df.empty or len(edge_df) < min_edges_len:
            if len(edge_df) > 0 and len(edge_df) < min_edges_len:
                logger.debug(
                    f"Skipping pair ({src_attr} → {trg_attr}): only {len(edge_df)} edges (min: {min_edges_len})"
                )
            continue

        scores = np.array(edge_df["score"])

        # Calculate edge retention fractions
        edge_fractions = np.array([np.sum(scores >= t) for t in thresholds]) / len(
            edge_df
        )

        # Calculate AUC (area under curve)
        auc = np.trapezoid(edge_fractions[::-1], thresholds[::-1])
        auc_matrix.loc[src_attr, trg_attr] = auc

        edge_results.append(
            {
                "pair": (src_attr, trg_attr),
                "x": -np.log(1 - thresholds),  # Transform for better visualization
                "edges_fractions": edge_fractions,
                "auc": auc,
                "n_edges": len(edge_df),
            }
        )

    return edge_results, auc_matrix


def _calculate_node_retention_by_sweep(
    s_table, attr, attr_values, thresholds, min_edges_len
):
    """
    Calculate node retention curves for each attribute value using sweep-and-sort approach.

    Node retention: For each attribute value, counts nodes that HAVE that specific
    attribute value and remain connected to at least one other node after filtering
    edges below each threshold. Uses sorted edges and sweeps through thresholds.

    Parameters:
    -----------
    s_table : pd.DataFrame
        Table with backbone scores and enriched attributes
    attr : str
        Attribute name to analyze
    attr_values : list
        List of unique attribute values to analyze
    thresholds : np.ndarray
        Array of threshold values to evaluate
    min_edges_len : int
        Minimum number of edges required for a group to be analyzed

    Returns:
    --------
    dict : node_results
        Dict with node retention data for each single attribute value
    """
    src_col = f"{attr}_source"
    trg_col = f"{attr}_target"

    node_results = {}

    for attr_value in attr_values:
        # Subset of edges connected to this attribute value
        attr_edges = s_table[
            (s_table[src_col] == attr_value) | (s_table[trg_col] == attr_value)
        ]

        if attr_edges.empty or len(attr_edges) < min_edges_len:
            continue

        # Preload columns as numpy arrays for speed
        src_attr_arr = attr_edges[src_col].values
        trg_attr_arr = attr_edges[trg_col].values
        src_arr = attr_edges["src"].values
        trg_arr = attr_edges["trg"].values
        scores_arr = attr_edges["score"].values

        # Get unique nodes that HAVE this specific attribute value (vectorized)
        src_mask = src_attr_arr == attr_value
        trg_mask = trg_attr_arr == attr_value

        nodes_with_attr = np.unique(
            np.concatenate([src_arr[src_mask], trg_arr[trg_mask]])
        )
        max_nodes = len(nodes_with_attr)

        if max_nodes == 0:
            continue

        # Sort by score descending for efficient threshold processing
        # Get the indices that would sort the array and use them to sort all the arrays
        sort_idx = np.argsort(-scores_arr)
        scores_subset = scores_arr[sort_idx]
        src_attr_subset = src_attr_arr[sort_idx]
        src_subset = src_arr[sort_idx]
        trg_attr_subset = trg_attr_arr[sort_idx]
        trg_subset = trg_arr[sort_idx]

        # Calculate node retention at each threshold (vectorized)
        nodes_fraction = []
        for t in thresholds:
            # Binary search to find cutoff point
            # Cutoff is the index of the first element that is less than or equal to t
            # 'right' side means that if t is equal to an element,
            # the cutoff will be the index of the first element that is greater than t
            cutoff = np.searchsorted(-scores_subset, -t, side="right")

            if cutoff == 0:
                nodes_fraction.append(0)
                continue

            # Get retained nodes with this attribute value (vectorized)
            # [:cutoff] means that we only consider the elements up to the cutoff point
            # [src_attr_subset[:cutoff] == attr_value] means that we only consider
            # the elements that have this attribute value

            src_retained = src_subset[:cutoff][src_attr_subset[:cutoff] == attr_value]
            trg_retained = trg_subset[:cutoff][trg_attr_subset[:cutoff] == attr_value]

            retained_nodes = np.unique(np.concatenate([src_retained, trg_retained]))
            nodes_fraction.append(len(retained_nodes) / max_nodes)

        nodes_fraction = np.array(nodes_fraction)

        node_results[attr_value] = {
            "attribute": attr_value,
            "x": -np.log(1 - thresholds),  # Transform for better visualization
            "nodes_fraction": nodes_fraction,
            "n_total_nodes": max_nodes,
            "n_edges": len(attr_edges),
        }

    return node_results


def _calculate_node_retention_by_max_edge(
    s_table, attr, attr_values, thresholds, min_edges_len
):
    """
    Calculate node retention curves for each attribute value using max-edge approach.

    Node retention: For each attribute value, assigns to each node the highest score
    among all its incident edges (both incoming and outgoing). Then for each threshold,
    counts how many nodes with that attribute value have max_score >= threshold.

    Parameters:
    -----------
    s_table : pd.DataFrame
        Table with backbone scores and enriched attributes (must have 'src', 'trg', 'score')
    attr : str
        Attribute name to analyze
    attr_values : list
        List of unique attribute values to analyze
    thresholds : np.ndarray
        Array of threshold values to evaluate
    min_edges_len : int
        Minimum number of edges required for a group to be analyzed

    Returns:
    --------
    dict : node_results
        Dict with node retention data for each single attribute value
    """
    src_col = f"{attr}_source"
    trg_col = f"{attr}_target"

    node_results = {}

    # Get all nodes and compute their max edge scores (vectorized)
    # Create two DataFrames: one for source nodes, one for target nodes
    src_edges = s_table[["src", src_col, "score"]].rename(
        columns={"src": "node", src_col: "attr_value"}
    )
    trg_edges = s_table[["trg", trg_col, "score"]].rename(
        columns={"trg": "node", trg_col: "attr_value"}
    )

    # Combine all edges (node appears once per incident edge)
    all_node_edges = pd.concat([src_edges, trg_edges], ignore_index=True)

    # Group by node and find max score for each node
    node_max_scores = (
        all_node_edges.groupby("node")
        .agg(
            {
                "score": "max",
                "attr_value": "first",  # Node's attribute value (should be same for all incident edges)
            }
        )
        .reset_index()
    )

    # For each attribute value, calculate retention
    for attr_value in attr_values:
        # Get edges connected to this attribute value for min_edges_len check
        attr_edges = s_table[
            (s_table[src_col] == attr_value) | (s_table[trg_col] == attr_value)
        ]

        if attr_edges.empty or len(attr_edges) < min_edges_len:
            continue

        # Filter nodes that HAVE this specific attribute value
        nodes_with_attr = node_max_scores[node_max_scores["attr_value"] == attr_value]
        max_nodes = len(nodes_with_attr)

        if max_nodes == 0:
            continue

        # Get max scores for these nodes as numpy array for vectorized comparison
        max_scores = nodes_with_attr["score"].values

        # For each threshold, compute fraction of nodes retained (vectorized)
        nodes_fraction = np.array(
            [np.sum(max_scores >= t) / max_nodes for t in thresholds]
        )

        node_results[attr_value] = {
            "attribute": attr_value,
            "x": -np.log(1 - thresholds),  # Transform for better visualization
            "nodes_fraction": nodes_fraction,
            "n_total_nodes": max_nodes,
            "n_edges": len(attr_edges),
        }

    return node_results


def compute_retention_curves(
    s_table, attr, min_edges_len=500, resolution=50, logspace_lower_bound=-14
):
    """
    Compute retention curves for all source-target pairs of a given attribute.

    Edge retention: For each source-target attribute pair (e.g., Female→Male),
    computes the fraction of edges retained at each significance threshold.

    Node retention: For each attribute value (e.g., Female), counts nodes that
    HAVE that specific attribute value and remain connected to at least one other
    node after filtering edges below each threshold. Nodes are only counted if they
    possess the attribute value being analyzed.

    Parameters:
    -----------
    s_table : pd.DataFrame
        Table with backbone scores and enriched attributes
    attr : str
        Attribute name to analyze (e.g., 'gender', 'un_region')
    min_edges_len : int, default=500
        Minimum number of edges required for a group to be analyzed.
        Groups with fewer edges are excluded to ensure statistical reliability
        and avoid noise from small sample sizes.
    resolution : int, default=50
        Number of threshold points to evaluate along the retention curve.
        Higher values produce smoother curves but increase computation time.
        Lower values are faster but may miss fine-grained patterns.
    logspace_lower_bound : float, default=-14
        Lower bound for logspace threshold calculation (as an exponent of 10).
        Thresholds are computed as 1 - 10^logspace_lower_bound to 1 - 10^0.
        Values below -14 risk floating-point precision errors.

    Returns:
    --------
    tuple : (edge_results_list, node_results_dict, edge_auc_matrix)
        - edge_results_list: List of dicts with edge retention data for each pair
        - node_results_dict: Dict with node retention data for each single attribute value
        - edge_auc_matrix: DataFrame with AUC values for each source-target pair

    Notes:
    ------
    Thresholds are calculated using logarithmic spacing from 10^logspace_lower_bound to 1,
    providing fine resolution near threshold=1 where most filtering occurs.
    """
    src_col = f"{attr}_source"
    trg_col = f"{attr}_target"

    # Filter out rows where this specific attribute is NaN (for either source or target)
    # This allows entities with missing data for THIS attribute to be excluded,
    # while still being included in analysis for other attributes where they have data
    s_table = s_table.dropna(subset=[src_col, trg_col])

    if len(s_table) == 0:
        logger.warning(
            f"No data remaining after dropping NaN values for attribute '{attr}'"
        )
        return [], {}, pd.DataFrame()

    # Get unique attribute values (already filtered NaNs above)
    attr_values = sorted(
        set(s_table[src_col].astype(str)) | set(s_table[trg_col].astype(str))
    )

    # Create threshold array with logarithmic spacing near 1
    x = np.logspace(logspace_lower_bound, 0, resolution)
    thresholds = 1 - x

    # Calculate edge retention
    edge_results, auc_matrix = _calculate_edge_retention(
        s_table, attr, attr_values, thresholds, min_edges_len
    )

    # Calculate node retention using max-edge approach
    node_results = _calculate_node_retention_by_max_edge(
        s_table, attr, attr_values, thresholds, min_edges_len
    )

    return edge_results, node_results, auc_matrix


def plot_retention_curves(
    retention_results_by_lang, attribute, languages=None, min_edges=500
):
    """
    Plot retention curves for edges and nodes side by side.

    Parameters:
    -----------
    retention_results_by_lang : dict
        {lang: {attr: (edge_results_list, node_results_dict, edge_auc_matrix)}}
    attribute : str
        Attribute to plot
    languages : list, optional
        List of languages to plot. If None, plots all languages.
    min_edges : int, default=500
        Minimum edges to include a group in the plot.
        Groups with fewer edges are filtered out for clarity.
    """
    if languages is None:
        languages = list(retention_results_by_lang.keys())

    for lang in languages:
        if lang not in retention_results_by_lang:
            continue

        edge_results, node_results, _ = retention_results_by_lang[lang][attribute]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot edges retention
        for res in edge_results:
            if res["n_edges"] < min_edges:
                continue
            label = f"{res['pair'][0]}→{res['pair'][1]} (n={res['n_edges']})"
            axes[0].plot(res["x"], res["edges_fractions"], label=label, alpha=0.7)

        axes[0].set_title(f"Edges Retention ({attribute}, {lang})")
        axes[0].set_xlabel("-log(1-threshold)")
        axes[0].set_ylabel("Fraction retained")
        axes[0].legend(loc="lower left", fontsize="x-small")

        # Plot nodes retention
        for attr_value, node_res in node_results.items():
            if node_res["n_edges"] < min_edges:
                continue
            label = f"{attr_value} (nodes={node_res['n_total_nodes']}, edges={node_res['n_edges']})"
            axes[1].plot(
                node_res["x"], node_res["nodes_fraction"], label=label, alpha=0.7
            )

        axes[1].set_title(f"Nodes Retention ({attribute}, {lang})")
        axes[1].set_xlabel("-log(1-threshold)")
        axes[1].set_ylabel("Fraction retained")
        axes[1].legend(loc="lower left", fontsize="x-small")

        plt.tight_layout()
        plt.show()


def normalize_and_sort_retention_matrices(
    retention_results_by_lang, attributes=None, min_edges=0
):
    """
    Normalize AUC matrices to [0,1] and sort by diagonal values.

    Parameters:
    -----------
    retention_results_by_lang : dict
        {lang: {attr: (edge_results_list, node_results_dict, auc_matrix)}}
    attributes : list, optional
        List of attributes to process
    min_edges : int
        Mask cells with fewer edges than this threshold

    Returns:
    --------
    dict : {lang: {attr: normalized_sorted_matrix}}
    """

    def normalize_auc(matrix):
        """Normalize matrix to [0,1] range."""
        valid = np.isfinite(matrix.values)
        n_valid = np.sum(valid)

        if n_valid <= 1:
            return matrix

        min_val = np.nanmin(matrix.values)
        max_val = np.nanmax(matrix.values)

        if min_val == max_val:
            return matrix

        return (matrix - min_val) / (max_val - min_val)

    def sort_diagonal_ascending(matrix):
        """Sort matrix rows/columns by diagonal values."""
        diag_values = matrix.values.diagonal()
        order = np.argsort(diag_values)
        ordered_labels = matrix.index[order]
        return matrix.reindex(index=ordered_labels, columns=ordered_labels)

    if attributes is None:
        attributes = list(
            retention_results_by_lang[next(iter(retention_results_by_lang))].keys()
        )

    results = {}
    for lang, attr_dict in retention_results_by_lang.items():
        results[lang] = {}
        for attr in attributes:
            edge_results_list, node_results_dict, auc_matrix = attr_dict[attr]

            # Mask cells with insufficient edges
            for src in auc_matrix.index:
                for trg in auc_matrix.columns:
                    n_edges = next(
                        (
                            r["n_edges"]
                            for r in edge_results_list
                            if r["pair"] == (src, trg)
                        ),
                        0,
                    )
                    if n_edges < min_edges:
                        auc_matrix.loc[src, trg] = np.nan

            norm_matrix = normalize_auc(auc_matrix)
            sorted_matrix = sort_diagonal_ascending(norm_matrix)
            results[lang][attr] = sorted_matrix

    return results


def plot_heatmaps(retention_matrix, lang, attributes):
    """
    Plot heatmaps of normalized AUC values for multiple attributes.

    Parameters:
    -----------
    retention_matrix : dict
        {lang: {attr: normalized_matrix}}
    lang : str
        Language to plot
    attributes : list
        List of attributes to plot
    """
    n = len(attributes)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))

    cmap = plt.colormaps.get_cmap("RdYlGn").copy()
    cmap.set_bad("darkgray")

    if n == 1:
        axes = [axes]

    for i, (ax, attr) in enumerate(zip(axes, attributes)):
        matrix = retention_matrix[lang][attr]

        sns.heatmap(
            matrix,
            annot=True,
            cmap=cmap,
            fmt=".2f",
            mask=np.isnan(matrix),
            ax=ax,
            cbar=(i == n - 1),
            vmin=0,
            vmax=1,
        )

        ax.set_title(f"AUC by {attr.capitalize()}")
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")

    fig.suptitle(f"Retention Bias Heatmaps for Language: {lang}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def run_bias_analysis(
    edges_df,
    meta_df,
    edge_id_map={
        "source_wikidata_id": "src",
        "target_wikidata_id": "trg",
        "weight": "nij",
    },
    meta_id_col="wikidata_id",
    transformations=[
        ("gender", lambda x: x),  # Identity transformation
        ("bigperiod_birth", _transform_period),
        ("un_subregion", _transform_region),
    ],
    selected_languages=None,
    min_edges=500,
    resolution=50,
    filter_zero_weights=True,
    language_col="language_code",
    add_aggregated_all=True,
    pre_transform_filters=None,
    post_transform_filters=None,
    dataset_label="dataset",
):
    """
    Run complete bias analysis pipeline.

    Parameters:
    -----------
    edges_df : pd.DataFrame
        DataFrame containing edge data with columns to be mapped via edge_id_map
    meta_df : pd.DataFrame
        DataFrame containing node metadata
    edge_id_map : dict
        Column name mapping for edge dataframe
    meta_id_col : str
        ID column name in metadata (default: 'wikidata_id')
    transformations : list of tuples, default uses bigperiod_birth and un_subregion
        List of (column_name, transform_function) tuples for attribute transformation.
        Default transformations:
        - gender: Identity (no transformation)
        - bigperiod_birth: Simplified period names via _transform_period()
        - un_subregion: Western/Non-Western classification via _transform_region()

        If a metadata column is not listed in transformations, it is assumed to use
        identity transformation (values unchanged). To analyze different attributes,
        add them to this list with appropriate transformation functions.
    selected_languages : list, optional
        Languages to analyze (None = all)
    min_edges : int
        Minimum edges for group analysis
    resolution : int
        Number of threshold points
    filter_zero_weights : bool
        Remove zero-weight edges
    language_col : str
        Name of language column (if exists)
    add_aggregated_all : bool
        Add aggregated 'all' language with summed weights
    pre_transform_filters : list of functions, optional
        Filters applied to RAW metadata values (e.g., birth year, un_subregion).
        Each function takes a row (Series) and returns bool.
        Filters are applied sequentially (all must pass - AND logic).
    post_transform_filters : list of functions, optional
        Filters applied AFTER transformations (e.g., 'Western', 'Contemporary').
        Each function takes a row (Series) and returns bool.
        Filters are applied sequentially (all must pass - AND logic).
    dataset_label : str
        Label to identify this dataset in plot titles (e.g., 'full', 'restricted_1750_1950')

    Returns:
    --------
    dict : retention_results_by_lang
        {lang: {attr: (edge_results_list, node_results_dict, auc_matrix, dataset_label)}}
    """
    # Import noise_corrected here to avoid requiring it at module level
    try:
        import sys

        sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ". .")))
        from src.modules.backboning import noise_corrected
    except ImportError:
        raise ImportError(
            "Could not import noise_corrected from src.modules.backboning.  "
            "Please ensure the module is available in your Python path."
        )

    logger.info("Processing dataframes")
    # Create copies to avoid modifying the original dataframes
    edges_df = edges_df.copy().rename(columns=edge_id_map)
    meta_df = meta_df.copy()

    if filter_zero_weights:
        edges_df = edges_df[edges_df["nij"] > 0]

    logger.info(f"Loaded {len(edges_df)} edges with positive weight")

    # Add aggregated "all" language if requested
    if add_aggregated_all and language_col in edges_df.columns:
        logger.debug("Creating aggregated 'all' language")
        aggregated_edges = (
            edges_df.groupby(["src", "trg"]).agg({"nij": "sum"}).reset_index()
        )
        aggregated_edges[language_col] = "all"
        edges_df = pd.concat([edges_df, aggregated_edges], ignore_index=True)
        logger.debug(f"Added {len(aggregated_edges)} aggregated edges")

    # Enrich edges with metadata
    logger.debug("Enriching edges with metadata")
    selected_attributes = [col for col, _ in transformations]

    # Partition by language if column exists
    if language_col in edges_df.columns:
        if selected_languages is not None:
            # Add 'all' to selected languages if aggregation was performed
            if add_aggregated_all and "all" not in selected_languages:
                selected_languages = list(selected_languages) + ["all"]
            language_dfs = partition_edges_by_language(
                edges_df, language_col, selected_languages
            )
        else:
            language_dfs = partition_edges_by_language(edges_df, language_col)
    else:
        language_dfs = {"all": edges_df}

    logger.info(
        f"Analyzing {len(language_dfs)} language(s): {list(language_dfs.keys())}"
    )

    # Calculate retention curves for each language
    retention_results_by_lang = {}

    for lang, lang_df in language_dfs.items():
        logger.info(f"Processing language: {lang}")

        # Calculate backbone scores
        logger.debug("Computing backbone scores")
        backbone_df = noise_corrected(lang_df, calculate_p_value=True)

        # Preserve language_code column for filters that need it
        if language_col in lang_df.columns:
            backbone_df[language_col] = lang

        # Step 1: Merge metadata onto edges (no transformations yet)
        columns_to_merge = [col for col, _ in transformations]

        # Always include 'birth' for year-based filtering, even if not in transformations
        if "birth" not in columns_to_merge and "birth" in meta_df.columns:
            columns_to_merge = columns_to_merge + ["birth"]

        backbone_df = merge_node_metadata(
            backbone_df, meta_df, columns_to_merge, meta_id_col=meta_id_col
        )

        # Step 2: Apply pre-transform filters (on raw metadata values)
        if pre_transform_filters is not None:
            for i, filter_func in enumerate(pre_transform_filters, 1):
                initial_rows = len(backbone_df)
                mask = backbone_df.apply(filter_func, axis=1)
                backbone_df = backbone_df[mask]
                logger.debug(
                    f"Pre-transform filter {i}: {initial_rows} edges before, {len(backbone_df)} edges after"
                )

            logger.info(
                f"After pre-transform filters for {lang}: {len(backbone_df)} edges remaining"
            )
            if len(backbone_df) == 0:
                logger.warning(
                    f"No data left after pre-transform filtering for language: {lang}, skipping"
                )
                continue

        # Step 3: Apply transformations to metadata columns
        backbone_df = transform_node_attributes(backbone_df, transformations)

        # Step 4: Apply post-transform filters (on transformed values)
        if post_transform_filters is not None:
            for i, filter_func in enumerate(post_transform_filters, 1):
                initial_rows = len(backbone_df)
                mask = backbone_df.apply(filter_func, axis=1)
                backbone_df = backbone_df[mask]
                logger.debug(
                    f"Post-transform filter {i}: {initial_rows} edges before, {len(backbone_df)} edges after"
                )

            logger.info(
                f"After post-transform filters for {lang}: {len(backbone_df)} edges remaining"
            )
            if len(backbone_df) == 0:
                logger.warning(
                    f"No data left after post-transform filtering for language: {lang}, skipping"
                )
                continue

        # Compute retention curves for each attribute
        retention_results = {}
        for attr in selected_attributes:
            logger.debug(f"Computing retention curves for {attr}")
            edge_results, node_results, auc_matrix = compute_retention_curves(
                backbone_df, attr, min_edges_len=min_edges, resolution=resolution
            )
            retention_results[attr] = (
                edge_results,
                node_results,
                auc_matrix,
                dataset_label,
            )

        retention_results_by_lang[lang] = retention_results

    logger.info("Analysis complete - returning raw retention results")
    return retention_results_by_lang


def plot_retention_results(
    retention_results,
    languages=None,
    attributes=None,
    min_edges=500,
    plot_type="both",
    figsize=(14, 6),
    save_dir=None,
):
    """
    Plot retention curves and/or heatmaps from retention analysis results.

    Parameters:
    -----------
    retention_results : dict
        Results from run_bias_analysis()
        {lang: {attr: (edge_results, node_results, auc_matrix, dataset_label)}}
    languages : list, optional
        Languages to plot. If None, plots all languages.
    attributes : list, optional
        Attributes to plot. If None, plots all attributes.
    min_edges : int
        Minimum edges to include a group in plots
    plot_type : str
        Type of plot: 'curves', 'heatmaps', or 'both' (default: 'both')
    figsize : tuple
        Figure size for plots (default: (14, 6))
    save_dir : str, optional
        Directory to save plots. If None, plots are shown but not saved.

    Returns:
    --------
    None : Displays plots and optionally saves them
    """
    if languages is None:
        languages = list(retention_results.keys())

    if attributes is None:
        # Get attributes from first language
        first_lang = next(iter(retention_results))
        attributes = list(retention_results[first_lang].keys())

    for lang in languages:
        if lang not in retention_results:
            logger.warning(f"Language '{lang}' not found in results, skipping")
            continue

        for attr in attributes:
            if attr not in retention_results[lang]:
                logger.warning(
                    f"Attribute '{attr}' not found for language '{lang}', skipping"
                )
                continue

            # Extract dataset_label from results (backward compatibility: default to 'dataset' if not present)
            result_tuple = retention_results[lang][attr]
            if len(result_tuple) == 4:
                edge_results, node_results, auc_matrix, dataset_label = result_tuple
            else:
                edge_results, node_results, auc_matrix = result_tuple
                dataset_label = "dataset"

            # Plot retention curves
            if plot_type in ["curves", "both"]:
                _plot_retention_curves_single(
                    edge_results,
                    node_results,
                    lang,
                    attr,
                    min_edges,
                    figsize,
                    save_dir,
                    dataset_label,
                )

            # Plot heatmap
            if plot_type in ["heatmaps", "both"]:
                _plot_heatmap_single(
                    edge_results,
                    auc_matrix,
                    lang,
                    attr,
                    min_edges,
                    save_dir,
                    dataset_label,
                )


def _plot_retention_curves_single(
    edge_results,
    node_results,
    lang,
    attr,
    min_edges,
    figsize,
    save_dir,
    dataset_label="dataset",
):
    """Plot retention curves for edges and nodes."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Filter results by minimum edges
    filtered_edge_results = [r for r in edge_results if r["n_edges"] >= min_edges]

    if not filtered_edge_results:
        logger.warning(
            f"No edge pairs meet minimum threshold ({min_edges}) for {lang} - {attr}"
        )
        plt.close(fig)
        return

    # Plot edges retention
    for res in filtered_edge_results:
        label = f"{res['pair'][0]}→{res['pair'][1]} (n={res['n_edges']:,})"
        axes[0].plot(res["x"], res["edges_fractions"], label=label, alpha=0.7)

    title_suffix = f" - {dataset_label}" if dataset_label else ""
    axes[0].set_title(
        f"Edges Retention ({attr}, {lang}){title_suffix}", fontweight="bold"
    )
    axes[0].set_xlabel("-log(1-threshold)")
    axes[0].set_ylabel("Fraction retained")
    axes[0].legend(loc="lower left", fontsize="x-small")
    axes[0].grid(True, alpha=0.3)

    # Plot nodes retention
    for attr_value, node_res in node_results.items():
        if node_res["n_edges"] < min_edges:
            continue
        label = f"{attr_value} (nodes={node_res['n_total_nodes']:,}, edges={node_res['n_edges']:,})"
        axes[1].plot(node_res["x"], node_res["nodes_fraction"], label=label, alpha=0.7)

    title_suffix = f" - {dataset_label}" if dataset_label else ""
    axes[1].set_title(
        f"Nodes Retention ({attr}, {lang}){title_suffix}", fontweight="bold"
    )
    axes[1].set_xlabel("-log(1-threshold)")
    axes[1].set_ylabel("Fraction retained")
    axes[1].legend(loc="lower left", fontsize="x-small")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        # Create subdirectory for retention curves
        curves_dir = os.path.join(save_dir, f"retention_curves_{dataset_label}")
        os.makedirs(curves_dir, exist_ok=True)
        filename = os.path.join(curves_dir, f"retention_curves_{lang}_{attr}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot: {filename}")

    plt.show()


def _plot_heatmap_single(
    edge_results, auc_matrix, lang, attr, min_edges, save_dir, dataset_label="dataset"
):
    """Plot AUC heatmap for a single attribute."""
    # Filter matrix by minimum edges
    filtered_matrix = auc_matrix.copy()

    for src in auc_matrix.index:
        for trg in auc_matrix.columns:
            n_edges = next(
                (r["n_edges"] for r in edge_results if r["pair"] == (src, trg)), 0
            )
            if n_edges < min_edges:
                filtered_matrix.loc[src, trg] = np.nan

    # Check if there's any data to plot
    if filtered_matrix.isna().all().all():
        logger.warning(
            f"No data meets minimum threshold ({min_edges}) for heatmap: {lang} - {attr}"
        )
        return

    # Normalize to [0, 1]
    valid_values = filtered_matrix.values[np.isfinite(filtered_matrix.values)]
    if len(valid_values) > 1:
        min_val = np.nanmin(filtered_matrix.values)
        max_val = np.nanmax(filtered_matrix.values)
        if min_val != max_val:
            normalized = (filtered_matrix - min_val) / (max_val - min_val)
        else:
            normalized = filtered_matrix
    else:
        normalized = filtered_matrix

    # Sort rows and columns to have increasing diagonal
    diagonal_values = np.diag(normalized.values)
    if not np.all(np.isnan(diagonal_values)):
        # Create order based on diagonal values (NaN last)
        order = np.argsort(np.nan_to_num(diagonal_values, nan=np.inf))
        ordered_labels = normalized.index[order]
        normalized = normalized.loc[ordered_labels, ordered_labels]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.colormaps.get_cmap("RdYlGn").copy()
    cmap.set_bad("darkgray")

    sns.heatmap(
        normalized,
        annot=True,
        cmap=cmap,
        fmt=".2f",
        mask=np.isnan(normalized),
        ax=ax,
        cbar=True,
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
    )

    title_suffix = f" - {dataset_label}" if dataset_label else ""
    ax.set_title(
        f"Normalized AUC by {attr.capitalize()} ({lang}){title_suffix}",
        fontweight="bold",
    )
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")

    plt.tight_layout()

    if save_dir:
        # Create subdirectory for heatmaps
        heatmap_dir = os.path.join(save_dir, f"heatmaps_{dataset_label}")
        os.makedirs(heatmap_dir, exist_ok=True)
        filename = os.path.join(heatmap_dir, f"heatmap_{lang}_{attr}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot: {filename}")

    plt.show()


def build_analysis_output_dirs(
    base_output_dir,
    dataset_name,
    filter_description="",
    resolution=50,
    min_edges=500,
    selected_languages=None,
    attributes=None,
    logspace_lower_bound=-14,
):
    """
    Build directory structure for analysis outputs with comprehensive cache key.

    Creates a nested directory structure for storing analysis results.
    The directory name includes all parameters that affect the analysis results,
    ensuring cached results are only reused when parameters match.

    Directory structure:
    {base_output_dir}/{cache_key}/
        ├── auc/
        ├── retention_data/
        ├── retention_curves/
        └── heatmaps/

    Parameters:
    -----------
    base_output_dir : str
        Base directory for all analysis outputs (e.g., "data/out/plots")
    dataset_name : str
        Name of the dataset/edge file (e.g., "SpotlightWeightSource_0102_0505")
    filter_description : str, optional
        Description of filters applied (e.g., "restricted_1750_1950", "pageview_q0.8_lang")
    resolution : int, default=50
        Number of threshold points for retention curves
    min_edges : int, default=500
        Minimum edges required for statistical reliability
    selected_languages : list of str, optional
        Languages included in analysis (None means all)
    attributes : list of str, optional
        Attributes analyzed (e.g., ['gender', 'un_subregion'])
    logspace_lower_bound : float, default=-14
        Lower bound for threshold calculation

    Returns:
    --------
    dict : Dictionary with keys 'base', 'auc', 'retention_data', 'retention_curves', 'heatmaps'
           Each value is an absolute path to the corresponding directory

    Example:
    --------
    >>> dirs = build_analysis_output_dirs(
    ...     "data/out/plots",
    ...     "SpotlightWeightSource_0102_0505",
    ...     filter_description="restricted_1750_1950",
    ...     resolution=50,
    ...     min_edges=500,
    ...     selected_languages=['en', 'de', 'fr'],
    ...     attributes=['gender', 'un_subregion']
    ... )
    >>> print(dirs['base'])
    data/out/plots/SpotlightWeightSource_0102_0505_restricted_1750_1950_res50_min500_langs3_attrs2
    """
    # Generate comprehensive cache key including all parameters
    cache_key = cache_utils.generate_cache_key(
        dataset_name=dataset_name,
        filter_description=filter_description,
        resolution=resolution,
        min_edges=min_edges,
        selected_languages=selected_languages,
        attributes=attributes,
        logspace_lower_bound=logspace_lower_bound,
    )

    dataset_dir = os.path.join(base_output_dir, cache_key)

    dirs = {
        "base": dataset_dir,
        "auc": os.path.join(dataset_dir, "auc"),
        "retention_data": os.path.join(dataset_dir, "retention_data"),
        "retention_curves": os.path.join(dataset_dir, "retention_curves"),
        "heatmaps": os.path.join(dataset_dir, "heatmaps"),
    }

    return dirs


def save_analysis_results(
    retention_data,
    output_dirs,
    languages,
    attributes,
    resolution=50,
    min_edges=500,
    logspace_lower_bound=-14,
    dataset_name="",
    filter_description="",
):
    """
    Save analysis results to disk for caching.

    Saves three types of files per language-attribute combination:
    1. AUC matrices as CSV
    2. Edge retention curves as CSV
    3. Node retention data as CSV

    Also saves a metadata file with all analysis parameters for cache validation.

    Parameters:
    -----------
    retention_data : dict
        Results from run_bias_analysis()
        {lang: {attr: (edge_results, node_results, auc_matrix, dataset_label)}}
    output_dirs : dict
        Directory paths from build_analysis_output_dirs()
    languages : list
        List of language codes to save
    attributes : list
        List of attribute names to save
    resolution : int, optional
        Resolution parameter used in analysis
    min_edges : int, optional
        Min edges parameter used in analysis
    logspace_lower_bound : float, optional
        Logspace lower bound used in analysis
    dataset_name : str, optional
        Dataset name for metadata
    filter_description : str, optional
        Filter description for metadata

    Example:
    --------
    >>> results = run_bias_analysis(...)
    >>> dirs = build_analysis_output_dirs("data/out/plots", "dataset", "filter")
    >>> save_analysis_results(results, dirs, ["en", "de"], ["gender", "un_subregion"])
    Saved: data/out/plots/dataset_filter/auc/auc_en_gender.csv
    ...
    """
    # Create directories
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    for lang in retention_data:
        for attr in retention_data[lang]:
            # Handle both 3-tuple (old format) and 4-tuple (new format with dataset_label)
            result_tuple = retention_data[lang][attr]
            if len(result_tuple) == 4:
                edge_results, node_results, auc_matrix, dataset_label = result_tuple
            else:
                edge_results, node_results, auc_matrix = result_tuple

            # Save AUC matrix
            auc_file = os.path.join(output_dirs["auc"], f"auc_{lang}_{attr}.csv")
            auc_matrix.to_csv(auc_file)
            logger.info(f"Saved: {auc_file}")

            # Save edge retention curves
            if edge_results:
                curves_data = []
                for res in edge_results:
                    src, trg = res["pair"]
                    for x_val, y_val in zip(res["x"], res["edges_fractions"]):
                        curves_data.append(
                            {
                                "source": src,
                                "target": trg,
                                "threshold_x": x_val,
                                "retention_fraction": y_val,
                                "auc": res["auc"],
                                "n_edges": res["n_edges"],
                            }
                        )

                curves_df = pd.DataFrame(curves_data)
                curves_file = os.path.join(
                    output_dirs["retention_data"], f"retention_curves_{lang}_{attr}.csv"
                )
                curves_df.to_csv(curves_file, index=False)
                logger.info(f"Saved: {curves_file}")

            # Save node retention curves
            if node_results:
                node_data = []
                for attr_value, node_res in node_results.items():
                    for x_val, y_val in zip(node_res["x"], node_res["nodes_fraction"]):
                        node_data.append(
                            {
                                "attribute_value": attr_value,
                                "threshold_x": x_val,
                                "nodes_fraction": y_val,
                                "n_total_nodes": node_res["n_total_nodes"],
                                "n_edges": node_res["n_edges"],
                            }
                        )

                node_df = pd.DataFrame(node_data)
                node_file = os.path.join(
                    output_dirs["retention_data"], f"node_retention_{lang}_{attr}.csv"
                )
                node_df.to_csv(node_file, index=False)
                logger.info(f"Saved: {node_file}")

    # Save cache metadata for validation
    cache_utils.save_cache_metadata(
        cache_dir=output_dirs["base"],
        dataset_name=dataset_name,
        filter_description=filter_description,
        resolution=resolution,
        min_edges=min_edges,
        logspace_lower_bound=logspace_lower_bound,
        selected_languages=languages,
        attributes=attributes,
    )
    logger.info(f"Saved cache metadata: {output_dirs['base']}/cache_metadata.json")


def load_cached_analysis_results(output_dirs, languages, attributes):
    """
    Load cached analysis results from disk.

    Checks if all expected files exist and loads them if available.
    Returns None if cache is incomplete.

    Parameters:
    -----------
    output_dirs : dict
        Directory paths from build_analysis_output_dirs()
    languages : list
        List of language codes to load (should include 'all' if aggregated)
    attributes : list
        List of attribute names to load

    Returns:
    --------
    dict or None :
        If cache complete: {lang: {attr: (edge_results, node_results, auc_matrix, dataset_label)}}
        If cache incomplete: None

    Example:
    --------
    >>> dirs = build_analysis_output_dirs("data/out/plots", "dataset", "filter")
    >>> results = load_cached_analysis_results(dirs, ["en", "de", "all"], ["gender"])
    >>> if results is None:
    ...     results = run_bias_analysis(...)  # Run fresh analysis
    ... else:
    ...     print("Using cached results")
    """
    # Check if all expected files exist
    expected_files = []
    for lang in languages + ["all"] if "all" not in languages else languages:
        for attr in attributes:
            auc_file = os.path.join(output_dirs["auc"], f"auc_{lang}_{attr}.csv")
            edge_curves_file = os.path.join(
                output_dirs["retention_data"], f"retention_curves_{lang}_{attr}.csv"
            )
            node_curves_file = os.path.join(
                output_dirs["retention_data"], f"node_retention_{lang}_{attr}.csv"
            )
            expected_files.extend([auc_file, edge_curves_file, node_curves_file])

    # Check if all files exist
    all_exist = all(os.path.exists(f) for f in expected_files)

    if not all_exist:
        return None

    logger.info("Found cached results. Loading...")

    # Infer dataset_label from directory structure
    base_dir = output_dirs["base"]
    parts = os.path.basename(base_dir).split("_")
    # Extract everything after the first underscore as filter description
    if len(parts) > 1:
        # Find the dataset name part (should be SpotlightWeightSource_MMDD_HHMM or similar)
        # The filter description is everything after that
        dataset_label = "_".join(parts[3:]) if len(parts) > 3 else "full"
    else:
        dataset_label = "full"

    # Load the cached data
    cached_results = {}
    langs_to_load = languages + ["all"] if "all" not in languages else languages

    for lang in langs_to_load:
        cached_results[lang] = {}
        for attr in attributes:
            auc_file = os.path.join(output_dirs["auc"], f"auc_{lang}_{attr}.csv")
            edge_curves_file = os.path.join(
                output_dirs["retention_data"], f"retention_curves_{lang}_{attr}.csv"
            )
            node_curves_file = os.path.join(
                output_dirs["retention_data"], f"node_retention_{lang}_{attr}.csv"
            )

            # Load AUC matrix
            auc_matrix = pd.read_csv(auc_file, index_col=0)

            # Load and reconstruct edge retention curves
            edge_curves_df = pd.read_csv(edge_curves_file)
            edge_results = []
            if not edge_curves_df.empty:
                for (src, trg), group in edge_curves_df.groupby(["source", "target"]):
                    edge_results.append(
                        {
                            "pair": (src, trg),
                            "x": group["threshold_x"].values,
                            "edges_fractions": group["retention_fraction"].values,
                            "auc": group["auc"].iloc[0],
                            "n_edges": int(group["n_edges"].iloc[0]),
                        }
                    )

            # Load and reconstruct node retention curves
            node_curves_df = pd.read_csv(node_curves_file)
            node_results = {}
            if not node_curves_df.empty:
                for attr_value, group in node_curves_df.groupby("attribute_value"):
                    node_results[attr_value] = {
                        "attribute": attr_value,
                        "x": group["threshold_x"].values,
                        "nodes_fraction": group["nodes_fraction"].values,
                        "n_total_nodes": int(group["n_total_nodes"].iloc[0]),
                        "n_edges": int(group["n_edges"].iloc[0]),
                    }

            # Store as 4-tuple with dataset_label
            cached_results[lang][attr] = (
                edge_results,
                node_results,
                auc_matrix,
                dataset_label,
            )

    logger.info("Cached results loaded successfully")
    return cached_results
