"""
Cache key generation utilities for bias analysis results.

This module provides utilities to generate consistent cache keys based on analysis
parameters, ensuring that cached results are only reused when all relevant parameters match.
"""

import hashlib
import json
import os
from typing import List, Optional, Dict, Any


def generate_cache_key(
    dataset_name: str,
    filter_description: str = "",
    resolution: int = 50,
    min_edges: int = 500,
    selected_languages: Optional[List[str]] = None,
    attributes: Optional[List[str]] = None,
    logspace_lower_bound: float = -14,
) -> str:
    """
    Generate a cache key string from analysis parameters.

    Creates a verbose, human-readable cache key that includes all parameters
    affecting the analysis results. This ensures cache hits only occur when
    all parameters match.

    Parameters:
    -----------
    dataset_name : str
        Base dataset name (e.g., "SpotlightWeightSource_0102_0505")
    filter_description : str, optional
        Description of applied filters (e.g., "restricted_1750_1950", "pageview_q80")
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
    str : Cache key string

    Example:
    --------
    >>> generate_cache_key(
    ...     "SpotlightWeightSource_0102_0505",
    ...     "restricted_1750_1950",
    ...     resolution=50,
    ...     min_edges=500,
    ...     selected_languages=['en', 'de', 'fr'],
    ...     attributes=['gender', 'un_subregion']
    ... )
    'SpotlightWeightSource_0102_0505_restricted_1750_1950_res50_min500_langs3_attrs2'
    """
    parts = [dataset_name]

    # Add filter description if provided
    if filter_description:
        parts.append(filter_description)

    # Add resolution
    parts.append(f"res{resolution}")

    # Add min_edges
    parts.append(f"min{min_edges}")

    # Add language count (more compact than listing all languages)
    if selected_languages is not None:
        parts.append(f"langs{len(selected_languages)}")
    else:
        parts.append("langsAll")

    # Add attribute count
    if attributes is not None:
        parts.append(f"attrs{len(attributes)}")

    # Add logspace_lower_bound if not default
    if logspace_lower_bound != -14:
        # Format as integer to avoid dots in directory names
        parts.append(f"logsp{int(abs(logspace_lower_bound))}")

    return "_".join(parts)


def save_cache_metadata(
    cache_dir: str,
    dataset_name: str,
    filter_description: str = "",
    resolution: int = 50,
    min_edges: int = 500,
    logspace_lower_bound: float = -14,
    selected_languages: Optional[List[str]] = None,
    attributes: Optional[List[str]] = None,
    transformations: Optional[List[tuple]] = None,
    **extra_params,
) -> None:
    """
    Save analysis parameters to a metadata file in the cache directory.

    This allows verification and debugging of what parameters were used
    to generate cached results.

    Parameters:
    -----------
    cache_dir : str
        Path to cache directory
    dataset_name : str
        Dataset name
    filter_description : str
        Filter description
    resolution : int
        Resolution parameter
    min_edges : int
        Minimum edges parameter
    logspace_lower_bound : float
        Logspace lower bound parameter
    selected_languages : list of str, optional
        Selected languages
    attributes : list of str, optional
        Analyzed attributes
    transformations : list of tuples, optional
        Attribute transformations applied
    **extra_params : dict
        Any additional parameters to store
    """
    metadata = {
        "dataset_name": dataset_name,
        "filter_description": filter_description,
        "resolution": resolution,
        "min_edges": min_edges,
        "logspace_lower_bound": logspace_lower_bound,
        "selected_languages": selected_languages,
        "attributes": attributes,
        "transformations_count": len(transformations) if transformations else None,
    }

    # Add any extra parameters
    metadata.update(extra_params)

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Save metadata
    metadata_file = os.path.join(cache_dir, "cache_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


def load_cache_metadata(cache_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load cache metadata from directory.

    Parameters:
    -----------
    cache_dir : str
        Path to cache directory

    Returns:
    --------
    dict or None : Metadata dictionary, or None if file doesn't exist
    """
    metadata_file = os.path.join(cache_dir, "cache_metadata.json")

    if not os.path.exists(metadata_file):
        return None

    with open(metadata_file, "r") as f:
        return json.load(f)


def validate_cache_parameters(
    cache_dir: str,
    resolution: int = 50,
    min_edges: int = 500,
    logspace_lower_bound: float = -14,
    selected_languages: Optional[List[str]] = None,
    attributes: Optional[List[str]] = None,
) -> bool:
    """
    Validate that cached results match the requested parameters.

    Parameters:
    -----------
    cache_dir : str
        Path to cache directory
    resolution : int
        Expected resolution
    min_edges : int
        Expected min_edges
    logspace_lower_bound : float
        Expected logspace_lower_bound
    selected_languages : list of str, optional
        Expected languages
    attributes : list of str, optional
        Expected attributes

    Returns:
    --------
    bool : True if parameters match, False otherwise
    """
    metadata = load_cache_metadata(cache_dir)

    if metadata is None:
        return False

    # Check each parameter
    if metadata.get("resolution") != resolution:
        return False

    if metadata.get("min_edges") != min_edges:
        return False

    if metadata.get("logspace_lower_bound") != logspace_lower_bound:
        return False

    # For languages and attributes, check if the sets match
    if selected_languages is not None:
        cached_langs = metadata.get("selected_languages")
        if cached_langs is None or set(cached_langs) != set(selected_languages):
            return False

    if attributes is not None:
        cached_attrs = metadata.get("attributes")
        if cached_attrs is None or set(cached_attrs) != set(attributes):
            return False

    return True
