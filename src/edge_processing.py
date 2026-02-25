import pandas as pd
import numpy as np
import regex
from collections import defaultdict
from tqdm import tqdm
from src.modules.duckdb_handler import DuckDBHandler
from src.filters import normalize_title, DEFAULT_STOPWORDS


def _group_edges(edges_df, tgt_titles_dict):
    """Group edge indices by (source_qid, lang).

    Returns
    -------
    dict[(qid, lang)] -> list[(edge_idx, raw_title)]
    """
    groups: dict[tuple, list] = defaultdict(list)
    src_col = edges_df["source_wikidata_id"].values
    tgt_col = edges_df["target_wikidata_id"].values
    lang_col = edges_df["language_code"].values

    for idx in range(len(edges_df)):
        src_key = (src_col[idx], lang_col[idx])
        tgt_key = (tgt_col[idx], lang_col[idx])
        tgt_title = tgt_titles_dict.get(tgt_key)
        groups[src_key].append((idx, tgt_title))

    return groups


def count_exact_matches_grouped(
    edges_df,
    src_extracts_dict: dict,
    tgt_titles_dict: dict,
) -> list:
    """Count **exact** substring matches, grouping by source for speed.

    Uses ``str.count()`` (~50 ns per check) on normalised, lowercased
    strings.  Extremely fast — expected to finish in under a minute for
    12 M+ edges.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Must have columns ``source_wikidata_id``, ``target_wikidata_id``,
        ``language_code``.
    src_extracts_dict : dict
        ``{(qid, lang): extract_text}``
    tgt_titles_dict : dict
        ``{(qid, lang): page_title}``

    Returns
    -------
    list[int]
        One count per row in *edges_df*, in the same order.
    """
    groups = _group_edges(edges_df, tgt_titles_dict)
    print(
        f"Grouped {len(edges_df)} edges into {len(groups)} source groups "
        f"(avg {len(edges_df) / len(groups):.1f} targets/source)."
    )

    # Precompute normalised titles
    _title_cache: dict[str, str | None] = {}

    def _norm_title(raw: str):
        if raw not in _title_cache:
            n = normalize_title(raw)
            _title_cache[raw] = n.lower() if n else None
        return _title_cache[raw]

    target_counts = [0] * len(edges_df)

    for src_key, targets in tqdm(groups.items(), desc="Source groups (exact)"):
        extract = src_extracts_dict.get(src_key)
        if not extract:
            continue
        extract_lower = extract.lower()

        for idx, tgt_title_raw in targets:
            if not tgt_title_raw:
                continue
            key = _norm_title(tgt_title_raw)
            if key is None:
                continue
            target_counts[idx] = extract_lower.count(key)

    return target_counts


def count_fuzzy_matches_grouped(
    edges_df,
    src_extracts_dict: dict,
    tgt_titles_dict: dict,
    max_errors: int = 2,
) -> list:
    """Count **fuzzy** substring matches, grouping by source for speed.

    Two-pass approach:
    1. **Exact** match via ``str.count()`` (~50 ns) — used when >= 1 hit.
    2. **Fuzzy** fallback via pre-compiled ``regex`` pattern (~3 ms) —
       only when exact match returns 0.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Must have columns ``source_wikidata_id``, ``target_wikidata_id``,
        ``language_code``.
    src_extracts_dict : dict
        ``{(qid, lang): extract_text}``
    tgt_titles_dict : dict
        ``{(qid, lang): page_title}``
    max_errors : int
        Maximum Levenshtein edit distance for the fuzzy fallback (default 2).

    Returns
    -------
    list[int]
        One count per row in *edges_df*, in the same order.
    """
    groups = _group_edges(edges_df, tgt_titles_dict)
    print(
        f"Grouped {len(edges_df)} edges into {len(groups)} source groups "
        f"(avg {len(edges_df) / len(groups):.1f} targets/source)."
    )

    # Pre-compile a regex for every unique normalised title
    print("Pre-compiling fuzzy patterns for unique titles...")
    _pattern_cache: dict[str, regex.Pattern] = {}
    _title_cache: dict[str, str | None] = {}

    def _get_title_and_pattern(raw_title: str):
        if raw_title not in _title_cache:
            norm = normalize_title(raw_title)
            if not norm:
                _title_cache[raw_title] = None
            else:
                key = norm.lower()
                _title_cache[raw_title] = key
                if key not in _pattern_cache:
                    pat_str = (
                        r"(?:" + regex.escape(key) + r"){e<=" + str(max_errors) + r"}"
                    )
                    _pattern_cache[key] = regex.compile(pat_str, flags=regex.IGNORECASE)
        return _title_cache[raw_title]

    # Warm the cache
    for raw_title in tgt_titles_dict.values():
        if raw_title:
            _get_title_and_pattern(raw_title)
    print(f"  Compiled {len(_pattern_cache)} unique patterns.")

    # Process
    counts = [0] * len(edges_df)
    n_exact = 0
    n_fuzzy = 0
    n_skip = 0

    for src_key, targets in tqdm(groups.items(), desc="Source groups (fuzzy)"):
        extract = src_extracts_dict.get(src_key)
        if not extract:
            n_skip += len(targets)
            continue
        norm_extract = extract
        if not norm_extract:
            n_skip += len(targets)
            continue
        extract_lower = norm_extract.lower()

        for idx, tgt_title_raw in targets:
            if not tgt_title_raw:
                n_skip += 1
                continue
            key = _get_title_and_pattern(tgt_title_raw)
            if key is None:
                n_skip += 1
                continue

            # Fast path: exact substring match
            exact = extract_lower.count(key)
            if exact > 0:
                counts[idx] = exact
                n_exact += 1
            else:
                # Slow path: fuzzy regex
                counts[idx] = len(_pattern_cache[key].findall(norm_extract))
                n_fuzzy += 1

    print(
        f"  Exact hits: {n_exact:,} | Fuzzy fallbacks: {n_fuzzy:,} | "
        f"Skipped: {n_skip:,}"
    )
    return counts


def read_edges_csv_and_filter_spurious(
    csv_path: str,
    db_path: str = "/home/vlr/Workspaces/WikipediaBiasProject/PyProject/data/out/graph_final.db",
    min_edges_for_stats: int = 5,
    custom_stopwords: set = None,
):
    """
    Reads an edges CSV and filters out spurious edges (pseudo-self-loops).
    If a 'fullmatch_count' column is present, it identifies suspect edges by:
        1. Partially matching titles (ignoring stopwords and short words)

    For suspect edges, the 'weight' is replaced with max(1, fullmatch_count).
    Returns the DataFrame with an added boolean 'is_suspect' column and a 'z_score' column (z-score per source node).
    The z-score threshold is not used for filtering, only for reference.
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Check for fullmatch_count
    if "fullmatch_count" not in df.columns:
        print("No 'fullmatch_count' column found. Returning DataFrame as is.")
        return df

    # Prepare for title fetching
    unique_qids = pd.unique(
        df[["source_wikidata_id", "target_wikidata_id"]].values.ravel("K")
    )
    unique_qids = [str(qid) for qid in unique_qids if pd.notna(qid)]
    unique_langs = df["language_code"].dropna().unique().tolist()

    print(
        f"Retrieving titles for {len(unique_qids):,} unique QIDs across {len(unique_langs)} languages..."
    )

    db = DuckDBHandler(db_path)

    # Batch title retrieval
    batch_size = 10000
    all_titles = []
    for i in range(0, len(unique_qids), batch_size):
        batch_qids = unique_qids[i : i + batch_size]
        batch_titles_df = db.get_titles_for_qids(qids=batch_qids, langs=unique_langs)
        all_titles.append(batch_titles_df)

    titles_df = (
        pd.concat(all_titles, ignore_index=True)
        if all_titles
        else pd.DataFrame(columns=["wikidata_id", "language_code", "page_title"])
    )
    db.close()

    # Temporarily rename columns to match what existing filters expect ('source', 'target', 'nij')
    df = df.rename(
        columns={
            "source_wikidata_id": "source",
            "target_wikidata_id": "target",
            "weight": "nij",
        }
    )

    # Merge source titles
    df = df.merge(
        titles_df.rename(columns={"wikidata_id": "source", "page_title": "src_title"}),
        on=["source", "language_code"],
        how="left",
    )

    # Merge target titles
    df = df.merge(
        titles_df.rename(columns={"wikidata_id": "target", "page_title": "trg_title"}),
        on=["target", "language_code"],
        how="left",
    )

    print("Computing shared names filter...")
    stopwords = custom_stopwords if custom_stopwords is not None else DEFAULT_STOPWORDS

    def filter_shared_names_strict(row):
        src_title = row.get("src_title")
        trg_title = row.get("trg_title")
        src_id = row.get("source")
        trg_id = row.get("target")
        # Exclude self-loops
        if src_id == trg_id:
            return True
        if pd.isna(src_title) or pd.isna(trg_title):
            return True

        src_norm = normalize_title(str(src_title))
        trg_norm = normalize_title(str(trg_title))
        src_words = set(
            w for w in src_norm.lower().split() if w not in stopwords and len(w) > 3
        )
        trg_words = set(
            w for w in trg_norm.lower().split() if w not in stopwords and len(w) > 3
        )
        shared_words = src_words & trg_words
        return len(shared_words) == 0

    # Add has_shared_word column
    def has_shared_word_func(row):
        src_title = row.get("src_title")
        trg_title = row.get("trg_title")
        if pd.isna(src_title) or pd.isna(trg_title):
            return False
        src_norm = normalize_title(str(src_title))
        trg_norm = normalize_title(str(trg_title))
        src_words = [
            w for w in src_norm.lower().split() if w not in stopwords and len(w) > 3
        ]
        trg_words = [
            w for w in trg_norm.lower().split() if w not in stopwords and len(w) > 3
        ]
        return any(w in src_words for w in trg_words)

    df["has_shared_word"] = df.apply(has_shared_word_func, axis=1)

    # Add ordered_substring column
    def ordered_substring_func(row):
        src_title = row.get("src_title")
        trg_title = row.get("trg_title")
        if pd.isna(src_title) or pd.isna(trg_title):
            return False
        src_norm = normalize_title(str(src_title)).lower()
        trg_norm = normalize_title(str(trg_title)).lower()
        src_words = [w for w in src_norm.split() if w not in stopwords and len(w) > 3]
        trg_words = [w for w in trg_norm.split() if w not in stopwords and len(w) > 3]
        if not trg_words:
            return False
        idx = 0
        for word in src_words:
            if word == trg_words[idx]:
                idx += 1
                if idx == len(trg_words):
                    return True
        return False

    df["ordered_substring"] = df.apply(ordered_substring_func, axis=1)

    # Group by source node and calculate mean/std
    node_stats = {}
    for source_node, group in df.groupby("source"):
        weights = group["nij"].values
        if len(weights) >= min_edges_for_stats:
            node_stats[source_node] = {
                "mean": np.mean(weights),
                "std": np.std(weights),
            }
    global_mean = df["nij"].mean()
    global_std = df["nij"].std()

    def compute_z_score(row):
        source = row["source"]
        weight = row["nij"]
        try:
            weight = float(weight)
        except (ValueError, TypeError):
            return np.nan
        stats = node_stats.get(source)
        if stats and stats["std"] != 0:
            return (weight - stats["mean"]) / stats["std"]
        elif global_std != 0:
            return (weight - global_mean) / global_std
        else:
            return np.nan

    df["z_score"] = df.apply(compute_z_score, axis=1)

    # Add substring column
    def is_wordwise_substring(src, trg):
        if pd.isna(src) or pd.isna(trg):
            return False
        src_words = set(str(src).split())
        trg_words = set(str(trg).split())
        return any(word in src_words for word in trg_words)

    df["substring"] = df.apply(
        lambda row: is_wordwise_substring(row.get("src_title"), row.get("trg_title")),
        axis=1,
    )

    # Clean up and revert column names
    cols_to_drop = [
        "src_title",
        "trg_title",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = df.rename(
        columns={
            "source": "source_wikidata_id",
            "target": "target_wikidata_id",
            "nij": "weight",
        }
    )

    return df
