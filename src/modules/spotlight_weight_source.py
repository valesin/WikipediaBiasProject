"""
This is an implementation of the WeightSource interface that provides weights for entities
based on their notable IDs.

The object wants to know:
- A set of notable IDs to consider.
- A set of languages to process.
- A mapping of language codes to DBpedia Spotlight endpoint URLs.
- A minimum confidence threshold for entity recognition.

"""

import re
import time
import logging
import gc
import os
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from SPARQLWrapper import SPARQLWrapper

logger = logging.getLogger(__name__)

if not PSUTIL_AVAILABLE:
    logger.debug("psutil not available. Memory monitoring will be disabled")

from modules.config import (
    SPOTLIGHT_ENDPOINTS,
    SPOTLIGHT_MIN_CONFIDENCE,
    SPOTLIGHT_TIMEOUT,
    SPOTLIGHT_MAX_RETRIES,
    SPOTLIGHT_RATE_LIMIT_DELAY,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    MEMORY_WARNING_THRESHOLD_GB,
)


def _default_if_none(value, default):
    """Return default if value is None, otherwise return value."""
    return default if value is None else value


class SpotlightWeightSource:
    def __init__(
        self,
        endpoints_urls: Dict[str, str] = None,
        min_confidence: float = None,
        batch_size: int = None,
        timeout: int = None,
        max_retries: int = None,
        checkpoint_dir: str = None,
        memory_warning_threshold_gb: float = None,
        rate_limit_delay: float = None,
        pool_connections: int = 10,
        pool_maxsize: int = 20,
    ) -> None:
        # Use config defaults if not explicitly provided
        self.endpoints_urls = _default_if_none(endpoints_urls, SPOTLIGHT_ENDPOINTS)
        self.min_confidence = _default_if_none(min_confidence, SPOTLIGHT_MIN_CONFIDENCE)
        self.batch_size = _default_if_none(batch_size, BATCH_SIZE)
        self.timeout = _default_if_none(timeout, SPOTLIGHT_TIMEOUT)
        self.max_retries = _default_if_none(max_retries, SPOTLIGHT_MAX_RETRIES)
        self.checkpoint_dir = _default_if_none(checkpoint_dir, CHECKPOINT_DIR)
        self.memory_warning_threshold_gb = _default_if_none(
            memory_warning_threshold_gb, MEMORY_WARNING_THRESHOLD_GB
        )
        self.rate_limit_delay = _default_if_none(
            rate_limit_delay, SPOTLIGHT_RATE_LIMIT_DELAY
        )
        self.pool_connections = pool_connections
        self.pool_maxsize = pool_maxsize

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Create session with connection pooling and retry logic
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with connection pooling and retry logic."""
        session = requests.Session()

        # Configure retry strategy with exponential backoff
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,  # 1, 2, 4 seconds
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.pool_connections,
            pool_maxsize=self.pool_maxsize,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB."""
        if not PSUTIL_AVAILABLE:
            return 0.0
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)

    def _check_memory_and_gc(self) -> None:
        """Check memory usage and run garbage collection if needed."""
        if not PSUTIL_AVAILABLE:
            return

        memory_gb = self._get_memory_usage_gb()
        if memory_gb > self.memory_warning_threshold_gb:
            logger.warning(
                f"Memory usage ({memory_gb:.2f} GB) exceeds threshold "
                f"({self.memory_warning_threshold_gb} GB). Running garbage collection..."
            )
            gc.collect()
            new_memory_gb = self._get_memory_usage_gb()
            logger.info(f"Memory after GC: {new_memory_gb:.2f} GB")

    def _save_checkpoint(self, checkpoint_file: str, weights_df: pd.DataFrame) -> None:
        """Save checkpoint to CSV file."""
        try:
            weights_df.to_csv(checkpoint_file, index=False)
            logger.info(f"Checkpoint saved to {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self, checkpoint_file: str) -> Optional[pd.DataFrame]:
        """Load checkpoint from CSV file."""
        if not os.path.exists(checkpoint_file):
            return None
        try:
            df = pd.read_csv(checkpoint_file)
            logger.info(f"Loaded checkpoint from {checkpoint_file} with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def __del__(self):
        """Close session on deletion."""
        if hasattr(self, "session"):
            self.session.close()

    def _combine_with_checkpoint(
        self, existing_weights_df: pd.DataFrame, new_weights: List[Dict]
    ) -> pd.DataFrame:
        """Helper method to combine new weights with existing checkpoint."""
        if not new_weights:
            return existing_weights_df

        new_df = pd.DataFrame(new_weights)
        if existing_weights_df.empty:
            return new_df
        return pd.concat([existing_weights_df, new_df], ignore_index=True)

    def assign_weights(
        self,
        entities_extracts_missing: pd.DataFrame,
        links_df: pd.DataFrame,
        title_to_id_map: Dict[str, Dict[str, str]],
        checkpoint_file: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Optimized version with batch processing and checkpointing.
        Process each extract once and immediately assign weights
        to all its outgoing links, instead of caching all mention counts.
        """
        logger.info(
            f"Starting link weight assignment for {len(links_df)} links in {links_df['language_code'].nunique()} languages..."
        )

        # Use default checkpoint file if none provided
        if checkpoint_file is None:
            checkpoint_file = os.path.join(
                self.checkpoint_dir,
                f"weights_checkpoint_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            )

        # Try to load existing checkpoint
        existing_weights_df = self._load_checkpoint(checkpoint_file)
        processed_keys = set()
        if existing_weights_df is not None:
            logger.info(
                f"Resuming from checkpoint with {len(existing_weights_df)} existing weights"
            )
            # Get already processed source entities (only if resuming)
            processed_keys = set(
                zip(
                    existing_weights_df["source_wikidata_id"],
                    existing_weights_df["language_code"],
                )
            )
        else:
            existing_weights_df = pd.DataFrame()

        # Build reverse map: {lang: {wikidata_id: title}}
        id_to_title_map = {
            lang: {wid: title for title, wid in title_map.items()}
            for lang, title_map in title_to_id_map.items()
        }

        # Group links by (source_wikidata_id, language_code)
        logger.info("Grouping links by source...")
        links_grouped = links_df.groupby(["source_wikidata_id", "language_code"])

        # Create a fast lookup: (source_id, lang) -> DataFrame of outgoing links
        source_links_map = {name: group for name, group in links_grouped}

        logger.info(
            f"Processing {len(source_links_map)} unique source entities in batches of {self.batch_size}..."
        )

        # Result list for current batch only
        batch_weights = []

        # Calculate actual sources to process (excluding already processed)
        sources_to_process = sum(
            1 for key in source_links_map.keys() if key not in processed_keys
        )

        # Process each extract once
        processed = 0
        batch_num = 0

        for idx, row in entities_extracts_missing.iterrows():
            wikidata_id = row["wikidata_id"]
            lang = row["language_code"]
            extract = row["extract"]

            # Check if this source has any outgoing links
            source_key = (wikidata_id, lang)
            if source_key not in source_links_map:
                continue  # Skip sources with no links

            # Skip if already processed in checkpoint
            if source_key in processed_keys:
                continue

            processed += 1

            # Progress logging
            if processed % 100 == 0:
                memory_gb = self._get_memory_usage_gb()
                logger.info(
                    f"Processed {processed}/{sources_to_process} sources with links... "
                    f"(Memory: {memory_gb:.2f} GB)"
                )

            # Get all outgoing links for this source
            outgoing_links = source_links_map[source_key]

            # Process extract and count mentions
            if not extract or not isinstance(extract, str):
                # No extract: assign weight 0 to all outgoing links
                for _, link_row in outgoing_links.iterrows():
                    batch_weights.append(
                        {
                            "source_wikidata_id": wikidata_id,
                            "target_wikidata_id": link_row["target_wikidata_id"],
                            "language_code": lang,
                            "weight": 0,
                        }
                    )
            else:
                # Count all mentions in this extract
                cleaned_extract = self._clean_text(extract)
                mention_counts = self._count_mentions_titles_in_extract(
                    cleaned_extract, lang
                )

                # Assign weights to all outgoing links from this source
                for _, link_row in outgoing_links.iterrows():
                    target_id = link_row["target_wikidata_id"]
                    target_title = id_to_title_map.get(lang, {}).get(target_id)

                    if target_title:
                        cleaned_target_title = self._clean_title(target_title)
                        weight = mention_counts.get(cleaned_target_title, 0)
                    else:
                        weight = 0

                    batch_weights.append(
                        {
                            "source_wikidata_id": wikidata_id,
                            "target_wikidata_id": target_id,
                            "language_code": lang,
                            "weight": weight,
                        }
                    )

                # Immediately discard mention_counts to free memory
                del mention_counts

            # Check if we've completed a batch
            if processed % self.batch_size == 0:
                batch_num += 1
                logger.info(
                    f"Completed batch {batch_num} ({self.batch_size} sources). "
                    f"Saving checkpoint with {len(batch_weights)} new weights..."
                )

                # Update existing checkpoint with new batch
                existing_weights_df = self._combine_with_checkpoint(
                    existing_weights_df, batch_weights
                )
                self._save_checkpoint(checkpoint_file, existing_weights_df)

                # Memory management
                self._check_memory_and_gc()

                # Clear batch to free memory
                batch_weights = []

        # Save final batch if any remaining
        if batch_weights:
            logger.info(
                f"Saving final checkpoint with {len(batch_weights)} remaining weights..."
            )
            existing_weights_df = self._combine_with_checkpoint(
                existing_weights_df, batch_weights
            )
            self._save_checkpoint(checkpoint_file, existing_weights_df)

        logger.info(
            f"Finished link weight assignment. Total links: {len(existing_weights_df)}"
        )
        return existing_weights_df

    def assign_weights_old(
        self,
        entities_extracts_missing: pd.DataFrame,
        links_df: pd.DataFrame,
        title_to_id_map: Dict[str, Dict[str, str]],
        batch_size: int = 100000,  # Process links in batches
    ) -> pd.DataFrame:
        """
        For each row in links_df, add a 'weight' column which is the count of how many times
        the target entity (by title) is mentioned in the extract of the source entity for the given language.
        """
        logger.info(
            f"Starting link weight assignment for {len(links_df)} links in {links_df['language_code'].nunique()} languages..."
        )

        # Build a reverse map: {lang: {wikidata_id: title}}
        id_to_title_map = {
            lang: {wid: title for title, wid in title_map.items()}
            for lang, title_map in title_to_id_map.items()
        }

        # Pre-compute mention counts for all unique (wikidata_id, language_code) pairs
        logger.info("Computing mention counts for all extracts...")
        mention_cache: Dict[Tuple[str, str], Dict[str, int]] = {}

        for idx, row in entities_extracts_missing.iterrows():
            if idx % 100 == 0:
                logger.info(
                    f"Processed {idx}/{len(entities_extracts_missing)} extracts..."
                )

            wikidata_id = row["wikidata_id"]
            lang = row["language_code"]
            extract = row["extract"]

            if not extract or not isinstance(extract, str):
                mention_cache[(wikidata_id, lang)] = {}
                continue

            cleaned_extract = self._clean_text(extract)
            mention_titles_dict = self._count_mentions_titles_in_extract(
                cleaned_extract, lang
            )
            mention_cache[(wikidata_id, lang)] = mention_titles_dict

        logger.info(f"Computed mentions for {len(mention_cache)} unique extracts.")

        # Process links in batches to avoid memory issues
        logger.info(f"Computing link weights in batches of {batch_size}...")
        weights = []

        for i in range(0, len(links_df), batch_size):
            batch_df = links_df.iloc[i : i + batch_size].copy()
            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(links_df)-1)//batch_size + 1} ({len(batch_df)} links)..."
            )

            # Vectorized approach for better performance
            batch_weights = []
            for _, row in batch_df.iterrows():
                source_id = row["source_wikidata_id"]
                target_id = row["target_wikidata_id"]
                lang = row["language_code"]

                # Look up pre-computed mentions
                mention_titles_dict = mention_cache.get((source_id, lang), {})
                if not mention_titles_dict:
                    batch_weights.append(0)
                    continue

                # Get the target's title for this language
                target_title = id_to_title_map.get(lang, {}).get(target_id)
                if not target_title:
                    batch_weights.append(0)
                    continue

                cleaned_target_title = self._clean_title(target_title)
                batch_weights.append(mention_titles_dict.get(cleaned_target_title, 0))

            batch_df["weight"] = batch_weights
            weights.append(batch_df)

            # Clear batch to free memory
            del batch_df, batch_weights

        result_df = pd.concat(weights, ignore_index=True)
        logger.info(f"Finished link weight assignment. Total links: {len(result_df)}")
        return result_df

    # def _count_mentions_in_extract(self, extract: str, lang: str) -> Dict[str, int]:
    #     """
    #     Annotate the extract using DBpedia Spotlight, count entity mentions,
    #     and map DBpedia URIs to Wikidata IDs.
    #     Returns: {wikidata_id: count}
    #     """

    #     if not extract.strip():
    #         return {}

    #     headers = {"accept": "application/json"}
    #     resources: List[Dict[str, Any]] = []
    #     try:
    #         response = requests.post(
    #             self.endpoints_urls.get(lang, ""),
    #             headers=headers,
    #             data={"text": extract, "confidence": self.min_confidence},
    #             timeout=5,
    #         )
    #         if response.status_code != 200:
    #             logger.debug(
    #                 f"Spotlight request failed with status code {response.status_code}."
    #             )
    #             logger.debug(response.text)
    #             return {}
    #         data = response.json()
    #         resources = data.get("Resources", [])
    #     except Exception as e:
    #         logger.debug("Exception occurred during Spotlight request.")
    #         logger.debug(str(e))

    #     uris = [r["@URI"] for r in resources]

    #     dbpedia_counts = dict(Counter(uris))
    #     logger.debug(f"Some DBpedia URIs counted: {list(dbpedia_counts.items())[:5]}")
    #     logger.debug(f"Annotated and found {len(dbpedia_counts)} unique DBpedia URIs.")
    #     if not dbpedia_counts:
    #         return {}

    #     dbpedia_endpoint = "https://dbpedia.org/sparql"
    #     uris_sparql = [f"<{uri}>" for uri in dbpedia_counts.keys()]
    #     values_clause = " ".join(uris_sparql)
    #     sparql_query = f"""
    #     SELECT ?dbpedia (SAMPLE(?wikidata) AS ?wikidata) WHERE {{
    #       VALUES ?dbpedia {{ {values_clause} }}
    #       ?dbpedia owl:sameAs ?wikidata .
    #       FILTER(STRSTARTS(STR(?wikidata), "http://www.wikidata.org/entity/"))
    #     }}
    #     GROUP BY ?dbpedia
    #     """
    #     logger.debug(
    #         f"Executing SPARQL query to map DBpedia URIs to Wikidata IDs: {sparql_query}"
    #     )
    #     sparql = SPARQLWrapper(dbpedia_endpoint)
    #     sparql.setQuery(sparql_query)
    #     sparql.setReturnFormat(JSON)
    #     sparql.setMethod(POST)

    #     results: Any = {}
    #     try:
    #         results = self._rate_limited_sparql_query(sparql)
    #     except Exception as e:
    #         logger.debug(f"SPARQL query failed: {e}")
    #         return {}

    #     dbpedia_to_wikidata: Dict[str, str] = {}

    #     for result in results["results"]["bindings"]:
    #         dbpedia_uri = result["dbpedia"]["value"]
    #         wikidata_uri = result["wikidata"]["value"]
    #         wikidata_id = wikidata_uri.split("/")[-1]
    #         dbpedia_to_wikidata[dbpedia_uri] = wikidata_id

    #     wikidata_counts: Dict[str, int] = {}
    #     for dbpedia_uri, count in dbpedia_counts.items():
    #         wikidata_id = dbpedia_to_wikidata.get(dbpedia_uri, None)
    #         if wikidata_id:
    #             wikidata_counts[wikidata_id] = count

    #     return wikidata_counts

    def _rate_limited_sparql_query(self, sparql: SPARQLWrapper) -> Any:
        time.sleep(0.01)
        return sparql.query().convert()

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\r?\n", " ", text)
        text = re.sub(r" +", " ", text)
        return text.strip()

    def _clean_title(self, title: str) -> str:
        new_title = title.replace("_", " ")
        new_title = str(requests.utils.unquote(new_title))  # type: ignore
        return new_title  # type : ignore

    def _count_mentions_titles_in_extract(
        self, extract: str, lang: str
    ) -> Dict[str, int]:
        if not extract.strip():
            return {}

        headers = {"accept": "application/json"}
        resources: List[Dict[str, Any]] = []

        # Retry logic with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting
                time.sleep(self.rate_limit_delay)

                response = self.session.post(
                    self.endpoints_urls.get(lang, ""),
                    headers=headers,
                    data={"text": extract, "confidence": self.min_confidence},
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    data = response.json()
                    resources = data.get("Resources", [])
                    break
                else:
                    logger.debug(
                        f"Spotlight request failed with status code {response.status_code}."
                    )
                    logger.debug(response.text)

                    # If not last attempt, wait with exponential backoff
                    if attempt < self.max_retries:
                        wait_time = 2**attempt
                        logger.debug(
                            f"Retrying in {wait_time} seconds... (attempt {attempt + 1}/{self.max_retries + 1})"
                        )
                        time.sleep(wait_time)
                    else:
                        return {}

            except requests.exceptions.Timeout:
                logger.debug(
                    f"Spotlight request timed out (attempt {attempt + 1}/{self.max_retries + 1})"
                )
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    logger.debug(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.warning(
                        f"Spotlight request timed out after {self.max_retries + 1} attempts"
                    )
                    return {}

            except Exception as e:
                logger.debug(
                    f"Exception occurred during Spotlight request (attempt {attempt + 1}/{self.max_retries + 1})"
                )
                logger.debug(str(e))
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    logger.debug(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.warning(
                        f"Spotlight request failed after {self.max_retries + 1} attempts: {str(e)}"
                    )
                    return {}

        uris = [r["@URI"] for r in resources]

        dbpedia_counts = dict(Counter(uris))
        logger.debug(f"Annotated and found {len(dbpedia_counts)} unique DBpedia URIs.")
        logger.debug(f"Some DBpedia URIs counted: {list(dbpedia_counts.items())[:5]}")
        if not dbpedia_counts:
            return {}

        # Preserve correct counts for normalized titles
        title_counts = {}
        for uri, count in dbpedia_counts.items():
            title = uri.split("/")[-1]
            cleaned_title = self._clean_title(title)
            title_counts[cleaned_title] = title_counts.get(cleaned_title, 0) + count

        if not title_counts:
            return {}

        return title_counts
