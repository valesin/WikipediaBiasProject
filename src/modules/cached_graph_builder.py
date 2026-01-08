"""
The DBBuilderTool uses a DataSource to fetch entity metadata and a DBHandler to interact with the database.
The DataSource interface defines methods to get entity metadata and links.
In the main the following steps are performed:
1. Fill the database using the DBBuilderTool. After this step, we assume that for all notable IDs and languages, the database contains the entity metadata.
2. Assign weights using the WeightAssignerTool. This tool reads the entity metadata from the database, computes weights, and updates the database with these weights.
3. The database is now ready for use.
"""

# PROBLEMS:
# It uses the full database and doesn't limit to notable IDs only.
# I just need to add a WHERE clause in the SQL queries in DBHandler implementations.

import logging

logger = logging.getLogger(__name__)

# import csv
# import datetime
import time
from typing import List, Tuple, Any

from modules.spotlight_weight_source import SpotlightWeightSource
from modules.wiki_api_request_handler import WikiApiRequestHandler
from modules.wiki_api_source import WikipediaAPIDataSource
from modules.duckdb_handler import DuckDBHandler
from modules.config import (
    WIKIPEDIA_USER_AGENT,
    WIKIPEDIA_ACCESS_TOKEN,
)


class UnweightedGraphBuilder:
    def __init__(
        self,
        notable_ids_list: List[str],
        languages: List[str],
        cache_handler: DuckDBHandler,
        data_source: WikipediaAPIDataSource,
    ):
        self.notable_ids_list = notable_ids_list
        self.languages = languages
        self.cache_handler = cache_handler
        self.data_source = data_source

    def build_unweighted_graph(
        self,
        fetch_extracts: bool = True,
        fetch_links: bool = True,
    ) -> None:
        """
        Build a unweighted graph of Wikipedia entities.
        For each notable ID and language, fetch titles, extracts, and links if not present in the cache.
        Args:
            fetch_extracts (bool): Whether to fetch extracts.
            fetch_links (bool): Whether to fetch links.
        """
        logger.info(
            "Starting unweighted graph build for %d entities in %d languages",
            len(self.notable_ids_list),
            len(self.languages),
        )

        # Load all notable IDs
        logger.debug("Loading notable entity IDs into the cache")
        self.cache_handler.load_notable_entity_ids(self.notable_ids_list)

        # Retrieve missing titles
        logger.debug("Retrieving missing titles for notable entities")
        missing_titles_list = self.cache_handler.get_entities_missing_titles(
            self.notable_ids_list, self.languages
        )

        if missing_titles_list:
            logger.info(f"Fetching {len(missing_titles_list)} missing titles")
            title_df = self.data_source.fetch_titles_for_qids(
                missing_titles_list, self.languages
            )
            if not title_df.empty:
                logger.debug(f"Inserting {len(title_df)} titles into cache")
                self.cache_handler.insert_entity_titles_from_df(title_df)
        else:
            logger.debug("No missing titles to fetch")

        if fetch_extracts:
            missing_extracts_df = self.cache_handler.get_df_missing_extracts_by_lang(
                self.notable_ids_list, self.languages
            )
            if not missing_extracts_df.empty:
                logger.info(f"Fetching {len(missing_extracts_df)} missing extracts")

                # Process extracts in batches to commit more frequently
                BATCH_SIZE = 100
                total_batches = (
                    len(missing_extracts_df) + BATCH_SIZE - 1
                ) // BATCH_SIZE

                for i in range(0, len(missing_extracts_df), BATCH_SIZE):
                    batch_df = missing_extracts_df.iloc[i : i + BATCH_SIZE]
                    batch_num = i // BATCH_SIZE + 1

                    logger.debug(
                        f"Fetching extract batch {batch_num}/{total_batches} ({len(batch_df)} items)"
                    )
                    extracts_df = self.data_source.fetch_extract_for_titles_df(batch_df)

                    logger.debug(
                        f"Committing {len(extracts_df)} extracts (batch {batch_num}/{total_batches})"
                    )
                    self.cache_handler.insert_entity_extracts_from_df(extracts_df)

                logger.info(
                    f"Completed fetching and committing {len(missing_extracts_df)} extracts in {total_batches} batches"
                )
            else:
                logger.debug("No missing extracts to fetch")

        if fetch_links:
            missing_links_df = self.cache_handler.get_df_missing_links_by_lang(
                self.notable_ids_list, self.languages
            )

            if not missing_links_df.empty:
                logger.info(f"Fetching links for {len(missing_links_df)} pages")
                # Fetch and commit links atomically per title
                self.data_source.fetch_links_for_qid_and_titles_df_with_commit(
                    missing_links_df, self.cache_handler
                )
            else:
                logger.debug("No missing links to fetch")


class GraphWeighter:
    def __init__(
        self,
        languages: List[str],
        cache_handler: DuckDBHandler,
        weight_assigner: SpotlightWeightSource,
        checkpoint_file: str = None,
        output_dir: str = "data",
    ):
        self.languages = languages
        self.cache_handler = cache_handler
        self.weight_assigner = weight_assigner
        self.checkpoint_file = checkpoint_file
        self.output_dir = output_dir

    def weight_graph(self) -> None:
        """
        Build a weighted graph of Wikipedia entities and export the list of links with weights to a CSV file.
        Supports resumable processing with checkpoints.

        Args:
            None
        """
        logger.info("Starting weight assignment for %d languages", len(self.languages))
        # Get all entities in the cache for the given languages
        missing_weights_df = (
            self.cache_handler.get_entities_and_extracts_missing_weights(
                list(self.languages),
                method=SpotlightWeightSource.__name__,
            )
        )

        if not missing_weights_df.empty:
            logger.info(f"Found {len(missing_weights_df)} entities missing weights")
            # Retrieve a dataframe containing all links for these entities
            links_df = self.cache_handler.get_links_for_entities(
                missing_weights_df["wikidata_id"].tolist(),
                list(self.languages),
                notable_only=True,  # Filter to notable links only for weight assignment
            )
            logger.debug(f"Processing {len(links_df)} links for weight assignment")

            logger.info("Assigning weights using DBpedia Spotlight")
            weights = self.weight_assigner.assign_weights(
                entities_extracts_missing=missing_weights_df,
                links_df=links_df,
                title_to_id_map=self.cache_handler.get_title_to_id_map(
                    list(self.languages)
                ),
                checkpoint_file=self.checkpoint_file,
            )
            logger.info(f"Assigned weights for {len(weights)} links")
            logger.debug("Inserting weights into cache")
            self.cache_handler.insert_link_weights(
                weights,
                method=SpotlightWeightSource.__name__,
            )
        else:
            logger.info("No missing weights to assign")

        # Always export the CSV file, whether we assigned new weights or not
        output_file = f"{self.output_dir}/{SpotlightWeightSource.__name__}_{time.strftime('%m%d_%H%M')}.csv"
        logger.info(f"Exporting weighted links to {output_file}")
        self.cache_handler.save_weighted_links_to_csv(
            SpotlightWeightSource.__name__,
            output_file,
        )


def create_handlers(
    cache_filename: str,
) -> Tuple[DuckDBHandler, WikipediaAPIDataSource, SpotlightWeightSource]:
    """
    Create and configure the database handler, data source, and weight assigner.

    Args:
        cache_filename: Path to the DuckDB database file

    Returns:
        Tuple of (cache_handler, data_source, weight_assigner)
    """
    cache_handler = DuckDBHandler(db_filename=cache_filename)

    data_source = WikipediaAPIDataSource(
        requester=WikiApiRequestHandler(
            user_agent=WIKIPEDIA_USER_AGENT,
            access_token=WIKIPEDIA_ACCESS_TOKEN,
        ),
    )

    weight_assigner = SpotlightWeightSource()

    return cache_handler, data_source, weight_assigner
