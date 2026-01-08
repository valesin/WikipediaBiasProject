""" "
This is the implementation of a DataSource that provides metadata for entities from Wikipedia/Wikidata.

It takes the wikidata ids and languages as input and returns the corresponding EntityMetadata objects.
The entity metadata includes text extracts/summaries and outgoing links to other entities.
The construction is as follows:
- Retrieve titles in all languages for the given wikidata ids, calling wikidata API.
- For each language, retrieve extracts and links using Wikipedia API.

The request are done transparently though the Requester module, which handles rate limiting and retries.
"""

import logging
import sys
import os
import json
import pandas as pd

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from modules.wiki_api_request_handler import WikiApiRequestHandler
    from modules.wiki_api_source import WikipediaAPIDataSource

    requester = WikiApiRequestHandler()
    datasource = WikipediaAPIDataSource(requester)

    qids = ["Q86010314", "Q42"]  # Example QIDs
    languages = ["en", "fr"]

    # Get titles for each QID
    all_titles = {}
    for qid in qids:
        titles = datasource.get_titles_for_qid(qid, languages)
        print(f"\nTitles for QID {qid}:")
        print(json.dumps(titles, indent=2, ensure_ascii=False))
        all_titles[qid] = titles

    # For each title found, fetch extracts and links
    for qid, lang_titles in all_titles.items():
        for lang, title in lang_titles.items():
            if title:
                extracts = datasource.get_extracts_for_title(title, [lang])
                links_qids = datasource.get_links_qids_for_title(title, [lang])
                print(f"\nExtract for ({lang}, {title}):")
                short_extract = (
                    extracts[lang][:80] + "..." if extracts[lang] else "None"
                )
                print(json.dumps({lang: short_extract}, indent=2, ensure_ascii=False))
                print(f"Links' QIDs for ({lang}, {title}):")
                print(
                    json.dumps({lang: links_qids[lang]}, indent=2, ensure_ascii=False)
                )


from typing import Any, Dict, Iterator, List, Optional

from modules.wiki_api_request_handler import WikiApiRequestHandler


class WikipediaAPIDataSource:
    """
    DataSource implementation that fetches entity metadata from Wikipedia/Wikidata APIs.
    """

    def __init__(
        self,
        requester: WikiApiRequestHandler,
    ) -> None:
        self.requester = requester

    # # def fetch_entity_meta_linksfrom(
    # #     self, entity_ids: List[str], languages: List[str], batch_size: int
    # # ) -> Optional[List[Dict[str, Any]]]:
    # #     logging.info("Fetching Wikidata titles...")
    # #     page_data_dict = self.requester.fetch_wikidata_titles(
    # #         entity_ids, languages, batch_size=batch_size
    # #     )
    # #     if not page_data_dict:
    # #         logging.info("No page data returned from Wikidata API.")
    # #         return None

    # #     raw_entities: list[Dict[str, Any]] = []
    # #     for entity_id in entity_ids:
    # #         for language in languages:
    # #             lang_data = page_data_dict.get(entity_id, {}).get(language)
    # #             if (
    # #                 lang_data
    # #                 and "title" in lang_data
    # #                 and lang_data["title"] is not None
    # #             ):
    # #                 raw_entities.append(
    # #                     {
    # #                         "id": entity_id,
    # #                         "language": language,
    # #                         "title": lang_data["title"],
    # #                         "description": lang_data.get("description"),
    # #                         "extract": None,
    # #                         "links": [],
    # #                     }
    # #                 )
    # #     logging.debug(f"Fetched {len(raw_entities)} raw entities with titles.")
    # #     return raw_entities

    # def fetch_metadata_for_entities(
    #     self, raw_entities: List[Dict[str, Any]]
    # ) -> Iterator[EntityMetadata]:
    #     logging.info(f"Fetching metadata for {len(raw_entities)} language-entities...")
    #     for raw in raw_entities:
    #         enriched_entity = self._fetch_metadata_for_entity(raw)
    #         if enriched_entity:
    #             yield EntityMetadata(**enriched_entity)

    # def _fetch_metadata_for_entity(
    #     self, entity: Dict[str, Any]
    # ) -> Optional[Dict[str, Any]]:
    #     logging.debug(f"Fetching page data: {entity['title']} ({entity['language']})")
    #     wiki_data = self.requester.fetch_page_details(
    #         entity["title"], entity["language"]
    #     )
    #     if wiki_data:
    #         entity["extract"] = wiki_data.get("extract")
    #         entity["links"] = wiki_data.get("links", [])
    #         entity["redirects"] = wiki_data.get("redirects", [])
    #         return entity
    #     return None

    def fetch_titles_for_qids(
        self, qid_list: List[str], languages: List[str]
    ) -> pd.DataFrame:
        """
        Given a DataFrame having QIDs,
        returns a DataFrame that adds Wikipedia titles and languages.
        """
        titles_data = []
        titles = self.requester.fetch_wikidata_sitelinks(qid_list, languages)
        # Titles is a dictionary mapping each Wikidata ID to a dictionary of language codes and their corresponding titles.
        for qid, lang_title_dict in titles.items():
            for lang, title in lang_title_dict.items():
                if title:
                    titles_data.append(
                        {"wikidata_id": qid, "page_title": title, "language_code": lang}
                    )
        titles_df = pd.DataFrame(titles_data)
        return titles_df

    def fetch_extract_for_title(self, title: str, language: str) -> str:
        """
        Given a Wikipedia title and a language, returns its extract.
        """
        extract = self.requester.fetch_page_extract(title, language)
        return extract

    def fetch_extract_for_titles_df(
        self, df: pd.DataFrame, log_frequency: int = 100
    ) -> pd.DataFrame:
        """
        Given a DataFrame with at least 'page_title' and 'language_code' columns,
        returns the same DataFrame with an added 'extract' column.

        Args:
            df: DataFrame with columns 'page_title' and 'language_code'
            log_frequency: Number of pages between progress logs (default: 100)
        """
        import logging

        logger = logging.getLogger(__name__)

        total_pages = len(df)
        extracts = []

        for idx, (_, row) in enumerate(df.iterrows(), 1):
            title = row["page_title"]
            language = row["language_code"]
            extract = self.requester.fetch_page_extract(title, language)
            extracts.append(extract)

            # Progress logging
            if idx % log_frequency == 0 or idx == total_pages:
                logger.info(f"Fetched extracts for {idx}/{total_pages} pages")

        logger.info(f"Completed extract fetching: {total_pages} pages processed")

        df = df.copy()
        df["extract"] = extracts
        return df

    def fetch_links_for_qid_and_titles_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a DataFrame with at least 'wikidata_id', 'page_title' and 'language_code' columns,
        returns another DataFrame with source_wikidata_id, target_wikidata_id, and language_code columns.
        """
        links_data = []
        for _, row in df.iterrows():
            source_qid = row["wikidata_id"]
            title = row["page_title"]
            language = row["language_code"]
            target_qids = self.requester.fetch_links_qids(title, language)
            for _, target_qid in target_qids:
                links_data.append(
                    {
                        "source_wikidata_id": source_qid,
                        "target_wikidata_id": target_qid,
                        "language_code": language,
                    }
                )
        links_df = pd.DataFrame(links_data)
        return links_df

    def fetch_links_for_qid_and_titles_df_with_commit(
        self, df: pd.DataFrame, cache_handler, log_frequency: int = 100
    ) -> None:
        """
        Given a DataFrame with at least 'wikidata_id', 'page_title' and 'language_code' columns,
        fetches links for each title and commits them atomically per title.

        This ensures that either ALL links for a given title are in the database, or none at all.
        No partial links for a title will be stored.

        Args:
            df: DataFrame with columns 'wikidata_id', 'page_title', 'language_code'
            cache_handler: Database handler to commit links
            log_frequency: Number of pages between progress logs (default: 100)
        """
        import logging

        logger = logging.getLogger(__name__)

        total_pages = len(df)
        total_links_committed = 0

        for idx, (_, row) in enumerate(df.iterrows(), 1):
            source_qid = row["wikidata_id"]
            title = row["page_title"]
            language = row["language_code"]

            # Fetch all links for this title
            target_qids = self.requester.fetch_links_qids(title, language)

            if not target_qids:
                logger.debug(f"No links found for '{title}' ({language})")
                continue

            # Build DataFrame for this title's links
            links_data = []
            for _, target_qid in target_qids:
                links_data.append(
                    {
                        "source_wikidata_id": source_qid,
                        "target_wikidata_id": target_qid,
                        "language_code": language,
                    }
                )

            links_df = pd.DataFrame(links_data)

            # Commit this title's links atomically
            cache_handler.insert_entity_links_from_df(links_df)
            total_links_committed += len(links_df)

            # Progress logging
            if idx % log_frequency == 0 or idx == total_pages:
                logger.info(
                    f"Fetched and committed links for {idx}/{total_pages} pages "
                    f"({total_links_committed} total links)"
                )

        logger.info(
            f"Completed link fetching: {total_pages} pages processed, "
            f"{total_links_committed} links committed"
        )
