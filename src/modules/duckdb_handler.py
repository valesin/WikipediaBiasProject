import logging
import os
from typing import Dict, List, Optional, Set

import duckdb
import pandas as pd
from duckdb import DuckDBPyConnection

from .config import (
    DUCKDB_THREADS,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_MAX_TEMP_DIR_SIZE,
    DUCKDB_TEMP_DIR,
)

logger = logging.getLogger(__name__)


def _qid_to_int(qid: str) -> Optional[int]:
    if qid.startswith("Q"):
        try:
            result = int(qid[1:])
            return result
        except ValueError:
            return None
    elif isinstance(qid, int):
        return qid
    return None


def _int_to_qid(qid_int: Optional[int]) -> str:
    if qid_int is None:
        return ""
    return f"Q{qid_int}"


class DuckDBHandler:
    def get_titles_for_qids(self, qids: List[str], langs: List[str]) -> pd.DataFrame:
        """
        Returns a DataFrame with columns: wikidata_id, language_code, page_title
        for all (qid, lang) pairs present in the database.
        """
        qid_ints = [_qid_to_int(qid) for qid in qids if _qid_to_int(qid) is not None]
        if not qid_ints or not langs:
            return pd.DataFrame(columns=["wikidata_id", "language_code", "page_title"])

        placeholders_qids = ", ".join(["?"] * len(qid_ints))
        placeholders_langs = ", ".join(["?"] * len(langs))
        query = f"""
            SELECT wikidata_id, language_code, page_title
            FROM wiki_page
            WHERE wikidata_id IN ({placeholders_qids})
              AND language_code IN ({placeholders_langs})
        """
        params = qid_ints + langs
        df = self._conn.execute(query, params).fetchdf()
        if not df.empty:
            df["wikidata_id"] = df["wikidata_id"].apply(_int_to_qid)
        return df[["wikidata_id", "language_code", "page_title"]]

    """
    DuckDB implementation of CacheHandler.

    Wikidata ID Conventions:
    ------------------------
    - **Internal storage**: IDs stored as integers (Q prefix removed)
    - **Interface contract**: ALL methods return IDs WITH Q prefix via _int_to_qid()
    - **Input handling**: Methods accept both formats (with/without Q)

    This ensures downstream code can always assume Q prefix is present.
    """

    def get_title_for_qid(self, qid: str, language_code: str) -> Optional[str]:
        """
        Returns the page title for the given QID and language code, or None if not found.
        """
        qid_int = _qid_to_int(qid)
        if qid_int is None:
            return None
        query = """
            SELECT page_title FROM wiki_page
            WHERE wikidata_id = ? AND language_code = ?
            LIMIT 1
        """
        result = self._conn.execute(query, [qid_int, language_code]).fetchone()
        if result:
            return result[0]
        return None

    def __init__(self, db_filename: str):
        # Place DB in data folder if only filename is given
        if not os.path.dirname(db_filename):
            db_filename = os.path.join("data", db_filename)

        logger.info(f"Initializing DuckDBHandler with database: {db_filename}")

        # Check if the database file exists (unless it will be created)
        # We allow creation of new databases, but warn if path seems incorrect
        db_dir = os.path.dirname(db_filename)
        if db_dir and not os.path.exists(db_dir):
            raise FileNotFoundError(
                f"Database directory does not exist: {db_dir}\n"
                f"Full path attempted: {os.path.abspath(db_filename)}\n"
                f"Please verify the database path is correct."
            )

        # If file doesn't exist, log that we're creating a new database
        if not os.path.exists(db_filename):
            logger.warning(
                f"Database file does not exist and will be created: {os.path.abspath(db_filename)}"
            )

        # Connect and initialize
        self._conn: DuckDBPyConnection = duckdb.connect(
            database=db_filename,
            config={
                "threads": str(DUCKDB_THREADS),
                "memory_limit": DUCKDB_MEMORY_LIMIT,
                "max_temp_directory_size": DUCKDB_MAX_TEMP_DIR_SIZE,
                "temp_directory": DUCKDB_TEMP_DIR,
                "parquet_metadata_cache": str(True),
            },
        )
        self._init_db()

    def close(self):
        logger.debug("Closing DuckDB connection")
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _init_db(self):
        logger.debug("Initializing database schema")
        con = self._conn
        con.sql(
            """
            CREATE SEQUENCE IF NOT EXISTS wiki_page_id START 1;
            CREATE SEQUENCE IF NOT EXISTS page_link_id START 1;

            CREATE TABLE IF NOT EXISTS wikidata_entity (
                wikidata_id INT PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS wiki_page (
                wiki_page_id INTEGER PRIMARY KEY DEFAULT nextval('wiki_page_id'),
                wikidata_id INT NOT NULL,
                language_code TEXT NOT NULL,
                page_title TEXT NOT NULL,
                FOREIGN KEY (wikidata_id) REFERENCES wikidata_entity(wikidata_id),
                UNIQUE(wikidata_id, language_code)
            );
            CREATE TABLE IF NOT EXISTS extract (
                wiki_page_id INT PRIMARY KEY,
                extract TEXT,
                FOREIGN KEY (wiki_page_id) REFERENCES wiki_page(wiki_page_id)
            );      
            CREATE TABLE IF NOT EXISTS page_link (
                link_id INTEGER PRIMARY KEY DEFAULT nextval('page_link_id'),
                source_wikidata_id INTEGER NOT NULL,
                target_wikidata_id INTEGER NOT NULL,
                language_code TEXT NOT NULL,
                notable BOOLEAN DEFAULT FALSE,
                UNIQUE(source_wikidata_id, target_wikidata_id, language_code),
                FOREIGN KEY (source_wikidata_id) REFERENCES wikidata_entity(wikidata_id),
                -- Note: target_wikidata_id may not be in wikidata_entity table
                -- Because we may expand the notable set later and want to keep links to potential future notables
            );
            CREATE TABLE IF NOT EXISTS weights (
                link_id INTEGER NOT NULL,
                weight REAL NOT NULL,
                method TEXT NOT NULL,
                PRIMARY KEY(link_id, method),
                FOREIGN KEY (link_id) REFERENCES page_link(link_id)
            );
            CREATE TABLE IF NOT EXISTS pageview (
                wiki_page_id INTEGER NOT NULL,
                language_code TEXT NOT NULL,
                interval_start DATE NOT NULL,
                interval_end DATE NOT NULL,
                pageviews INTEGER NOT NULL,
                PRIMARY KEY(wiki_page_id, interval_start, interval_end),
                FOREIGN KEY (wiki_page_id) REFERENCES wiki_page(wiki_page_id)
            );

            """
        )
        logging.info("Creating indexes...")
        con.sql(
            """
        CREATE INDEX IF NOT EXISTS idx_page_link_source_lang 
            ON page_link(source_wikidata_id, language_code);
        
        CREATE INDEX IF NOT EXISTS idx_page_link_notable 
            ON page_link(notable);
        
        CREATE INDEX IF NOT EXISTS idx_weights_link_method 
            ON weights(link_id, method);
        
        CREATE INDEX IF NOT EXISTS idx_wiki_page_lang 
            ON wiki_page(language_code);
        
        CREATE INDEX IF NOT EXISTS idx_pageview_interval
            ON pageview(interval_start, interval_end);
    """
        )
        logging.info("DuckDB database initialized.")
        logging.info("Ending _init_db")

    def insert_entity_titles(self, qid: str, titles: Dict[str, str]) -> None:
        """
        Inserts or updates the page titles for a given entity (qid) for each language in the titles dict.
        """
        qid_int = _qid_to_int(qid)
        if qid_int is None or not titles:
            return
        for lang, title in titles.items():
            self._conn.execute(
                """
                INSERT INTO wiki_page (wikidata_id, language_code, page_title)
                VALUES (?, ?, ?)
                ON CONFLICT(wikidata_id, language_code)
                DO UPDATE SET page_title = excluded.page_title
                """,
                (qid_int, lang, title),
            )
        self._conn.commit()

    def get_df_missing_extracts_by_lang(
        self, qids: List[str], langs: List[str]
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns 'wiki_page_id', 'wikidata_id', 'page_title' and 'language_code'
        for page titles missing an extract for that language.
        """
        qid_ints = [
            qid_int for qid in qids if (qid_int := _qid_to_int(qid)) is not None
        ]
        if not qid_ints:
            return pd.DataFrame(
                columns=["wiki_page_id", "wikidata_id", "page_title", "language_code"]
            )

        qid_tuple = tuple(qid_ints)
        lang_tuple = tuple(langs)

        query = f"""
        SELECT wp.wiki_page_id, wp.wikidata_id, wp.page_title, wp.language_code
            FROM wiki_page wp
            LEFT JOIN extract e ON wp.wiki_page_id = e.wiki_page_id
            WHERE (e.extract IS NULL OR e.extract = '')
            AND wp.wikidata_id IN {qid_tuple}
            AND wp.language_code IN {lang_tuple};
        """

        results = self._conn.execute(query).fetchdf()

        # Convert wikidata_id back to QID format
        if not results.empty:
            results["wikidata_id"] = results["wikidata_id"].apply(_int_to_qid)
        else:
            return pd.DataFrame(
                columns=["wiki_page_id", "wikidata_id", "page_title", "language_code"]
            )
        return results[["wiki_page_id", "wikidata_id", "page_title", "language_code"]]

    def insert_entity_titles_from_df(self, titles_df: pd.DataFrame) -> None:
        """
        Inserts or updates the page titles for entities from a DataFrame with columns 'wikidata_id', 'page_title', and 'language_code'.
        """
        if titles_df.empty:
            return
        for _, row in titles_df.iterrows():
            qid = row["wikidata_id"]
            title = row["page_title"]
            lang = row["language_code"]
            qid_int = _qid_to_int(qid)
            if qid_int is None or not title or not lang:
                continue
            self._conn.execute(
                """
                INSERT INTO wiki_page (wikidata_id, language_code, page_title)
                VALUES (?, ?, ?)
                ON CONFLICT(wikidata_id, language_code)
                DO UPDATE SET page_title = excluded.page_title
                """,
                (qid_int, lang, title),
            )
        self._conn.commit()

    def get_entities_missing_titles(
        self, qids: List[str], langs: List[str]
    ) -> List[str]:
        """
        Returns a list of QIDs (as strings) for the qids provided that are missing page titles for the specified languages.
        """
        qids_ints = [
            qid_int for qid in qids if (qid_int := _qid_to_int(qid)) is not None
        ]
        if not qids_ints:
            return []

        qid_tuple = tuple(qids_ints)
        lang_tuple = tuple(langs)

        query = f"""
        SELECT wikidata_id, language_code FROM wiki_page
        WHERE wikidata_id IN {qid_tuple}
        AND language_code IN {lang_tuple}
        """

        results = self._conn.execute(query).fetchall()
        dict_found: Dict[int, Set[str]] = {}
        for row in results:
            wikidata_id = row[0]
            language_code = row[1]
            if wikidata_id not in dict_found:
                dict_found[wikidata_id] = set()
            dict_found[wikidata_id].add(language_code)

        for qid_int in qids_ints:
            if qid_int not in dict_found:
                dict_found[qid_int] = set()  # Ensure all qids are represented

        missing_qids: Set[str] = set()
        for qid_int in qids_ints:
            found_langs = dict_found.get(qid_int, set())
            for lang in langs:
                if lang not in found_langs:
                    missing_qids.add(_int_to_qid(qid_int))
                    break  # No need to check other languages for this qid

        return list(missing_qids)

    def load_notable_entity_ids(self, qids: List[str]) -> None:
        """
        Loads notable entity IDs into the wikidata_entity table efficiently.
        """
        qid_ints = [
            qid_int for qid in qids if (qid_int := _qid_to_int(qid)) is not None
        ]
        if not qid_ints:
            return

        import pandas as pd

        df = pd.DataFrame({"wikidata_id": qid_ints})

        self._conn.register("notable_qids_df", df)
        self._conn.execute(
            """
            INSERT INTO wikidata_entity (wikidata_id)
            SELECT wikidata_id FROM notable_qids_df
            ON CONFLICT(wikidata_id) DO NOTHING
            """
        )
        self._conn.unregister("notable_qids_df")
        self._conn.commit()

    def refresh_notable_links(self, force: bool = False) -> None:
        """
        Update the notable flag in page_link table for links where both
        source and target are in wikidata_entity (i.e., both are notable).

        Args:
            force: If True, reset all flags and rebuild.
                   If False, only update if no flags are currently set.
        """
        logging.info("Starting refresh_notable_links")

        # Check if notable flags are already set
        count_result = self._conn.execute(
            "SELECT COUNT(*) FROM page_link WHERE notable = TRUE"
        ).fetchone()
        existing_count = count_result[0] if count_result else 0

        if not force and existing_count > 0:
            logging.info(
                f"notable flags already set for {existing_count} links. Skipping refresh."
            )
            return

        # Reset all flags if force rebuild
        if force:
            logging.info("Force rebuild requested, resetting all notable flags...")
            self._conn.execute("UPDATE page_link SET notable = FALSE")
            self._conn.commit()

        # Get total links before filtering
        total_links = self._conn.execute("SELECT COUNT(*) FROM page_link").fetchone()[0]

        logging.info(f"Updating notable flags for {total_links} total links...")

        # Update notable flag for links where both source and target are notable entities
        self._conn.execute(
            """
            UPDATE page_link
            SET notable = TRUE
            WHERE EXISTS (
                SELECT 1 FROM wikidata_entity we1 WHERE page_link.source_wikidata_id = we1.wikidata_id
            )
            AND EXISTS (
                SELECT 1 FROM wikidata_entity we2 WHERE page_link.target_wikidata_id = we2.wikidata_id
            )
        """
        )
        self._conn.commit()

        # Get count after filtering
        notable_count = self._conn.execute(
            "SELECT COUNT(*) FROM page_link WHERE notable = TRUE"
        ).fetchone()[0]

        reduction_pct = (
            ((total_links - notable_count) / total_links * 100)
            if total_links > 0
            else 0
        )

        logging.info(
            f"notable flags set: {notable_count} links ({reduction_pct:.1f}% reduction)"
        )
        logging.info("Ending refresh_notable_links")

    def insert_entity_extracts_from_df(self, extracts_df: pd.DataFrame) -> None:
        """
        Inserts or updates the extracts for entities from a DataFrame
        with index 'wiki_page_id' and 'extract' column.
        """
        if extracts_df.empty:
            return

        # Ensure the DataFrame has the required columns
        if (
            "wiki_page_id" not in extracts_df.columns
            or "extract" not in extracts_df.columns
        ):
            raise ValueError(
                "DataFrame must have 'wiki_page_id' and 'extract' columns."
            )

        df = extracts_df.dropna(subset=["wiki_page_id", "extract"])

        # Register the DataFrame as a DuckDB table
        self._conn.register("extracts_df", df)

        self._conn.execute(
            """
            INSERT INTO extract (wiki_page_id, extract)
            SELECT wiki_page_id, extract
            FROM extracts_df
            ON CONFLICT(wiki_page_id)
            DO UPDATE SET extract = excluded.extract
            """
        )
        self._conn.unregister("extracts_df")
        self._conn.commit()

    def get_df_missing_links_by_lang(
        self, qids: List[str], langs: List[str]
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns 'page_title' and 'language_code' for the (qid, lang) pairs
        that do not have any link for the specified languages.
        """
        qids_ints = [
            qid_int for qid in qids if (qid_int := _qid_to_int(qid)) is not None
        ]
        if not qids_ints:
            return pd.DataFrame(columns=["wikidata_id", "page_title", "language_code"])

        qid_tuple = tuple(qids_ints)
        lang_tuple = tuple(langs)

        query = f"""
        SELECT wp.wikidata_id, wp.page_title, wp.language_code
        FROM wiki_page wp
        LEFT JOIN page_link pl
            ON wp.wikidata_id = pl.source_wikidata_id
            AND wp.language_code = pl.language_code
        WHERE wp.wikidata_id IN {qid_tuple}
          AND wp.language_code IN {lang_tuple}
          AND pl.source_wikidata_id IS NULL
        """
        df = self._conn.execute(query).fetchdf()
        if not df.empty:
            df["wikidata_id"] = df["wikidata_id"].apply(_int_to_qid)
        return df[["wikidata_id", "page_title", "language_code"]]

    def insert_entity_links_from_df(self, links_df: pd.DataFrame) -> None:
        """
        Inserts entity links from a DataFrame with at least columns 'source_wikidata_id', 'target_wikidata_id', and 'language_code'.
        Only these columns are used. Assumes there will be no conflicts.
        """
        if links_df.empty:
            return

        # Restrict to required columns and drop rows with missing values
        required_cols = ["source_wikidata_id", "target_wikidata_id", "language_code"]
        df = links_df[required_cols].dropna(subset=required_cols).copy()

        # Convert QIDs to integers
        df["source_wikidata_id"] = df["source_wikidata_id"].apply(_qid_to_int)
        df["target_wikidata_id"] = df["target_wikidata_id"].apply(_qid_to_int)
        df = df.dropna(
            subset=["source_wikidata_id", "target_wikidata_id", "language_code"]
        )

        # Register DataFrame and insert
        self._conn.register("links_df_temp", df)
        self._conn.execute(
            """
            INSERT INTO page_link (source_wikidata_id, target_wikidata_id, language_code)
            SELECT source_wikidata_id, target_wikidata_id, language_code
            FROM links_df_temp
            ON CONFLICT (source_wikidata_id, target_wikidata_id, language_code) DO NOTHING
            """
        )
        self._conn.unregister("links_df_temp")
        self._conn.commit()

    def get_entities_and_extracts_missing_weights(
        self, langs: List[str], method: str
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
        'wikidata_id', 'wiki_page_id', 'page_title', 'extract', 'language_code'
        for pages that are missing a weight for the given method, for all specified languages.
        """
        logging.info("Handler: retrieving entities missing weights...")

        if not langs:
            return pd.DataFrame(
                columns=[
                    "wikidata_id",
                    "wiki_page_id",
                    "page_title",
                    "extract",
                    "language_code",
                ]
            )

        placeholders = ", ".join(["?"] * len(langs))

        query = f"""
        SELECT DISTINCT
            wp.wikidata_id,
            wp.wiki_page_id,
            wp.page_title,
            e.extract,
            wp.language_code
        FROM wiki_page wp
        JOIN extract e ON wp.wiki_page_id = e.wiki_page_id
        WHERE wp.language_code IN ({placeholders})
        AND EXISTS (
            SELECT 1 
            FROM page_link pl
            LEFT JOIN weights w ON pl.link_id = w.link_id AND w.method = ?
            WHERE pl.source_wikidata_id = wp.wikidata_id
                AND pl.language_code = wp.language_code
                AND pl.notable = TRUE
                AND w.weight IS NULL
        )
        """

        params = list(langs) + [method]
        df = self._conn.execute(query, params).fetchdf()
        if not df.empty:
            df["wikidata_id"] = df["wikidata_id"].apply(_int_to_qid)
        else:
            df = pd.DataFrame(
                columns=[
                    "wikidata_id",
                    "wiki_page_id",
                    "page_title",
                    "extract",
                    "language_code",
                ]
            )
        return df

    def get_title_to_id_map(self, langs: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Returns a dictionary mapping language_code to a dictionary of {page_title: wikidata_id (QID string)}
        for all titles in the database for the specified languages.
        """
        if not langs:
            return {}

        placeholders = ", ".join(["?"] * len(langs))
        query = f"""
            SELECT page_title, wikidata_id, language_code
            FROM wiki_page
            WHERE language_code IN ({placeholders})
        """
        df = self._conn.execute(query, langs).fetchdf()
        if df.empty:
            return {}

        # Convert wikidata_id to QID string
        df["wikidata_id"] = df["wikidata_id"].apply(_int_to_qid)

        title_to_id_map: Dict[str, Dict[str, str]] = {}
        for lang in df["language_code"].unique():
            lang_df = df[df["language_code"] == lang]
            title_to_id_map[lang] = dict(
                zip(lang_df["page_title"], lang_df["wikidata_id"])
            )
        return title_to_id_map

    def insert_link_weights(self, weights_df: pd.DataFrame, method: str) -> None:
        """
        Inserts or updates weights for links from a DataFrame with columns:
        'source_wikidata_id', 'target_wikidata_id', 'language_code', 'weight'.
        The 'method' argument is used for all rows.
        """
        if weights_df.empty:
            return

        # Convert QIDs to integers
        df = weights_df.copy()
        df["source_wikidata_id"] = df["source_wikidata_id"].apply(_qid_to_int)
        df["target_wikidata_id"] = df["target_wikidata_id"].apply(_qid_to_int)
        df = df.dropna(
            subset=[
                "source_wikidata_id",
                "target_wikidata_id",
                "language_code",
                "weight",
            ]
        )

        # Register DataFrame for DuckDB
        self._conn.register("weights_temp_df", df)

        # Insert or update weights
        self._conn.execute(
            f"""
            INSERT INTO weights (link_id, weight, method)
            SELECT pl.link_id, w.weight, ?
            FROM weights_temp_df w
            JOIN page_link pl
              ON pl.source_wikidata_id = w.source_wikidata_id
             AND pl.target_wikidata_id = w.target_wikidata_id
             AND pl.language_code = w.language_code
            ON CONFLICT(link_id, method)
            DO UPDATE SET weight = excluded.weight
            """,
            [method],
        )
        self._conn.unregister("weights_temp_df")
        self._conn.commit()

    def save_weighted_links_to_csv(self, method: str, output_filename: str) -> None:
        """
        Exports all weights for the specified method to a CSV file.
        Columns: language_code, source_wikidata_id, target_wikidata_id, weight
        """
        query = """
        SELECT
            pl.language_code,
            pl.source_wikidata_id,
            pl.target_wikidata_id,
            w.weight
        FROM weights w
        JOIN page_link pl ON w.link_id = pl.link_id
        WHERE w.method = ?
    """
        df = self._conn.execute(query, [method]).fetchdf()
        if not df.empty:
            # Convert IDs to QID format
            df["source_wikidata_id"] = df["source_wikidata_id"].apply(_int_to_qid)
            df["target_wikidata_id"] = df["target_wikidata_id"].apply(_int_to_qid)
            df.to_csv(output_filename, index=False)
            logging.info(f"Exported {len(df)} weighted links to {output_filename}")
        else:
            logging.info("No weighted links found for export.")

    def get_links_for_entities(
        self, qids: List[str], langs: List[str], notable_only: bool = False
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with all links (source_wikidata_id, target_wikidata_id, language_code, link_id)
        for the given entities (as QIDs) and languages.

        Args:
            qids: List of Wikidata IDs (as strings like 'Q123')
            langs: List of language codes
            notable_only: If True, only return links where both source and target are notable
        """
        qid_ints = [
            qid_int for qid in qids if (qid_int := _qid_to_int(qid)) is not None
        ]
        if not qid_ints or not langs:
            return pd.DataFrame(
                columns=[
                    "source_wikidata_id",
                    "target_wikidata_id",
                    "language_code",
                    "link_id",
                ]
            )

        notable_filter = "AND notable = TRUE" if notable_only else ""
        placeholders_qids = ", ".join(["?"] * len(qid_ints))
        placeholders_langs = ", ".join(["?"] * len(langs))
        query = f"""
            SELECT source_wikidata_id, target_wikidata_id, language_code, link_id
            FROM page_link
            WHERE source_wikidata_id IN ({placeholders_qids})
              AND language_code IN ({placeholders_langs})
              {notable_filter}
        """
        params = qid_ints + langs
        df = self._conn.execute(query, params).fetchdf()
        if not df.empty:
            df["source_wikidata_id"] = df["source_wikidata_id"].apply(_int_to_qid)
            df["target_wikidata_id"] = df["target_wikidata_id"].apply(_int_to_qid)
        return df[
            ["source_wikidata_id", "target_wikidata_id", "language_code", "link_id"]
        ]

    def get_pages_data(self, qids: List[str], langs: List[str]) -> pd.DataFrame:
        """
        Returns a DataFrame with pages data for the given entities (as QIDs) and languages.
        Columns: wikidata_id, wiki_page_id, page_title, extract, language_code
        """
        qid_ints = [
            qid_int for qid in qids if (qid_int := _qid_to_int(qid)) is not None
        ]
        if not qid_ints or not langs:
            return pd.DataFrame(
                columns=[
                    "wikidata_id",
                    "wiki_page_id",
                    "page_title",
                    "extract",
                    "language_code",
                ]
            )

        placeholders_qids = ", ".join(["?"] * len(qid_ints))
        placeholders_langs = ", ".join(["?"] * len(langs))
        query = f"""
            SELECT 
                wp.wikidata_id,
                wp.wiki_page_id,
                wp.page_title,
                e.extract,
                wp.language_code
            FROM wiki_page wp
            JOIN extract e ON wp.wiki_page_id = e.wiki_page_id
            WHERE wp.wikidata_id IN ({placeholders_qids})
              AND wp.language_code IN ({placeholders_langs})
        """
        params = qid_ints + langs
        df = self._conn.execute(query, params).fetchdf()
        if not df.empty:
            df["wikidata_id"] = df["wikidata_id"].apply(_int_to_qid)
        return df[
            ["wikidata_id", "wiki_page_id", "page_title", "extract", "language_code"]
        ]

    def get_titles_for_qids_english(self, qid_list, language_code="en"):
        """
        Returns a DataFrame mapping QID to page title for the given QIDs in the specified language.
        """
        qid_ints = [
            _qid_to_int(qid) for qid in qid_list if _qid_to_int(qid) is not None
        ]
        if not qid_ints:
            import pandas as pd

            return pd.DataFrame(columns=["qid", "title"])

        placeholders = ", ".join(["?"] * len(qid_ints))
        query = f"""
            SELECT wikidata_id, page_title
            FROM wiki_page
            WHERE wikidata_id IN ({placeholders})
            AND language_code = ?
        """
        result = self._conn.execute(query, qid_ints + [language_code]).fetchall()
        import pandas as pd

        df = pd.DataFrame(result, columns=["qid", "title"])
        df["qid"] = df["qid"].apply(_int_to_qid)
        return df

    def get_pages_missing_pageviews(
        self, interval_start: str, interval_end: str
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns 'wiki_page_id', 'page_title', 'language_code'
        for pages that are missing pageview data for the specified interval.

        Args:
            interval_start: Start date in ISO format (YYYY-MM-DD)
            interval_end: End date in ISO format (YYYY-MM-DD)
        """
        query = """
        SELECT wp.wiki_page_id, wp.page_title, wp.language_code
        FROM wiki_page wp
        LEFT JOIN pageview pv
            ON wp.wiki_page_id = pv.wiki_page_id
            AND pv.interval_start = ?
            AND pv.interval_end = ?
        WHERE pv.pageviews IS NULL
        """
        df = self._conn.execute(query, [interval_start, interval_end]).fetchdf()
        return df[["wiki_page_id", "page_title", "language_code"]]

    def insert_pageviews(self, pageviews_df: pd.DataFrame) -> None:
        """
        Inserts pageview data from a DataFrame with columns:
        'wiki_page_id', 'language_code', 'interval_start', 'interval_end', 'pageviews'.
        """
        if pageviews_df.empty:
            return

        required_cols = [
            "wiki_page_id",
            "language_code",
            "interval_start",
            "interval_end",
            "pageviews",
        ]
        df = pageviews_df[required_cols].dropna(subset=required_cols).copy()

        if df.empty:
            return

        self._conn.register("pageviews_temp_df", df)
        self._conn.execute(
            """
            INSERT INTO pageview (wiki_page_id, language_code, interval_start, interval_end, pageviews)
            SELECT wiki_page_id, language_code, interval_start, interval_end, pageviews
            FROM pageviews_temp_df
            ON CONFLICT(wiki_page_id, interval_start, interval_end) 
            DO UPDATE SET pageviews = excluded.pageviews
            """
        )
        self._conn.unregister("pageviews_temp_df")
        self._conn.commit()

    def get_all_pageviews(self, interval_start: str, interval_end: str) -> pd.DataFrame:
        """
        Returns all pageview data for the specified interval.
        Columns: wiki_page_id, wikidata_id, language_code, pageviews
        """
        query = """
        SELECT pv.wiki_page_id, wp.wikidata_id, pv.language_code, pv.pageviews
        FROM pageview pv
        JOIN wiki_page wp ON pv.wiki_page_id = wp.wiki_page_id
        WHERE pv.interval_start = ? AND pv.interval_end = ?
        """
        df = self._conn.execute(query, [interval_start, interval_end]).fetchdf()
        if not df.empty:
            df["wikidata_id"] = df["wikidata_id"].apply(_int_to_qid)
        return df

    def get_filtered_weighted_links(
        self,
        method: str,
        interval_start: str,
        interval_end: str,
        threshold: float,
    ) -> pd.DataFrame:
        """
        Returns weighted links where both source and target pages have pageviews >= threshold.
        Only considers notable links (where notable flag is TRUE).
        The threshold is already normalized if needed (done in Python).

        Columns: source_wikidata_id, target_wikidata_id, weight, language_code
        """
        query = """
        SELECT
            pl.source_wikidata_id,
            pl.target_wikidata_id,
            w.weight,
            pl.language_code
        FROM weights w
        JOIN page_link pl ON w.link_id = pl.link_id
        JOIN wiki_page wp_source ON pl.source_wikidata_id = wp_source.wikidata_id 
            AND pl.language_code = wp_source.language_code
        JOIN wiki_page wp_target ON pl.target_wikidata_id = wp_target.wikidata_id 
            AND pl.language_code = wp_target.language_code
        JOIN pageview pv_source ON wp_source.wiki_page_id = pv_source.wiki_page_id
            AND pv_source.language_code = wp_source.language_code
            AND pv_source.interval_start = ?
            AND pv_source.interval_end = ?
        JOIN pageview pv_target ON wp_target.wiki_page_id = pv_target.wiki_page_id
            AND pv_target.language_code = wp_target.language_code
            AND pv_target.interval_start = ?
            AND pv_target.interval_end = ?
        WHERE w.method = ?
            AND pl.notable = TRUE
            AND pv_source.pageviews >= ?
            AND pv_target.pageviews >= ?
        """
        params = [
            interval_start,
            interval_end,
            interval_start,
            interval_end,
            method,
            threshold,
            threshold,
        ]
        df = self._conn.execute(query, params).fetchdf()
        if not df.empty:
            df["source_wikidata_id"] = df["source_wikidata_id"].apply(_int_to_qid)
            df["target_wikidata_id"] = df["target_wikidata_id"].apply(_int_to_qid)
        return df[
            ["source_wikidata_id", "target_wikidata_id", "weight", "language_code"]
        ]
