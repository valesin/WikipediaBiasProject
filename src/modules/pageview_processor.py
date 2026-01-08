"""
Pageview Processor - Fetches and processes Wikipedia pageviews

This module handles:
1. Fetching pageview data from the Wikimedia REST API
2. Storing pageviews in the database
3. Filtering edges based on pageview thresholds
"""

import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd

from modules.duckdb_handler import DuckDBHandler
from modules.wiki_api_request_handler import WikiApiRequestHandler
from modules.spotlight_weight_source import SpotlightWeightSource


logger = logging.getLogger(__name__)


class PageviewProcessor:
    """
    Handles fetching pageviews and filtering edges based on pageview thresholds.
    """

    def __init__(
        self,
        cache_handler: DuckDBHandler,
        api_handler: WikiApiRequestHandler,
        interval_start: str,
        interval_end: str,
    ):
        """
        Initialize the pageview processor.

        Args:
            cache_handler: DuckDB handler for database operations
            api_handler: Wikipedia API handler for fetching pageviews
            interval_start: Start date in ISO format (YYYY-MM-DD)
            interval_end: End date in ISO format (YYYY-MM-DD)
        """
        self.cache_handler = cache_handler
        self.api_handler = api_handler
        self.interval_start = interval_start
        self.interval_end = interval_end

    def _iso_to_yyyymmdd(self, date_str: str) -> str:
        """Convert ISO date (YYYY-MM-DD) to YYYYMMDD format."""
        return date_str.replace("-", "")

    def _calculate_days_in_interval(self) -> int:
        """Calculate the number of days between interval_start and interval_end (inclusive)."""
        from datetime import datetime

        start = datetime.fromisoformat(self.interval_start)
        end = datetime.fromisoformat(self.interval_end)
        return (end - start).days + 1

    def fetch_missing_pageviews(
        self, batch_size: int = 50, log_frequency: int = 1000
    ) -> None:
        """
        Fetch pageviews for all pages missing data for the configured interval.

        Args:
            batch_size: Number of pages to process per batch
            log_frequency: Number of pages between progress logs (default: 1000)
        """
        logger.info(
            f"Fetching pageviews for interval {self.interval_start} to {self.interval_end}"
        )

        # Get pages missing pageview data
        missing_df = self.cache_handler.get_pages_missing_pageviews(
            self.interval_start, self.interval_end
        )

        if missing_df.empty:
            logger.info("No missing pageviews to fetch")
            return

        total_pages = len(missing_df)
        logger.info(f"Fetching pageviews for {total_pages} pages")

        # Convert dates to YYYYMMDD format for API
        start_yyyymmdd = self._iso_to_yyyymmdd(self.interval_start)
        end_yyyymmdd = self._iso_to_yyyymmdd(self.interval_end)

        # Calculate number of days for averaging
        num_days = self._calculate_days_in_interval()
        logger.info(f"Interval spans {num_days} days - will calculate daily average")

        # Process in batches
        pageviews_data = []
        success_count = 0
        fail_count = 0
        processed_count = 0

        for i in range(0, total_pages, batch_size):
            batch_df = missing_df.iloc[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_pages + batch_size - 1) // batch_size

            logger.debug(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_df)} pages)"
            )

            for _, row in batch_df.iterrows():
                wiki_page_id = row["wiki_page_id"]
                title = row["page_title"]
                language = row["language_code"]

                # Fetch pageviews from API
                pageviews = self.api_handler.fetch_pageviews(
                    title, language, start_yyyymmdd, end_yyyymmdd
                )

                if pageviews is not None:
                    # Calculate daily average
                    daily_avg = pageviews / num_days
                    pageviews_data.append(
                        {
                            "wiki_page_id": wiki_page_id,
                            "language_code": language,
                            "interval_start": self.interval_start,
                            "interval_end": self.interval_end,
                            "pageviews": daily_avg,
                        }
                    )
                    success_count += 1
                else:
                    fail_count += 1
                    logger.debug(
                        f"Failed to fetch pageviews for '{title}' ({language})"
                    )

                processed_count += 1

                # Progress logging
                if (
                    processed_count % log_frequency == 0
                    or processed_count == total_pages
                ):
                    logger.info(
                        f"Fetched pageviews for {processed_count}/{total_pages} pages "
                        f"({success_count} succeeded, {fail_count} failed)"
                    )

            # Insert batch results into database
            if pageviews_data:
                batch_pageviews_df = pd.DataFrame(pageviews_data)
                self.cache_handler.insert_pageviews(batch_pageviews_df)
                logger.debug(f"Inserted {len(pageviews_data)} pageview records")
                pageviews_data = []  # Clear for next batch

        logger.info(
            f"Pageview fetching complete: {success_count} fetched, {fail_count} failed"
        )

    def fetch_missing_pageviews_parallel(
        self,
        batch_size: int = 500,
        max_workers: Optional[int] = None,
        log_frequency: int = 1000,
    ) -> None:
        """
        Fetch pageviews for all pages missing data using parallel threading.

        This is a parallel version of fetch_missing_pageviews that uses
        ThreadPoolExecutor to make concurrent API requests, significantly
        improving performance for large datasets.

        Args:
            batch_size: Number of pages to process before committing to DB
            max_workers: Maximum number of concurrent threads.
                        Defaults to requests_per_second (typically 180) for
                        optimal rate limit utilization.
            log_frequency: Number of pages between progress logs (default: 1000)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from modules.thread_safe_rate_limiter import SlidingWindowRateLimiter

        logger.info(
            f"Fetching pageviews (PARALLEL) for interval {self.interval_start} to {self.interval_end}"
        )

        # Get pages missing pageview data
        missing_df = self.cache_handler.get_pages_missing_pageviews(
            self.interval_start, self.interval_end
        )

        if missing_df.empty:
            logger.info("No missing pageviews to fetch")
            return

        # Default max_workers to match rate limit for optimal performance
        if max_workers is None:
            max_workers = self.api_handler.requests_per_second

        total_pages = len(missing_df)
        logger.info(
            f"Fetching pageviews for {total_pages} pages using {max_workers} workers "
            f"(rate limit: {self.api_handler.requests_per_second} req/s)"
        )

        # Convert dates to YYYYMMDD format for API
        start_yyyymmdd = self._iso_to_yyyymmdd(self.interval_start)
        end_yyyymmdd = self._iso_to_yyyymmdd(self.interval_end)

        # Calculate number of days for averaging
        num_days = self._calculate_days_in_interval()
        logger.info(f"Interval spans {num_days} days - will calculate daily average")

        # Create rate limiter for the API handler
        rate_limiter = SlidingWindowRateLimiter(
            max_per_second=self.api_handler.requests_per_second
        )

        # Wrapper function that includes rate limiting
        def fetch_with_rate_limit(row):
            """Fetch pageviews for a single page with rate limiting."""
            rate_limiter.acquire()

            wiki_page_id = row["wiki_page_id"]
            title = row["page_title"]
            language = row["language_code"]

            pageviews = self.api_handler.fetch_pageviews(
                title, language, start_yyyymmdd, end_yyyymmdd
            )

            if pageviews is not None:
                daily_avg = pageviews / num_days
                return {
                    "wiki_page_id": wiki_page_id,
                    "language_code": language,
                    "interval_start": self.interval_start,
                    "interval_end": self.interval_end,
                    "pageviews": daily_avg,
                }
            return None

        # Process with thread pool
        pageviews_data = []
        success_count = 0
        fail_count = 0
        processed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(fetch_with_rate_limit, row): idx
                for idx, row in missing_df.iterrows()
            }

            # Process results as they complete
            for future in as_completed(futures):
                processed_count += 1

                try:
                    result = future.result()
                    if result is not None:
                        pageviews_data.append(result)
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    logger.debug(f"Exception in worker thread: {e}")

                # Progress logging
                if (
                    processed_count % log_frequency == 0
                    or processed_count == total_pages
                ):
                    logger.info(
                        f"Fetched pageviews for {processed_count}/{total_pages} pages "
                        f"({success_count} succeeded, {fail_count} failed)"
                    )

                # Insert batch when we hit batch_size
                if len(pageviews_data) >= batch_size:
                    batch_pageviews_df = pd.DataFrame(pageviews_data)
                    self.cache_handler.insert_pageviews(batch_pageviews_df)
                    logger.debug(
                        f"Inserted batch of {len(pageviews_data)} pageview records"
                    )
                    pageviews_data = []

            # Insert remaining data
            if pageviews_data:
                batch_pageviews_df = pd.DataFrame(pageviews_data)
                self.cache_handler.insert_pageviews(batch_pageviews_df)
                logger.debug(f"Inserted final batch of {len(pageviews_data)} records")

        logger.info(
            f"Pageview fetching complete: {success_count} fetched, {fail_count} failed"
        )

    def export_pageviews_to_csv(self, output_filename: str) -> None:
        """
        Export pageview data to CSV for the configured interval.

        Args:
            output_filename: Path to output CSV file
        """
        logger.info(
            f"Exporting pageviews for interval {self.interval_start} to {self.interval_end}"
        )

        # Get all pageviews for the interval
        pageviews_df = self.cache_handler.get_all_pageviews(
            self.interval_start, self.interval_end
        )

        if pageviews_df.empty:
            logger.warning("No pageview data found for interval. Creating empty CSV.")
            # Create empty CSV with headers
            pageviews_df = pd.DataFrame(
                columns=["wikidata_id", "language_code", "pageviews"]
            )
        else:
            logger.info(f"Found pageview data for {len(pageviews_df)} pages")
            logger.info(
                f"Pageview statistics - Min: {pageviews_df['pageviews'].min():.2f}, "
                f"Median: {pageviews_df['pageviews'].median():.2f}, "
                f"Max: {pageviews_df['pageviews'].max():.2f}"
            )

        # Select only the required columns for export
        export_df = pageviews_df[["wikidata_id", "language_code", "pageviews"]]

        # Save to CSV
        export_df.to_csv(output_filename, index=False)
        logger.info(f"Exported {len(export_df)} pageview records to {output_filename}")


def process_pageviews(
    graph_db_path: str,
    interval_start: str,
    interval_end: str,
    output_dir: str = "data",
    batch_size: int = 50,
) -> None:
    """
    Main function to fetch and export pageviews to CSV.

    Args:
        graph_db_path: Path to the graph database
        interval_start: Start date in ISO format (YYYY-MM-DD)
        interval_end: End date in ISO format (YYYY-MM-DD)
        output_dir: Directory to save output CSV
        batch_size: Number of pages to process per API batch
    """
    start_time = time.time()

    # Initialize handlers
    cache_handler = DuckDBHandler(graph_db_path)
    api_handler = WikiApiRequestHandler()

    # Create processor
    processor = PageviewProcessor(
        cache_handler=cache_handler,
        api_handler=api_handler,
        interval_start=interval_start,
        interval_end=interval_end,
    )

    # Step 1: Fetch missing pageviews
    logger.info("Step 1: Fetching missing pageviews")
    processor.fetch_missing_pageviews(batch_size=batch_size)

    # Step 2: Export pageviews to CSV
    logger.info("Step 2: Exporting pageviews to CSV")

    # Generate output filename based on interval
    output_filename = f"{output_dir}/pageviews_{interval_start}_to_{interval_end}.csv"

    processor.export_pageviews_to_csv(output_filename)

    cache_handler.close()

    end_time = time.time()
    logger.info(f"Pageview processing completed in {end_time - start_time:.2f} seconds")
