"""
Wikipedia Bias Graph Pipeline - Main Entry Point

This script provides a unified interface for building and weighting Wikipedia bias graphs.
It can perform three main operations:
1. Build an unweighted graph from Wikidata IDs
2. Assign weights to an existing graph using DBpedia Spotlight
3. Filter edges based on pageview thresholds
"""

import os
import time
import logging
import argparse
import pandas as pd
from datetime import datetime, timedelta
from modules.cached_graph_builder import (
    UnweightedGraphBuilder,
    GraphWeighter,
    create_handlers,
)
from modules.spotlight_weight_source import SpotlightWeightSource
from modules.config import DEFAULT_LANGUAGES
from modules.pageview_processor import process_pageviews


def setup_logging(operation: str, log_level: str = "INFO") -> None:
    """Configure logging based on the operation being performed.

    Args:
        operation: Name of the operation (for log filename)
        log_level: Logging level (case-insensitive: CRITICAL, ERROR, WARNING, INFO, DEBUG)
    """
    level = getattr(logging, log_level.upper(), logging.WARNING)
    log_filename = f"logs/{operation}_logfile_{time.strftime('%Y%m%d_%H%M%S')}.log"

    os.makedirs("logs", exist_ok=True)

    # Remove any existing handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure file handler
    file_handler = logging.FileHandler(log_filename, mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    # Configure root logger
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)

    # Log startup message to confirm logging is working
    logging.info(f"Starting {operation} operation with log level {log_level}")


def build_graph(args: argparse.Namespace) -> None:
    """Build an unweighted graph from Wikidata IDs."""
    setup_logging("builder", args.log_level)

    input_path = os.path.join(os.getcwd(), args.input)
    output_path = os.path.join(os.getcwd(), args.output)

    # Read Wikidata IDs from CSV
    df = pd.read_csv(input_path, encoding="utf-8")  # type: ignore
    notable_ids_list = df["wikidata_code"].dropna().astype(str).tolist()
    if args.limit:
        notable_ids_list = notable_ids_list[: args.limit]

    logging.info(f"Processing {len(notable_ids_list)} notable IDs")

    # Create handlers
    cache_handler, data_source, _ = create_handlers(cache_filename=output_path)

    builder = UnweightedGraphBuilder(
        notable_ids_list=notable_ids_list,
        languages=args.languages,
        cache_handler=cache_handler,
        data_source=data_source,
    )

    start_time = time.time()
    logging.info(
        "Started graph building at: %s",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
    )
    logging.info("Building unweighted graph...")
    builder.build_unweighted_graph(fetch_extracts=True, fetch_links=True)

    end_time = time.time()
    logging.info(
        "Unweighted graph built successfully in %.2f seconds", end_time - start_time
    )


def weight_graph(args: argparse.Namespace) -> None:
    """Assign weights to an existing graph using DBpedia Spotlight."""
    setup_logging("weighter", args.log_level)

    graph_db_path = os.path.join(os.getcwd(), args.graph_db)

    # Create handlers
    cache_handler, _, weight_assigner = create_handlers(cache_filename=graph_db_path)

    # Refresh notable links table if requested
    if args.refresh_notable_links:
        cache_handler.refresh_notable_links(force=True)

    weighter = GraphWeighter(
        languages=args.languages,
        cache_handler=cache_handler,
        weight_assigner=weight_assigner,
        checkpoint_file=args.checkpoint_file,
        output_dir=args.output_dir,
    )

    start_time = time.time()
    logging.info(
        "Started weight assignment at: %s",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
    )
    logging.info("Assigning weights to graph...")
    weighter.weight_graph()

    end_time = time.time()
    logging.info("Weights assigned successfully in %.2f seconds", end_time - start_time)


def export_pageviews(args: argparse.Namespace) -> None:
    """Export pageview data to CSV."""
    setup_logging("pageviews", args.log_level)

    graph_db_path = os.path.join(os.getcwd(), args.graph_db)

    # Set default dates if not provided
    end_date = args.end_date if args.end_date else datetime.now().strftime("%Y-%m-%d")
    start_date = (
        args.start_date
        if args.start_date
        else (datetime.now() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
    )

    # Calculate days in interval for context
    days_diff = (
        datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)
    ).days + 1
    logging.info(f"Using date range: {start_date} to {end_date} ({days_diff} days)")

    # Process and export pageviews
    process_pageviews(
        graph_db_path=graph_db_path,
        interval_start=start_date,
        interval_end=end_date,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


def main():
    """Main entry point for the Wikipedia bias graph pipeline."""
    parser = argparse.ArgumentParser(
        description="Wikipedia Bias Graph Pipeline - Build, weight, and filter graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build a graph from Wikidata IDs
  python main.py build --input entities.csv --output graph.db --languages en de fr --log-level INFO

  # Assign weights to an existing graph
  python main.py weight --graph-db graph.db --output-dir data/out --languages en de fr --log-level DEBUG

  # Export pageview data to CSV
  python main.py pageviews --graph-db graph.db --output-dir data/out --start-date 2023-01-01 --end-date 2025-12-14

  # Full pipeline
  python main.py build --input entities.csv --output graph.db --languages en
  python main.py weight --graph-db graph.db --output-dir data/out --languages en
  python main.py pageviews --graph-db graph.db --output-dir data/out --start-date 2023-01-01 --end-date 2025-12-14
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Operation to perform")

    # Build subcommand
    build_parser = subparsers.add_parser(
        "build", help="Build an unweighted graph from Wikidata IDs"
    )
    build_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file with 'wikidata_code' column (relative to current directory)",
    )
    build_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output database filename (relative to current directory)",
    )
    build_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of IDs to process (for testing)",
    )
    build_parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="Language codes to process (default: en)",
    )
    build_parser.add_argument(
        "--log-level",
        type=str.upper,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Set logging level (case-insensitive, default: INFO)",
    )

    # Weight subcommand
    weight_parser = subparsers.add_parser(
        "weight", help="Assign weights to an existing graph"
    )
    weight_parser.add_argument(
        "--graph-db",
        type=str,
        required=True,
        help="Input graph database filename (relative to current directory)",
    )
    weight_parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for CSV file (default: data)",
    )
    weight_parser.add_argument(
        "--languages",
        nargs="+",
        default=DEFAULT_LANGUAGES,
        help=f"Language codes to process (default: {' '.join(DEFAULT_LANGUAGES)})",
    )
    weight_parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Checkpoint file path for resumable processing (optional)",
    )
    weight_parser.add_argument(
        "--refresh-notable-links",
        action="store_true",
        help="Refresh the notable flags in page_link table before weight assignment",
    )
    weight_parser.add_argument(
        "--log-level",
        type=str.upper,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Set logging level (case-insensitive, default: INFO)",
    )

    # Pageviews subcommand
    pageviews_parser = subparsers.add_parser(
        "pageviews", help="Export pageview data to CSV"
    )
    pageviews_parser.add_argument(
        "--graph-db",
        type=str,
        required=True,
        help="Input graph database filename (relative to current directory)",
    )
    pageviews_parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for CSV file (default: data)",
    )
    pageviews_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for pageview interval in ISO format (YYYY-MM-DD). Default: 3 years ago",
    )
    pageviews_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for pageview interval in ISO format (YYYY-MM-DD). Default: today",
    )

    pageviews_parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of pages to process per API batch (default: 50)",
    )
    pageviews_parser.add_argument(
        "--log-level",
        type=str.upper,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Set logging level (case-insensitive, default: INFO)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "build":
        build_graph(args)
    elif args.command == "weight":
        weight_graph(args)
    elif args.command == "pageviews":
        export_pageviews(args)


if __name__ == "__main__":
    main()
