"""
Configuration module for the Wikipedia Bias Graph Pipeline.

This module loads configuration from environment variables with sensible defaults.
Users can customize settings by creating a .env file in the project root.
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default values as constants
_DEFAULT_SPOTLIGHT_PORTS = "it:2223,en:2221,es:2225,fr:2224,de:2222"
_DEFAULT_LANGUAGES = "it,en,es,fr,de"


def _parse_spotlight_ports() -> Dict[str, str]:
    """
    Parse spotlight ports from environment variable.
    Format: SPOTLIGHT_PORTS="it:2223,en:2221,es:2225,fr:2224,de:2222"
    Returns a dict mapping language codes to full endpoint URLs.
    """
    ports_str = os.getenv("SPOTLIGHT_PORTS", _DEFAULT_SPOTLIGHT_PORTS)
    base_url = os.getenv("SPOTLIGHT_BASE_URL", "http://localhost")
    endpoint_path = os.getenv("SPOTLIGHT_ENDPOINT_PATH", "/rest/annotate")

    endpoints = {}
    for pair in ports_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            lang, port = pair.split(":", 1)
            lang = lang.strip()
            port = port.strip()
            endpoints[lang] = f"{base_url}:{port}{endpoint_path}"

    return endpoints


def _parse_languages() -> List[str]:
    """
    Parse default languages from environment variable.
    Format: DEFAULT_LANGUAGES="it,en,es,fr,de"
    """
    langs_str = os.getenv("DEFAULT_LANGUAGES", _DEFAULT_LANGUAGES)
    return [lang.strip() for lang in langs_str.split(",") if lang.strip()]


# Wikipedia API Configuration
WIKIPEDIA_USER_AGENT: str = os.getenv(
    "WIKIPEDIA_USER_AGENT",
    "WikipediaBiasProject/1.0 (contact: your@email.com)",
)

WIKIPEDIA_ACCESS_TOKEN: Optional[str] = os.getenv("WIKIPEDIA_ACCESS_TOKEN")

WIKIPEDIA_REQUESTS_PER_SECOND: int = int(
    os.getenv("WIKIPEDIA_REQUESTS_PER_SECOND", "180")
)

WIKIPEDIA_TIMEOUT: int = int(os.getenv("WIKIPEDIA_TIMEOUT", "10"))
WIKIPEDIA_MAX_RETRIES: int = int(os.getenv("WIKIPEDIA_MAX_RETRIES", "3"))

# DBpedia Spotlight Configuration
SPOTLIGHT_ENDPOINTS: Dict[str, str] = _parse_spotlight_ports()

SPOTLIGHT_MIN_CONFIDENCE: float = float(os.getenv("SPOTLIGHT_MIN_CONFIDENCE", "0.8"))

SPOTLIGHT_TIMEOUT: int = int(os.getenv("SPOTLIGHT_TIMEOUT", "10"))
SPOTLIGHT_MAX_RETRIES: int = int(os.getenv("SPOTLIGHT_MAX_RETRIES", "3"))
SPOTLIGHT_RATE_LIMIT_DELAY: float = float(
    os.getenv("SPOTLIGHT_RATE_LIMIT_DELAY", "0.01")
)

# Default Languages (used for graph building and weighting)
DEFAULT_LANGUAGES: List[str] = _parse_languages()

# Database Configuration
DUCKDB_THREADS: int = int(os.getenv("DUCKDB_THREADS", "32"))
DUCKDB_MEMORY_LIMIT: str = os.getenv("DUCKDB_MEMORY_LIMIT", "32GB")
DUCKDB_MAX_TEMP_DIR_SIZE: str = os.getenv("DUCKDB_MAX_TEMP_DIR_SIZE", "200GB")
DUCKDB_TEMP_DIR: str = os.getenv("DUCKDB_TEMP_DIR", "/tmp/duckdb_temp")

# Processing Configuration
CHECKPOINT_DIR: str = os.getenv("CHECKPOINT_DIR", "data/checkpoints")
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "1000"))
MEMORY_WARNING_THRESHOLD_GB: float = float(
    os.getenv("MEMORY_WARNING_THRESHOLD_GB", "50.0")
)


def get_spotlight_ports_for_compose() -> Dict[str, int]:
    """
    Get spotlight ports as a dict of language to port number.
    Useful for generating docker-compose files.
    """
    ports_str = os.getenv("SPOTLIGHT_PORTS", _DEFAULT_SPOTLIGHT_PORTS)

    ports = {}
    for pair in ports_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            lang, port = pair.split(":", 1)
            lang = lang.strip()
            port = port.strip()
            try:
                ports[lang] = int(port)
            except ValueError:
                pass

    return ports
