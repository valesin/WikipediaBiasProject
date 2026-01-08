#!/usr/bin/env python3
"""
Generate docker-compose file for DBpedia Spotlight services based on configuration.

This script reads the SPOTLIGHT_PORTS environment variable and generates
a docker-compose.yml file with the appropriate port mappings.

Usage:
    python scripts/generate_spotlight_compose.py > container/spotlight-compose.yml
"""

import os
import sys

# Add src to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.config import get_spotlight_ports_for_compose, DEFAULT_LANGUAGES


def generate_compose() -> str:
    """Generate docker-compose YAML content."""
    ports = get_spotlight_ports_for_compose()
    
    # Filter to only configured languages
    configured_langs = [lang for lang in DEFAULT_LANGUAGES if lang in ports]
    
    lines = ["services:"]
    
    for lang in configured_langs:
        port = ports[lang]
        lines.extend([
            f"  spotlight.{lang}:",
            f"    image: dbpedia/dbpedia-spotlight",
            f"    container_name: dbpedia-spotlight.{lang}",
            f"    volumes:",
            f"       - spotlight-model:/opt/spotlight/models",
            f"    restart: unless-stopped",
            f"    ports:",
            f'       - "0.0.0.0:{port}:80"',
            f"    command: /bin/spotlight.sh {lang}",
            "",
        ])
    
    lines.extend([
        "volumes:",
        "  spotlight-model:",
        "    external: true",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_compose())
