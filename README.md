# Wikipedia Bias Graph Pipeline - Usage Guide

Two main sections:
1. **Graph Pipeline**: Building the graph, assigning weights, exporting pageviews
2. **Bias Analysis**: File structure, filtering, analysis notebooks

---

# Part 1: Graph Pipeline

## Overview

Three CLI commands:
1. **build** - Build unweighted graph from Wikidata IDs
2. **weight** - Assign weights using DBpedia Spotlight
3. **pageviews** - Export Wikipedia pageview data to CSV

## Data Schema Conventions

Wikidata identifier column naming:

- **`wikidata_code`**: Input column from cross-verified database
- **`wikidata_id`**: Output column after filtering (metadata files)
- **`source_wikidata_id`, `target_wikidata_id`**: Edge data columns
- **Format**: Always includes Q prefix (e.g., `Q42`, `Q5`)
  - Note: DuckDB stores as integers internally, returns with Q prefix

**Pipeline column transformations**:
1. Cross-verified database: `wikidata_code` column
2. `python src/main.py build`: Reads `wikidata_code` from input CSV
3. `filter_by_languages.ipynb`: Renames `wikidata_code` → `wikidata_id` in `entities.csv`
4. `python src/main.py weight`: Outputs `source_wikidata_id`, `target_wikidata_id` in edges CSV
5. Analysis notebooks: Use `wikidata_id` in metadata, `source_wikidata_id`/`target_wikidata_id` in edges

## Prerequisites

**System Requirements:**
- Python 3.10+
- 16GB RAM (32GB recommended for large datasets)
- ~50GB disk space
- Docker

**External Services:**
- DBpedia Spotlight instances (must be configured and running before weight assignment)

**Data:**
- Cross-verified notable entities database (see Data Sources section)

## Installation

Create and activate a virtual environment:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r src/requirements.txt
```

## Configuration

The pipeline uses environment variables for configuration. Copy `.env.example` to `.env` and customize as needed:

```bash
cp .env.example .env
# Edit .env with your settings
```

### Key Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `SPOTLIGHT_PORTS` | Language:port mapping for Spotlight instances | `it:2223,en:2221,es:2225,fr:2224,de:2222` |
| `SPOTLIGHT_MIN_CONFIDENCE` | Minimum confidence for entity recognition | `0.8` |
| `DEFAULT_LANGUAGES` | Default languages for processing | `it,en,es,fr,de` |
| `WIKIPEDIA_USER_AGENT` | User agent for Wikipedia API requests | `WikipediaBiasProject/1.0 (...)` |
| `WIKIPEDIA_ACCESS_TOKEN` | Optional API token for higher rate limits | (empty) |

### DBpedia Spotlight Setup

Weight assignment requires local DBpedia Spotlight instances:

1. Configure ports in `.env`:
   ```
   SPOTLIGHT_PORTS=it:2223,en:2221,es:2225,fr:2224,de:2222
   ```

2. Generate docker-compose file (optional):
   ```bash
   python scripts/generate_spotlight_compose.py > container/spotlight-compose.yml
   ```

3. Start Spotlight services:
   ```bash
   cd container
   docker-compose -f spotlight-compose.yml up -d
   ```

## CLI Commands

### Global Arguments

All commands support:
- `--log-level` (optional): Set logging verbosity (default: `INFO`)
  - Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

### 1. Building a Graph

Build an unweighted graph from a CSV file containing Wikidata IDs:

```bash
python src/main.py build \
    --input data/entities.csv \
    --output data/out/graph.db \
    --languages en de fr it es
```

**Arguments:**
- `--input` (required): Path to CSV with `wikidata_code` column
- `--output` (required): Output database filename
- `--languages` (optional): Language codes (default: `en`)
- `--limit` (optional): Limit entities for testing

### 2. Weighting a Graph

Assign weights to an existing graph using DBpedia Spotlight:

```bash
python src/main.py weight \
    --graph-db data/out/graph_try.db \
    --output-dir data/out \
    --languages en de fr it es \
    --refresh-notable-links
```

**Arguments:**
- `--graph-db` (required): Path to existing graph database
- `--output-dir` (optional): Output directory for CSV file (default: `data`)
- `--languages` (optional): Language codes to process (default: `en de it fr es`)
- `--checkpoint-file` (optional): Path to checkpoint file for resumable processing
- `--refresh-notable-links` (optional): Force refresh of notable_page_link table

**Output:**
- CSV file: `data/out/SpotlightWeightSource_MMDD_HHMM.csv`
- Columns: `source_wikidata_id`, `target_wikidata_id`, `weight`, `language_code`

**Note:** A CSV file with all weighted links will always be generated, even if there are no new weights to assign.

### 3. Exporting Pageview Data

Export Wikipedia pageview statistics to CSV:

```bash
python src/main.py pageviews \
    --graph-db data/out/graph_try.db \
    --output-dir data/out \
    --start-date 2023-01-01 \
    --end-date 2025-12-14
```

**Arguments:**
- `--graph-db` (required): Path to existing graph database
- `--output-dir` (optional): Output directory for CSV file (default: `data`)
- `--start-date` (optional): Start date in ISO format (YYYY-MM-DD). Default: 3 years ago
- `--end-date` (optional): End date in ISO format (YYYY-MM-DD). Default: today
- `--batch-size` (optional): Number of pages per DB commit (default: `50`)

**Output:**
- CSV file: `pageviews_{start_date}_to_{end_date}.csv`
- Columns: `wikidata_id`, `language_code`, `pageviews` (daily average)

**Implementation:**
1. Fetches pageview data from Wikimedia API for specified interval
2. Calculates daily average (total views ÷ number of days)
3. Exports data for all notable entities in graph
4. Parallel threading: up to 180 concurrent requests/second (default rate limit)

**Examples:**

```bash
# Export with default dates (last 3 years)
python src/main.py pageviews --graph-db data/out/graph.db --output-dir data/out

# Export with custom date range
python src/main.py pageviews \
    --graph-db data/out/graph.db \
    --output-dir data/out \
    --start-date 2023-01-01 \
    --end-date 2025-12-14
```

## Complete Pipeline Example

```bash
# Activate virtual environment
source .venv/bin/activate

# Step 1: Build the graph
python src/main.py build \
    --input data/entities.csv \
    --output data/out/graph.db \
    --languages en de fr it es

# Step 2: Assign weights
python src/main.py weight \
    --graph-db data/out/graph.db \
    --output-dir data/out \
    --languages en de fr it es \
    --checkpoint-file data/out/weight_checkpoint.json \
    --refresh-notable-links

# Step 3: Export pageview data
python src/main.py pageviews \
    --graph-db data/out/graph.db \
    --output-dir data/out \
    --start-date 2023-01-01 \
    --end-date 2025-12-14

# Step 4: Proceed to analysis (see Part 2 below)
```

## Resuming Interrupted Operations

The weight operation supports checkpointing for long-running processes:

```bash
# If interrupted, resume with same checkpoint file
python src/main.py weight \
    --graph-db data/out/graph.db \
    --output-dir data/out \
    --checkpoint-file data/out/weight_checkpoint.json
```

## Logs

Logs are saved to the `logs/` directory:
- Build: `logs/builder_logfile_YYYYMMDD_HHMMSS.log`
- Weight: `logs/weighter_logfile_YYYYMMDD_HHMMSS.log`
- Pageviews: `logs/pageviews_logfile_YYYYMMDD_HHMMSS.log`

---

# Statistical Methodology

## Noise-Corrected Backbone Extraction

Uses **Noise-Corrected (NC)** (*Coscia & Neffke “Network Backboning with Noisy Data”, ICDE 2017*) backbone extraction to identify statistically significant links.

Method:
1. Calculates expected link weights under null model
2. Compares observed weights against expectations
3. Computes p-values for statistical significance
4. Retains edges based on significance thresholds

Accounts for degree heterogeneity.

## Weight Calculation

Edge weights represent **the number of entity mentions** detected by DBpedia Spotlight:

- **Source**: The Wikipedia article being analyzed
- **Target**: An entity mentioned within that article's text
- **Weight**: Count of mentions above confidence threshold (≥ 0.8)

Higher weights = more frequent co-mentions.

---

# Part 2: Bias Analysis

## Overview

After building graph and exporting data, analyze retention bias using analysis module and Jupyter notebooks.

## Data Sources

### Cross-Verified Database
The initial filtering step requires the cross-verified notable entities database:

- **File**: `data/cross-verified-database.csv.gz`
- **Source**: [BHHT Datascape](https://medialab.github.io/bhht-datascape/)
- **Citation**: [TODO - Add full citation]
- **License**: [TODO - Add license information]

Database contains:
- Wikidata IDs for notable figures
- Demographic attributes (gender, birth/death dates, geographic data)
- Wikipedia edition coverage

## Pipeline Overview

The analysis pipeline consists of four main stages:

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Entity Filtering                                  │
│  notebooks/filter_by_languages.ipynb                        │
│                                                              │
│  Input:  data/cross-verified-database.csv.gz (wikidata_code)│
│  Output: data/entities.csv (wikidata_id)                    │
│  Purpose: Filter entities with articles in all target langs │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Graph Building                                    │
│  python src/main.py build                                   │
│                                                              │
│  Input:  CSV file with wikidata_code column                 │
│  Output: data/out/graph.db (DuckDB database)                │
│  Purpose: Fetch Wikipedia links for all entities            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Weight Assignment                                 │
│  python src/main.py weight                                  │
│                                                              │
│  Input:  data/out/graph.db                                  │
│  Output: data/out/SpotlightWeightSource_MMDD_HHMM.csv       │
│  Purpose: Assign weights via DBpedia Spotlight entity recog │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Bias Analysis                                     │
│  notebooks/bias_analysis_full_vs_restricted.ipynb           │
│                                                              │
│  Input:  - SpotlightWeightSource_*.csv (edges)              │
│          - data/entities.csv (metadata)                     │
│  Output: - Retention curves                                 │
│          - AUC heatmaps                                     │
│  Purpose: Analyze retention bias across demographics        │
└─────────────────────────────────────────────────────────────┘
```

**Intermediate Files**:
- `data/out/graph.db`: DuckDB database containing graph structure and metadata
- `data/checkpoints/*.json`: Resumable processing checkpoints for weight assignment
- `data/out/SpotlightWeightSource_*.csv`: Timestamped weighted edge files

## File Structure

The analysis requires these input files:

### 1. Weighted Edges CSV
- **Location**: `data/out/SpotlightWeightSource_MMDD_HHMM.csv`
- **Generated by**: `python src/main.py weight`
- **Columns**:
  - `source_wikidata_id`: Source node Wikidata ID
  - `target_wikidata_id`: Target node Wikidata ID
  - `weight`: Edge weight from DBpedia Spotlight
  - `language_code`: Wikipedia language

### 2. Metadata CSV
- **Location**: `data/entities.csv`
- **Required columns**:
  - `wikidata_id`: Wikidata ID
  - `gender`: Gender attribute
  - `birth`: Birth year (numeric)
  - `un_region`: UN region

### 3. Pageviews CSV (Optional)
- **Location**: `data/out/pageviews_YYYY-MM-DD_to_YYYY-MM-DD.csv`
- **Generated by**: `python src/main.py pageviews`
- **Columns**:
  - `wikidata_id`: Wikidata ID
  - `language_code`: Wikipedia language
  - `pageviews`: Daily average pageviews

## Filters Module

`src/filters.py` contains reusable edge filter functions.

### Available Filters

```python
from src import filters

# Year-based filters
filter_1800s = filters.restrict_years(1800, 1900)
filter_modern = filters.modern_period_only  # Birth > 1500
filter_historic = filters.historic_period_only  # Birth <= 1500

# Region-based filters
filter_western = filters.western_regions_only  # Europe, America, Oceania

# Gender-based filters
filter_same_gender = filters.same_gender_only
filter_cross_gender = filters.cross_gender_only

# Pageview-based filters (requires pageviews CSV)
import pandas as pd
pageviews = pd.read_csv("data/out/pageviews_2023-01-01_to_2025-12-14.csv")

# Language-specific: Top 20% per language (different threshold for each language)
filter_top_20_lang = filters.restrict_by_pageviews_quantile(pageviews, 0.8)

# Language-specific: Pages with >= 1000 daily views in that language
filter_1000_lang = filters.restrict_by_pageviews_absolute(pageviews, 1000)

# Global/multilayer: Top 20% by aggregated pageviews (same nodes across all languages)
filter_top_20_global = filters.restrict_by_pageviews_quantile_global(pageviews, 0.8)

# Global/multilayer: Pages with >= 5000 aggregated daily views
filter_5000_global = filters.restrict_by_pageviews_absolute_global(pageviews, 5000)
```

### Creating Custom Filters

Filters are functions that take a DataFrame row and return `True` to keep it:

```python
def my_custom_filter(row):
    # Access enriched columns
    birth_src = row.get("birth_source")
    birth_trg = row.get("birth_target")
    
    # Apply logic
    if pd.isna(birth_src) or pd.isna(birth_trg):
        return False
    
    return (1750 <= birth_src <= 1950) and (1750 <= birth_trg <= 1950)
```

## Analysis Module

`src/analysis.py` contains the main analysis pipeline.

### Basic Usage

```python
from src.analysis import run_bias_analysis
import pandas as pd

# Load data
edges_df = pd.read_csv("data/out/SpotlightWeightSource_1216_1948.csv")
meta_df = pd.read_csv("data/entities.csv")

# Run analysis
results = run_bias_analysis(
    edges_df=edges_df,
    meta_df=meta_df,
    selected_languages=["en", "de", "fr", "it", "es"],
    min_edges=100,
    pre_transform_filters=None,  # Optional: list of filter functions for raw data
    post_transform_filters=None  # Optional: list of filter functions for transformed data
)
```

### With Filters

```python
from src import filters

# Load pageviews
pageviews = pd.read_csv("data/out/pageviews_2023-01-01_to_2025-12-14.csv")

# Create filters
filter_top_20 = filters.restrict_by_pageviews_quantile(pageviews, 0.8)
filter_years = filters.restrict_years(1750, 1950)

# Run analysis with filters
# Filters on raw data (years, pageviews) go in pre_transform_filters
results = run_bias_analysis(
    edges_df=edges_df,
    meta_df=meta_df,
    selected_languages=["en", "de", "fr", "it", "es"],
    min_edges=100,
    pre_transform_filters=[filter_top_20, filter_years]  # Filters applied sequentially (AND logic)
)
```

### Return Value Structure

```python
{
    'en': {
        'gender': (edge_results_list, node_results_dict, auc_matrix),
        'birth': (edge_results_list, node_results_dict, auc_matrix),
        'un_region': (edge_results_list, node_results_dict, auc_matrix)
    },
    'de': { ... },
    'fr': { ... }
}
```

## Analysis Notebooks

### Year-Restricted Analysis: `bias_analysis_full_vs_restricted.ipynb`

**Location**: `notebooks/bias_analysis_full_vs_restricted.ipynb`

Compares bias patterns in full dataset vs time-restricted subset (1750-1950).

**Workflow**:
1. Load edges and metadata
2. Run full dataset analysis
3. Run time-restricted analysis (1750-1950)
4. Display and save results

**Features**:
- Parameter-aware caching with validation metadata
- Cache directories include resolution, min_edges, language count, attribute count
- Formatted AUC matrices and edge pair summaries
- Retention curve data export
- Side-by-side comparison

**Caching Limitation**: The caching system tracks all parameters except custom transformation functions. If you modify transformations while keeping other parameters the same, manually move previous results to a different directory to avoid cache conflicts.

### Pageview-Filtered Analysis: `bias_analysis_pageview_filtered.ipynb`

**Location**: `notebooks/bias_analysis_pageview_filtered.ipynb`

Analyzes bias patterns for high-pageview entities.

**Two Filtering Modes**:
- **Language-specific**: Different pageview thresholds per Wikipedia edition
- **Global/multilayer**: Single threshold based on aggregated pageviews (same nodes across all languages)

**Workflow**:
1. **Exploration**: Compare quantile thresholds (0.5, 0.7, 0.8, 0.9)
   - Shows entity and edge counts per threshold
   - No full analysis - statistics only
2. **Analysis**: Select threshold and run complete bias analysis
   - Toggle between language-specific and global via `USE_GLOBAL_PAGEVIEWS`
   - Results cached

**Key Features**:
- Exact edge counting in exploration
- Easy threshold modification
- Supports quantile and absolute thresholds
- Parameter-inclusive directory structure: `plots/{dataset}_{filter}_{params}/`
- Cache validation via metadata files

**Output Example**:
```
Quantile 0.80 (top 20%)
--------------------------------------------------
  Threshold value: 1247.89 daily views
  Entities retained: 18,880 / 94,403 (20.0%)
  Edges retained: 2,488,836 / 12,444,180 (20.0%)
```

## Analysis Workflow Example

```python
# 1. Import modules
from src.analysis import run_bias_analysis
from src import filters
import pandas as pd

# 2. Load data
edges = pd.read_csv("data/out/SpotlightWeightSource_1216_1948.csv")
meta = pd.read_csv("data/entities.csv")
pageviews = pd.read_csv("data/out/pageviews_2023-01-01_to_2025-12-14.csv")

# 3. Create filters (optional)
filter_top_20 = filters.restrict_by_pageviews_quantile(pageviews, 0.8)
filter_years = filters.restrict_years(1750, 1950)

# 4. Run analysis
results = run_bias_analysis(
    edges_df=edges,
    meta_df=meta,
    selected_languages=["en", "de", "fr", "it", "es"],
    min_edges=100,
    pre_transform_filters=[filter_top_20, filter_years]  # Applied to raw data
)

# 5. Visualize results (in notebook)
from src.analysis import plot_retention_results

plot_retention_results(
    results,
    languages=["en"],
    attributes=["gender", "birth", "un_region"],
    min_edges=100,
    plot_type="both"  # curves + heatmaps
)
```