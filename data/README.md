# Data Management for Carbon-EF Project

This directory contains all data for the Carbon Emission Factors (Carbon-EF) project. Not all data files are stored in GitHub due to size limitations and best practices.

## Directory Structure

- `raw/`: Original, immutable data
- `interim/`: Intermediate data that has been transformed
- `processed/`: Final, canonical data sets for modeling
- `scripts/`: Code for processing data
- `logs/`: Log files from data processing

## How to Obtain the Data

For data files not included in this repository, please follow these instructions:

### Climate Trace Data

1. Download from: https://climatetrace.org/data
2. Place in `data/raw/climate_trace/`

### IPCC AR6 Data

1. Download PDFs from: https://www.ipcc.ch/report/ar6/wg3/
2. Place in `data/raw/`

### Exiobase Data

1. Register at: https://www.exiobase.eu/
2. Download version 3.8
3. Place in `data/raw/exiobase_3.8/`

### Agribalyse Data

1. Download from: https://www.agribalyse.fr/
2. Place in `data/raw/`

### GREET Model Data

1. Download from: https://greet.es.anl.gov/
2. Place simulated output in `data/raw/greet/`

### USEEIO Data

1. Download from: https://www.epa.gov/land-research/us-environmentally-extended-input-output-useeio-technical-content
2. Place in `data/raw/useeio/`

## Data Processing Workflow

1. Raw data is downloaded and placed in the `raw/` directory
2. Extraction scripts in `scripts/extractors/` convert raw data to a standard format in `interim/`
3. Harmonization scripts in `scripts/harmonization/` combine data into the final dataset in `processed/`

## Data Verification

Each processed file has an accompanying `.md5` checksum file to verify integrity.

## Adding New Data Sources

1. Create a new extractor in `scripts/extractors/`
2. Update `scripts/main.py` to include the new extractor
3. Document the source in this README

## Large File Storage

For collaborators needing the complete dataset:

1. Request access to our shared Google Drive folder
2. Or contact the maintainer for access to our S3 bucket
