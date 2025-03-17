#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation module for Mistral-7B fine-tuning.
This module handles loading, formatting, and preparing the training data
from both JSON files and Neo4j database.
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


def setup_logging(log_level=logging.INFO):
    """Set up logging with handlers that work in any environment"""
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add the console handler
    root_logger.addHandler(console_handler)

    return root_logger


# Run setup_logging at module import to ensure it's configured
setup_logging()


def load_and_prepare_data(
    train_path: str = "training/data/instructions_train.json",
    val_path: str = "training/data/instructions_val.json",
    test_path: str = "training/data/instructions_test.json",
) -> Tuple[DatasetDict, DatasetDict]:
    """
    Load and prepare data for training from JSON files.

    Args:
        train_path: Path to training data JSON file
        val_path: Path to validation data JSON file
        test_path: Path to test data JSON file

    Returns:
        Tuple of (train_dataset, val_dataset) as DatasetDict objects
    """
    logger.info("Loading data for training and validation...")
    logger.info(f"Loading data from JSON files: {train_path}, {val_path}")

    train_data = _load_json_file(train_path)
    val_data = _load_json_file(val_path)

    # Check if data was loaded successfully
    if not train_data or not val_data:
        logger.warning(
            "No data found in files, using minimal example datasets as fallback"
        )
        # Create minimal examples
        minimal_examples = [
            {
                "instruction": "What is the emission factor for cement production in USA?",
                "input": "",
                "output": "The emission factor for cement production in the USA is 0.92 kg CO2e/kg. This data is sourced from USEEIO_v2.1.",
                "metadata": {
                    "regions": ["USA"],
                    "entity_types": ["product"],
                    "difficulty": "basic",
                    "sources": ["USEEIO_v2.1"],
                },
            },
            {
                "instruction": "Compare the emission factor for wheat production between France and the USA.",
                "input": "",
                "output": "The emission factor for wheat production in France is 0.38 kg CO2e/kg, while in the USA it is 0.41 kg CO2e/kg. The USA has a slightly higher emission factor (by 7.9%). This data is sourced from Agribalyse_3.1 for France and USEEIO_v2.1 for the USA.",
                "metadata": {
                    "regions": ["FR", "USA"],
                    "entity_types": ["product"],
                    "difficulty": "moderate",
                    "sources": ["Agribalyse_3.1", "USEEIO_v2.1"],
                },
            },
        ]
        train_data = minimal_examples
        val_data = [minimal_examples[0]]

    # Convert to Dataset objects
    train_dataset = DatasetDict({"train": Dataset.from_list(train_data)})
    val_dataset = DatasetDict({"train": Dataset.from_list(val_data)})

    logger.info(
        f"Loaded {len(train_data)} training examples and {len(val_data)} validation examples"
    )

    return train_dataset, val_dataset


def _load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of instruction examples
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        # Try to load line by line in case it's a JSONL file
        try:
            data = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            if data:
                logger.info(f"Successfully loaded as JSONL: {file_path}")
                return data
        except Exception as e:
            logger.error(f"Failed to load as JSONL: {e}")
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")

    return []


def format_instruction(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Format example as instruction for Mistral-7B.

    Args:
        example: Dictionary containing instruction data

    Returns:
        Dictionary with formatted text
    """
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    # Format using Mistral chat template
    if input_text:
        formatted = f"<s>[INST] {instruction}\n\n{input_text} [/INST] {output} </s>"
    else:
        formatted = f"<s>[INST] {instruction} [/INST] {output} </s>"

    return {"text": formatted}


if __name__ == "__main__":
    # Test data loading and preparation
    train_dataset, val_dataset = load_and_prepare_data()
    logger.info(f"Train dataset size: {len(train_dataset['train'])}")
    logger.info(f"Validation dataset size: {len(val_dataset['train'])}")

    # Format samples
    train_dataset = train_dataset.map(format_instruction)

    # Print a sample
    logger.info("Sample formatted instruction:")
    logger.info(train_dataset["train"][0]["text"][:500] + "...")
