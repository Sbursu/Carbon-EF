#!/usr/bin/env python3
"""
Test runner for the Carbon EF system.
"""

import logging
import os
import sys

import pytest

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set environment variables for testing
os.environ["TEST_MODE"] = "1"


def main():
    """Main function to run tests."""
    print("\n=== Carbon EF Test Runner ===\n")

    # Get test path from command line arguments or use default
    test_path = sys.argv[1] if len(sys.argv) > 1 else "src"

    print(f"Running tests from: {test_path}\n")

    # Run tests with pytest
    exit_code = pytest.main(
        [
            "-xvs",  # Exit on first failure, verbose output, no capture
            "--ignore=models",  # Ignore model directories
            "--ignore=data",  # Ignore data directories
            test_path,
        ]
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
