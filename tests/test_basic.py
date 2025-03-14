"""
Basic tests that should always pass to ensure CI is working.
"""

import unittest


class TestBasic(unittest.TestCase):
    """Basic tests that should always pass."""

    def test_addition(self):
        """Test that addition works."""
        self.assertEqual(1 + 1, 2)

    def test_string(self):
        """Test string operations."""
        self.assertEqual("hello" + " world", "hello world")


if __name__ == "__main__":
    unittest.main()
