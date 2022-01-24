"""
Unit and regression test for the dfa_recommender package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import dfa_recommender


def test_dfa_recommender_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "dfa_recommender" in sys.modules
