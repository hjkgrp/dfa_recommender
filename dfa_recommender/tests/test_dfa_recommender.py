"""
Unit and regression test for the dfa_recommender package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest
import dfa_recommender


def test_dfa_recommender_imported():
    '''
    Sample test, will always pass so long as import statement worked
    '''
    assert "dfa_recommender" in sys.modules
    
def test_psi4_import():
    '''
    Test whether psi4 can be imported
    '''
    try:
        import psi4
        assert "psi4" in sys.modules
    except ImportError:
        assert 0

def test_torch_import():
    '''
    Test whether torch can be imported
    '''
    try:
        import torch
        assert "torch" in sys.modules
    except ImportError:
        assert 0