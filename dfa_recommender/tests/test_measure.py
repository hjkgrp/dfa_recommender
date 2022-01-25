'''
test suites for measurement functions
'''

import dfa_recommender
import pytest


def test_euclidean_distance():

    r1 = [0, 0, 0]
    r2 = [1, 0, 0]
    
    ref_dist = 1
    calc_dist = dfa_recommender.euclidean_dist(r1, r2)
    assert ref_dist == calc_dist

@pytest.mark.parametrize("x", [0, 1])
@pytest.mark.parametrize("y", [2, 3])
def test_foo(x, y):
    pass
