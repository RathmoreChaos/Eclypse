#!/usr/bin/env python

"""
test_problems.py: tests common problems for Eclypse
"""

from eclypse.problems import SimilarityProblem
from eclypse.coders import BinaryCoder
from eclypse.ind import Individual


def test_SimilarityProblem():
    problem = SimilarityProblem([1,1,1,1,1])
    coder = BinaryCoder(5)
    ind1 = Individual(problem, coder, [0,0,0,0,0])
    ind2 = Individual(problem, coder, [1,1,1,1,1])

    ind1.evaluate()
    ind2.evaluate()

    assert(ind2.better_than(ind1))
    assert(ind1.better_than(ind2) == False)
    assert(ind1.equivalent_to(ind2) == False)
    print("passed")


