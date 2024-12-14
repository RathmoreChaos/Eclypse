#!/usr/bin/env python

"""
ea.py: defines high level EA classes
"""

from eclypse.ind import Individual


#class GenerationalEA():
#class SteadyStateEA():


def test_GenerationalEA():
    from eclypse.problems import SimilarityProblem
    from eclypse.coders import BinaryCoder

    problem = SimilarityProblem([1,1,1,1,1])  # OneMax
    coder = BinaryCoder(5)
    ind1 = Individual(problem, coder, [0,0,0,0,0])
    ind2 = Individual(problem, coder, [1,1,1,1,1])

    ind1.evaluate()
    ind2.evaluate()

    assert(ind2.better_than(ind1))
    assert(ind1.better_than(ind2) == False)
    assert(ind1.equivalent_to(ind2) == False)
    print("passed")


