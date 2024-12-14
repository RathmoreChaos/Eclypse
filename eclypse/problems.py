#!/usr/bin/env python

"""
problems.py: defines common problems for Eclypse
"""

import random
import copy


#############################################################################
#
# BaseProblem
#
#############################################################################
class BaseProblem():
    def evaluate(self, phenome):
        raise NotImplementedError

    def better_than(self, fit1, fit2):
        raise NotImplementedError

    def equivalent_to(self, fit1, fit2):
        raise NotImplementedError


#############################################################################
#
# SimilarityProblem
#
# This is essentially a generalization of the one-max problem.  The goal is to
# find a binary string that matches and existing target string.  If the target
# string consists of all ones, then it is the one-max problem.
#
#############################################################################
class SimilarityProblem(BaseProblem):
    def __init__(self, target):
        self.target = target

    def evaluate(self, phenome):
        return sum([p == t for p,t in zip(phenome, self.target)])

    def better_than(self, fit1, fit2):
        return fit1 > fit2

    def equivalent_to(self, fit1, fit2):
        return fit1 == fit2


#############################################################################
#
# FuncOptProblem
#
#############################################################################
class FuncOptProblem(BaseProblem):
    "A general purpose class for function optimization problems."
    def __init__(self, function, maximize=True):
        self.function = function
        self.maximize = maximize

    def evaluate(self, phenome):
        fitness = self.function(phenome)
        return fitness

    def better_than(self, fit1, fit2):
        if self.maximize:
            return fit1 > fit2
        else:
            return fit1 < fit2

    def equivalent_to(self, fit1, fit2):
        return fit1 == fit2


#############################################################################
#
# unit_test
#
#############################################################################
if __name__ == "__main__":
    from eclypse.coders import BinaryCoder
    from eclypse.ind import Individual

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


