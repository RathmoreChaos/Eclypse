#!/usr/bin/env python

"""
ea.py: defines high level EA classes
"""

import random
import copy

from eclypse.ind import Individual


#############################################################################
#
# GenerationalEA
#
#############################################################################
class GenerationalEA():
    def __init__(self, problem, coder, pipeline, pop_size, max_gen):
        self.problem = problem
        self.coder = coder
        self.pipeline = pipeline
        self.pop_size = pop_size
        self.max_gen = max_gen

    def step(self, prior_generation):
        self.pipeline.new_population(prior_generation)
        new_population = [pipeline.pull() for _ in range(self.pop_size)]
        return new_population

    def run(self):
        # Generate initial population
        population = [Individual(self.problem, self.coder) for _ in range(self.pop_size)]
        for gen in range(1, max_gen+1):
            population = self.step(population)


#############################################################################
#
# SteadyStateEA
#
#############################################################################
class SteadyStateEA():
    def __init__(self, problem, coder, pipeline, pop_size):
        self.problem = problem
        self.coder = coder
        self.pipeline = pipeline
        self.pop_size = pop_size

    def step(self):
        pass

    def run(self):
        pass


#############################################################################
#
# unit_test
#
#############################################################################
if __name__ == "__main__":
    from eclypse.problems import OneMaxProblem
    from eclypse.coders import BinaryCoder

    problem = OneMaxProblem()
    coder = BinaryCoder(5)
    ind1 = Individual(problem, coder, [0,0,0,0,0])
    ind2 = Individual(problem, coder, [1,1,1,1,1])

    ind1.evaluate()
    ind2.evaluate()

    assert(ind2.better_than(ind1))
    assert(ind1.better_than(ind2) == False)
    assert(ind1.equivalent_to(ind2) == False)
    print("passed")


