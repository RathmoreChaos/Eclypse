#!/usr/bin/env python

"""
es_example1.py: A simple example to illustrate how the library works.
"""

import random
import time

from eclypse.ind import Individual
from eclypse.problems import FuncOptProblem
from eclypse.coders import FloatCoder
from eclypse.select import DeterministicSelection
from eclypse.ops import Clone, GaussianMutation, Evaluate
from eclypse.survive import MuPlusLambdaSurvival


#############################################################################
#
# es_example1
#
#############################################################################
def es_example1():
    genome_size=10

    sphere_ranges = [(-5.12, 5.12)] * genome_size
    def sphere_function(phenome):
        return sum([x**2 for x in phenome])

    problem = FuncOptProblem(sphere_function, maximize=False)
    print("sphere_ranges = ", sphere_ranges)
    coder = FloatCoder(sphere_ranges)

    pipeline = DeterministicSelection()
    pipeline = Clone(pipeline)
    pipeline = GaussianMutation(pipeline, sigma=0.5)
    pipeline = Evaluate(pipeline)
    pipeline = MuPlusLambdaSurvival(pipeline, num_lambda=10)

    bsf = 0

    # Generate initial population
    pop_size = 10
    population = [Individual(problem, coder) for _ in range(pop_size)]
    print("\nGeneration", 0)
    bsf = population[0]
    bsf.evaluate()
    for ind in population:
        ind.evaluate()
        print(ind)
        if ind.better_than(bsf):
            bsf = ind

    num_generations = 100
    for generation in range(1, num_generations+1):
        pipeline.new_generation(population)
        new_population = [pipeline.pull() for _ in range(pop_size)]

        # Print the population
        print("\nGeneration", generation)
        for ind in new_population:
            print(ind)
            if ind.better_than(bsf):
                bsf = ind

        population = new_population
        #time.sleep(1)

    print("bsf =", bsf)


#############################################################################
#
# command line execute
#
#############################################################################
if __name__ == "__main__":
    es_example1()

