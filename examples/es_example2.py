#!/usr/bin/env python

"""
es_example2.py: A simple example to illustrate how the library works.
"""

import random
import time

from eclypse.ind import Individual
from eclypse.problems import FuncOptProblem
from eclypse.coders import AdaptiveFloatCoder
from eclypse.select import DeterministicSelection
from eclypse.ops import Clone, AdaptiveMutation, Evaluate
from eclypse.survive import MuPlusLambdaSurvival, MuCommaLambdaSurvival


#############################################################################
#
# es_example2
#
#############################################################################
def es_example2():
    genome_size=10

    sphere_ranges = [(-5.12, 5.12)] * genome_size
    def sphere_function(phenome):
        return sum([x**2 for x in phenome])

    init_ranges=sphere_ranges
    sigma_bounds = [(0.0, (r[1]-r[0]) / 6) for r in init_ranges]

    problem = FuncOptProblem(sphere_function, maximize=False)
    #coder = FloatCoder(genome_size, init_ranges=sphere_ranges)
    coder = AdaptiveFloatCoder(init_ranges=init_ranges)

    pipeline = DeterministicSelection()
    pipeline = Clone(pipeline)
    pipeline = mut_op = AdaptiveMutation(pipeline, sigma_bounds)
    pipeline = Evaluate(pipeline)
    pipeline = MuCommaLambdaSurvival(pipeline, num_lambda=20)

    # Generate initial population
    pop_size = 10
    population = [Individual(problem, coder) for _ in range(pop_size)]
    print("\nGeneration", 0)
    bog = population[0]
    bog.evaluate()
    for ind in population:
        ind.evaluate()
        print(ind)
        if ind.better_than(bog):
            bog = ind

    bsf = bog

    print("bog =", bsf)
    print("bsf =", bsf)

    num_generations = 100
    for generation in range(1, num_generations+1):
        pipeline.new_generation(population)
        new_population = [pipeline.pull() for _ in range(pop_size)]

        # Print the population
        print("\nGeneration", generation)
        bog = new_population[0]   # bog : best of generation
        bog.evaluate()
        for ind in new_population:
            print(ind)
            if ind.better_than(bog):
                bog = ind

        if bog.better_than(bsf):
            bsf = bog

        print("bog =", bsf)
        print("bsf =", bsf)

        population = new_population
        #time.sleep(1)

#    print()
#    print("bsf =", bsf)


#############################################################################
#
# command line execute
#
#############################################################################
if __name__ == "__main__":
    es_example2()

