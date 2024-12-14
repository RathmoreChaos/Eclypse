#!/usr/bin/env python

"""
cma_es_example.py: A simple example to illustrate how the library works.
"""

import random
import time

import cma

from eclypse.ind import Individual
from eclypse.problems import FuncOptProblem
from eclypse.coders import FloatCoder
from eclypse.select import DeterministicSelection
from eclypse.ops import Evaluate, CMA_Generate, CMA_Update
from eclypse.survive import MuPlusLambdaSurvival


#############################################################################
#
# cma_es_example
#
#############################################################################
def cma_es_example():
    num_dimensions = 10
    sphere_ranges = [(-5.12, 5.12)] * num_dimensions
    def sphere_function(phenome):
        return sum([x**2 for x in phenome])

    problem = FuncOptProblem(sphere_function, maximize=False)
    coder = FloatCoder(sphere_ranges)

    x0 = [r[0] + (r[1]-r[0]) for r in sphere_ranges]
    sigma0 = sum([r[1]-r[0] for r in sphere_ranges]) / num_dimensions / 3.0
    print("x0 =", x0)
    print("sigma0 =", sigma0)
    cma_es = cma.CMAEvolutionStrategy(x0, sigma0)

    pipeline = CMA_Generate(cma_es, problem, coder)
    pipeline = Evaluate(pipeline)
    pipeline = CMA_Update(pipeline, cma_es)

    population=[]
    popsize = cma_es.popsize

    # Generate initial population
    num_generations = 100
    for generation in range(1, num_generations+1):
        pipeline.new_generation(population)
        new_population = [pipeline.pull() for _ in range(popsize)]

        if generation == 1:
            bsf = new_population[0]

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
    cma_es_example()

