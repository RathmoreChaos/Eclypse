#!/usr/bin/env python

"""
ga_example.py: A simple example to illustrate how the library works.
               Function optimization (sphere function), 
"""

import random
import time

from eclypse.ind import Individual
from eclypse.problems import FuncOptProblem
from eclypse.coders import Binary2FloatCoder, GrayBinary2FloatCoder
from eclypse.select import TournamentSelection
from eclypse.ops import Clone, BitFlipMutation, UniformCrossover, Evaluate
from eclypse.survive import Elitism


#############################################################################
#
# ga_example2
#
#############################################################################
def ga_example2():
    num_vars=10
    bits_per_float=10
    pop_size = 100
    num_generations = 1000

    bog = None
    bsf = None

    sphere_bounds = [(-5.12, 5.12)] * num_vars
    def sphere_function(phenome):
        return sum([x**2 for x in phenome])

    problem = FuncOptProblem(sphere_function, maximize=False)
    coder = Binary2FloatCoder([bits_per_float] * num_vars, sphere_bounds)
    #coder = GrayBinary2FloatCoder([bits_per_float] * num_vars, sphere_bounds)

    pipeline = TournamentSelection(tournament_size=2)
    pipeline = Clone(pipeline)
    pipeline = UniformCrossover(pipeline, p_cross=1.0, p_swap=0.3)
    pipeline = BitFlipMutation(pipeline, p_mut=1.0 / (num_vars*bits_per_float))
    pipeline = Evaluate(pipeline)
    pipeline = Elitism(pipeline, num_elite=1)

    # Generate initial population
    population = [Individual(problem, coder) for i in range(pop_size)]
    print("\nGeneration", 0)
    bsf = population[0]
    bsf.evaluate()
    for ind in population:
        ind.evaluate()
        if ind.better_than(bsf):
            bsf = ind

    for generation in range(1, num_generations+1):
        pipeline.new_generation(population)
        new_population = [pipeline.pull() for _ in range(pop_size)]

        # Print the population
        print("\nGeneration", generation)
        bog = new_population[0]
        for ind in new_population:
            #print(ind)
            if ind.better_than(bog):
                bog = ind

        if bog.better_than(bsf):
            bsf = bog

        print("bsf =", bsf)

        population = new_population
        #time.sleep(1)

    print()
    print("bsf =", bsf)


#############################################################################
#
# main
#
#############################################################################
if __name__ == "__main__":
    ga_example2()

