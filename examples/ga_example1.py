#!/usr/bin/env python

"""
ga_example1.py: A simple example to illustrate how the library works.
                MaxOnes problem, tournament selection,
                uniform crossover (50%/bit), bit flip mutation (1/l),
                elitism (1), 20 generations
"""

import random
import sys

from eclypse.ind import Individual
from eclypse.problems import SimilarityProblem
from eclypse.coders import BinaryCoder
from eclypse.select import TournamentSelection
from eclypse.ops import Clone, BitFlipMutation, UniformCrossover, Evaluate
from eclypse.survive import Elitism


#############################################################################
#
# ga_example1
#
#############################################################################
def ga_example1():
    genome_size=10

    problem = SimilarityProblem([1] * genome_size)  # Max Ones
    coder = BinaryCoder(genome_size=genome_size)

    pipeline = TournamentSelection(tournament_size=2)
    pipeline = Clone(pipeline)
    pipeline = UniformCrossover(pipeline, p_cross=1.0, p_swap=0.5)
    pipeline = BitFlipMutation(pipeline, p_mut=1.0/genome_size)
    pipeline = Evaluate(pipeline)
    pipeline = Elitism(pipeline, num_elite=1)

    bog = None
    bsf = None

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

    num_generations = 20
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

    print()
    print("bsf =", bsf)


#############################################################################
#
# command line execute
#
#############################################################################
if __name__ == "__main__":
    print(sys.version)
    ga_example1()

