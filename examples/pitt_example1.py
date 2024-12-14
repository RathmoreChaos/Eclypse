#!/usr/bin/env python

"""
pitt_example1.py: A simple example to illustrate how a Pittsburgh rule system
                  works.  Genomes are all fixed length, and standard (i.e.
                  fixed length) reproductive operators are used.
"""

import random
import sys

from eclypse.ind import Individual
from eclypse.problems import BaseProblem
from eclypse.coders import FloatCoder
from eclypse.select import TournamentSelection
from eclypse.ops import Clone, GaussianMutation, UniformCrossover, Evaluate
from eclypse.survive import Elitism
from eclypse.exec.pitt import PittPointCoder, PittUniformCrossover


#############################################################################
#
# class ExecTestProblem
#
#  +-----+
#  |\  1 |
#  | \   |
#  |  \  |
#  |   \ |
#  | 0  \|
#  +-----+
#
#############################################################################
class ExecTestProblem(BaseProblem):
    """
    This is essentially a classification problem.  There are two inputs, each
    ranging from 0.0 to 1.0.  Any input in the lower left half of the space
    will have a classification of 0.0.  The upper right half has a
    classification of 1.0.

    The phenome is evaluated using a number of input points.  The fitness is
    the ratio of the number of correct responses to the total set of inputs.
    """
    def evaluate(self, phenome):
        fitness = 0
        total = 0
        for input1 in [i*0.1 for i in range(11)]:   # loop 0.0 to 1.0 by 0.1
            for input2 in [i*0.1 + .05 for i in range(10)]:  # Avoid x+y=1
                answer = [float((input1 + input2) >= 1.0)]
                output = phenome.execute([input1, input2])
                # XXX Consider using one-hot encoding instead
                refined_output = [float(o > 0.5) for o in output]
                total += 1
                if answer == refined_output:
                    fitness += 1

        fitness = float(fitness) / total
        return fitness

    def better_than(self, fit1, fit2):
        return fit1 > fit2

    def equivalent_to(self, fit1, fit2):
        return fit1 == fit2



#############################################################################
#
# pitt_example1
#
#############################################################################
def pitt_example1():
    num_inputs=2
    num_outputs=1
    init_ranges = [[0.0, 1.0]] * (num_inputs + num_outputs)
    min_rules = max_rules = 5

    problem = ExecTestProblem()
    rule_coder = FloatCoder(init_ranges)
    coder = PittPointCoder(rule_coder, min_rules, max_rules, \
                           num_inputs, num_outputs)

    pipeline = TournamentSelection(tournament_size=2)
    pipeline = Clone(pipeline)
    #pipeline = PittUniformCrossover(pipeline, p_cross=0.8, p_xfer=0.2)
    pipeline = UniformCrossover(pipeline, p_cross=0.8, p_swap=0.2)
    pipeline = GaussianMutation(pipeline, sigma=0.2)
    pipeline = Evaluate(pipeline)

    bog = None
    bsf = None

    # Generate initial population
    pop_size = 10
    population = [Individual(problem, coder) for _ in range(pop_size)]
    print("\nGeneration", 0)
    bog = population[0]
    bog.evaluate()
    for ind in population:
        ind.evaluate()
        #print(ind)
        if ind.better_than(bog):
            bog = ind
    bsf = bog
    print("bsf =", bsf)

    num_generations = 20
    for generation in range(1, num_generations+1):
        pipeline.new_generation(population)
        new_population = [pipeline.pull() for _ in range(pop_size)]
        bog = population[0]

        # Print the population
        print("\nGeneration", generation)
        for ind in new_population:
            if ind.better_than(bog):
                bog = ind
            #print(ind)

        if bog.better_than(bsf):
            bsf = bog
        print("bsf =", bsf)
        population = new_population
        #time.sleep(1)



#############################################################################
#
# command line execute
#
#############################################################################
if __name__ == "__main__":
    print(sys.version)
    pitt_example1()

