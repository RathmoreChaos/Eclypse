#!/usr/bin/env python

"""
pitt_example3.py: A simple example to illustrate how a Pittsburgh rule system
                  works.  Genomes are all fixed length, and standard (i.e.
                  fixed length) reproductive operators are used.
"""

import random
import sys

import gym

from numpy import inf

from eclypse.ind import Individual
from eclypse.problems import BaseProblem
from eclypse.coders import FloatCoder
from eclypse.select import TournamentSelection, select_cmp_default,\
              select_cmp_lexicographic_parsimony, select_cmp_penalty_parsimony
from eclypse.ops import Clone, GaussianMutation, UniformCrossover, Evaluate
from eclypse.survive import Elitism
from eclypse.exec.pitt import PittPointCoder, PittUniformCrossover
from eclypse.env.agent.gym import OpenAIGymProblem


#############################################################################
#
# pitt_example3
#
#############################################################################
def pitt_example3():
    num_inputs=4
    num_outputs=1
    init_ranges = [[-1.0, 1.0]] * (num_inputs + num_outputs)
    min_rules = 5
    max_rules = 10

    #sim_name = "CartPole-v1"
    sim_name = "CartPole-v0"
    n_steps = 500
    problem = OpenAIGymProblem(sim_name, max_timestep=n_steps, num_episodes=10)
    rule_coder = FloatCoder(init_ranges)
    coder = PittPointCoder(rule_coder, min_rules, max_rules, \
                           num_inputs, num_outputs)
    #parsimony = select_cmp_default
    parsimony = select_cmp_lexicographic_parsimony
    #parsimony = select_cmp_penalty_parsimony(penalty=10.0, problem=problem)

    pipeline = TournamentSelection(tournament_size=2, select_cmp=parsimony)
    pipeline = Clone(pipeline)
    #pipeline = UniformCrossover(pipeline, p_cross=0.8, p_swap=0.2)
    pipeline = PittUniformCrossover(pipeline, p_cross=0.8, p_xfer=0.2)
    pipeline = GaussianMutation(pipeline, sigma=0.1)
    pipeline = Elitism(pipeline, num_elite=1, select_cmp=parsimony)
    pipeline = Evaluate(pipeline)

    bog = bsf = None  # Best of generation, best so far
    bsf_last_fitness = -inf

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
        print(len(ind.genome), ind.fitness)
    bsf = bog
    bsf_last_fitness = bsf.fitness
    print()
    #print("bsf =", bsf)
    print("bsf =", bsf, len(bsf.genome), bsf.fitness)

    # Run the EA
    num_generations = 50
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
            print(len(ind.genome), ind.fitness)

        if bog.better_than(bsf) or\
           (bog.equivalent_to(bsf) and parsimony(bog, bsf) == 1):
            bsf = bog

        #print("bsf =", bsf)
        print("bsf =", len(bsf.genome), bsf.fitness)
        population = new_population
        bsf_last_fitness = bsf.fitness

    print(bsf)
    exec_phenome = bsf.genetic_coder.decode_genome(bsf.genome)

    # Show the solution in the simulation
    env = gym.make(sim_name)

    observation = env.reset()
    for timestep in range(n_steps * 10):
        env.render()
        output = exec_phenome.execute(observation.tolist())
        action = int(output[0] > 0.5)
        observation, reward, done, info = env.step(action)
        if done:
            break

    env.close()


#############################################################################
#
# command line execute
#
#############################################################################
if __name__ == "__main__":
    print(sys.version)
    pitt_example3()

