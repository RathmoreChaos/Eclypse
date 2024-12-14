#!/usr/bin/env python

"""
test_select.py: Test the selection operators for a pipeline system.
"""

from eclypse.problems import SimilarityProblem
from eclypse.coders import BinaryCoder
from eclypse.ind import Individual
from eclypse.ops import Clone
from eclypse.ops import BitFlipMutation
from eclypse.ops import Evaluate
from eclypse.select import TournamentSelection
from eclypse.select import DeterministicSelection


def test_TournamentSelection():

    problem = SimilarityProblem([1,1,1,1,1])  # OneMax
    coder = BinaryCoder(5)

    ind1 = Individual(problem, coder, [0,0,0,0,0])
    ind2 = Individual(problem, coder, [0,0,0,0,1])
    ind3 = Individual(problem, coder, [0,0,0,1,0])
    ind4 = Individual(problem, coder, [0,0,0,1,1])
    ind5 = Individual(problem, coder, [1,1,1,1,1])
    population = [ind1, ind2, ind3, ind4, ind5]
    for ind in population:
        ind.evaluate()
        print(ind.genome, ind.fitness)

    print()

    #pipeline = DeterministicSelection()
    pipeline = TournamentSelection(2)
    pipeline = Clone(pipeline)
    pipeline = BitFlipMutation(pipeline, p_mut=0.2)
    #pipeline = Evaluate(pipeline)

    #for generatation in range(100):
    if 1:
        pipeline.new_generation(population)
        new_pop = [pipeline.pull() for i in range( len(population)*2 )]
        population = new_pop

    for ind in new_pop:
        ind.evaluate()
        print(ind.genome, ind.fitness)


