#!/usr/bin/env python

"""
test_ops.py: Tests the basic operators for a pipeline system.
"""

#class BaseOp():
#class Clone(BaseOp):
#class Evaluate(BaseOp):
#class BaseMutationOp(BaseOp):
#class BitFlipMutation(BaseMutationOp):
#class GaussianMutation(BaseMutationOp):
#class AdaptiveMutation(BaseOp):
#class CMA_Generate(BaseOp):
#class CMA_Update(BaseOp):
#class UniformCrossover(BaseOp):


def test_pipeline():
    from eclypse.problems import SimilarityProblem
    from eclypse.coders import BinaryCoder
    from eclypse.ind import Individual
    from eclypse.select import DeterministicSelection
    from eclypse.ops import Clone
    from eclypse.ops import UniformCrossover
    from eclypse.ops import BitFlipMutation
    from eclypse.ops import Evaluate

    problem = SimilarityProblem([1,1,1,1,1])  # OneMax)
    coder = BinaryCoder(5)

    ind1 = Individual(problem, coder, [0,0,0,0,0])
    ind2 = Individual(problem, coder, [1,1,1,1,1])
    population = [ind1, ind2]
    for ind in population:
        ind.evaluate()
        print(ind.genome, ind.fitness)

    print()

    pipeline = DeterministicSelection(shuffle=False)
    pipeline = Clone(pipeline)
    pipeline = UniformCrossover(pipeline, p_cross=1.0, p_swap=0.5)
    pipeline = BitFlipMutation(pipeline, p_mut=1.0)
    pipeline = Evaluate(pipeline)

    pipeline.new_generation(population)
    new_pop = [pipeline.pull() for i in range( len(population)*1 )]

    for ind in new_pop:
        print(ind.genome, ind.fitness)


