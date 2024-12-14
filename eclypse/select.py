#!/usr/bin/env python

"""
select.py: Define the parent selection operators for a pipeline system.
"""

import random
import copy

from eclypse.ops import BaseOp

# Not yet implemented:
#  ProportionalSelection
#  RankSelection
#  UniformSelection (i.e. random w/ replacement)
#  TruncationSelection


#############################################################################
#
# select_cmp_default
#
#############################################################################
def select_cmp_default(ind1, ind2):
    if ind1.better_than(ind2):
        return 1
    elif ind1.equivalent_to(ind2):
        return 0

    return -1


#############################################################################
#
# select_cmp_lexicographic_parsimony
#
#############################################################################
def select_cmp_lexicographic_parsimony(ind1, ind2):
    if ind1.better_than(ind2):
        return 1
    elif ind1.equivalent_to(ind2):
        if len(ind1.genome) < len(ind2.genome):
            return 1
        elif len(ind1.genome) == len(ind2.genome):
            return 0

    return -1


#############################################################################
#
# select_cmp_penalty_parsimony
#
#############################################################################
class select_cmp_penalty_parsimony():
    """
    When comparing two individuals, the fitness of each is temporarily
    adjusted by adding the genome size times a penalty value.  Note that this
    means the penalty should be negative if using a minimization problem.
    """
    def __init__(self, penalty, problem):
        self.penalty = penalty

    def __call__(self, ind1, ind2):
        fitness1 = ind1.fitness + ind1.size() * self.penalty
        fitness2 = ind2.fitness + ind2.size() * self.penalty
        if ind1.problem.better_than(fitness1, fitness2):
            return 1
        elif ind1.problem.equivalent_to(fitness1, fitness2):
            return 0

        return -1


#############################################################################
#
# BaseSelection
#
#############################################################################
class BaseSelection(BaseOp):
    """
    """
    def __init__(self, select_cmp=select_cmp_default):
        super().__init__(provider=None)
        self.select_cmp = select_cmp


#############################################################################
#
# DeterministicSelection
#
#############################################################################
class DeterministicSelection(BaseSelection):
    """
    Random selection without replacement, then repeat.  The population is
    shuffled and each individual will be selected only once until every
    individual has been selected.  At that point the process starts over,
    reshuffling and selecting each individual in a new order.  This continues
    indefinitely as long as new individuals are pulled.
    """
    def __init__(self, shuffle=True):
        #super().__init__(provider=None)
        super().__init__()
        self.shuffle = shuffle

    def generator(self):
        shuffled_pop = self.prior_generation[:]
        while 1:
            if self.shuffle:
                random.shuffle(shuffled_pop)
            for ind in shuffled_pop:
                yield ind



#############################################################################
#
# TournamentSelection
#
#############################################################################
class TournamentSelection(BaseSelection):
    """
    Uses DeterministicSelection to cycle through the entire population in
    random order, each time collecting a pool of size tournament_size then
    returning the most fit individual, and "discarding" the rest from
    consideration, at least until the next time through the population.
    Selection continues indefinitely until the desired number of individuals
    have been selected.

    Note that this is slightly different from the standard implementation
    which simply selects the pool members randomly with replacement.  This
    change was made to reduce genetic drift by increasing every individual's
    chance to compete in a tournament.
    """
    def __init__(self, tournament_size, select_cmp=select_cmp_default):
        #super().__init__(provider=None, select_cmp=select_cmp)
        super().__init__(select_cmp=select_cmp)
        self.tournament_size = tournament_size
        self.det_select = DeterministicSelection()


    def new_generation(self, population):
        self.det_select.new_generation(population)
        super().new_generation(population)


    def generator(self):
        while 1:
            ts = self.tournament_size
            tournament = [self.det_select.pull() for _ in range(ts)]
            besties = [tournament[0]]  # All the ties for best

            # Search for the best in the tournament
            for ind in tournament[1:]:
                cmp_result = self.select_cmp(ind, besties[0])
                if cmp_result == 0:  # ind ties besties
                    besties.append(ind)
                elif cmp_result > 0:  # ind is better than besties
                    besties = [ind]

            yield random.choice(besties)  # randomize ties


#############################################################################
#
# unit_test
#
#############################################################################
if __name__ == "__main__":

    from eclypse.problems import SimilarityProblem
    from eclypse.coders import BinaryCoder
    from eclypse.ind import Individual
    from eclypse.ops import Clone
    from eclypse.ops import BitFlipMutation
    from eclypse.ops import Evaluate

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


