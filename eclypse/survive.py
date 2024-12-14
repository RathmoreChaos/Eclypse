#!/usr/bin/env python

"""
survive.py: Define the survival selection operators for a pipeline system.
"""

import random
import heapq
import functools

from eclypse.ops import BaseOp
from eclypse.select import select_cmp_default


#############################################################################
#
# BaseSurvival
#
#############################################################################
class BaseSurvival(BaseOp):
    """
    """
    def __init__(self, provider, select_cmp=select_cmp_default):
        super().__init__(provider=provider)
        self.select_cmp = select_cmp


#############################################################################
#
# Elitism
#
#############################################################################
class Elitism(BaseSurvival):
    def __init__(self, provider, num_elite, select_cmp=select_cmp_default):
        super().__init__(provider=provider, select_cmp=select_cmp)
        self.num_elite = num_elite

    def new_generation(self, population):
        super().new_generation(population)
        self.elite = heapq.nlargest(self.num_elite, population,\
                                    key=functools.cmp_to_key(self.select_cmp))
        #print("Storing Elite:")
        #for e in self.elite:
        #    print(e.genome, e.fitness)

    def generator(self):
        # Ideally I should shuffle the elite into the offspring population in
        # order to avoid any biases, however that means collecting everthing
        # into a pool first.  This would mean more work, and more potential
        # for error.  For now I won't worry about it.
        #print("Starting Elitism Generator")
        while 1:
            while self.elite:
                e = self.elite.pop()
                #print("Elite:", e.genome, e.fitness)
                yield e
            yield self.provider.pull()


#############################################################################
#
# BaseMuLambdaSurvival
#
#############################################################################
class BaseMuLambdaSurvival(BaseSurvival):
    def __init__(self, provider, num_lambda, select_cmp=select_cmp_default):
        super().__init__(provider=provider, select_cmp=select_cmp)
        self.num_lambda = num_lambda

    def new_generation(self, population):
        super().new_generation(population)
        self._mu = self.prior_generation
        self._lambda = []
        self.combined = []

    def combine_mu_lambda(_mu, _lambda):
        raise(NotImplementedError)

    def generator(self):
        #print("Starting mu + lambda Generator")
        while len(self._lambda) < self.num_lambda:
            self._lambda.append(self.provider.pull()) 

        self.combined = self.combine_mu_lambda(self._mu, self._lambda)
        self.combined = heapq.nlargest(len(self._mu), self.combined,
                                    key=functools.cmp_to_key(self.select_cmp))

        while 1:
            new_population = self.combined[:]
            random.shuffle(new_population)
            while new_population:
                yield new_population.pop()


#############################################################################
#
# MuCommaLambdaSurvival
#
#############################################################################
class MuCommaLambdaSurvival(BaseMuLambdaSurvival):
    def combine_mu_lambda(self, _mu, _lambda):
        return(_lambda)


#############################################################################
#
# MuPlusLambdaSurvival
#
#############################################################################
class MuPlusLambdaSurvival(BaseMuLambdaSurvival):
    def combine_mu_lambda(self, _mu, _lambda):
        return(_mu + _lambda)


#############################################################################
#
# unit_test
#
#############################################################################
if __name__ == "__main__":
    from eclypse.select import DeterministicSelection
    from eclypse.ops import Clone

    pipeline = DeterministicSelection()
    pipeline = Clone(pipeline)

    # This doesn't work
    population = [1, 2, 3, 4, 5]
    pipeline.new_generation(population)
    new_pop = [pipeline.pull() for ind in population]
    print(new_pop)

