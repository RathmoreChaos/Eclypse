#!/usr/bin/env python

"""
ops.py: Define the basic operators for a pipeline system.
"""

import random
import copy
import math
import numpy as np

from eclypse.ind import Individual, is_iterable


#############################################################################
#
# BaseOp
#
# Operator base class
#
#############################################################################
class BaseOp():
    """This is the Operator base class."""
    def __init__(self, provider=None):
        self.provider = provider

    def new_generation(self, population):
        self.prior_generation = population
        self.op_iter = self.generator()
        if self.provider is not None:
            self.provider.new_generation(population)

    def pull(self):
        return next(self.op_iter)

    def generator(self):
        raise NotImplementedError


#############################################################################
#
# Clone
#
#############################################################################
class Clone(BaseOp):
    """ Clones an individual.  Making this explicit saved confusion and time."""
    def generator(self):
        while 1:
            ind = self.provider.pull()
            new_ind = ind.clone()
            yield new_ind


#############################################################################
#
# Evaluate
#
#############################################################################
class Evaluate(BaseOp):
    """
    Calculates the fitness of an individual as it comes through the pipeline.
    Some evaluations are expensive, so it pays to make this explicit to avoid
    unnecessary duplication.
    """
    def generator(self):
        while 1:
            ind = self.provider.pull()
            #print("Genome:", ind)
            ind.evaluate()
            yield ind


#############################################################################
#
# BaseMutationOp
#
#############################################################################
class BaseMutationOp(BaseOp):
    """
    A base class to make it easier to write mutation operators.
    Instead of writing a complete generator function, subclasses can just
    implement the mutate_gene() method.
    """
    def __init__(self, provider, p_mut=None, e_mut=None, recurse=True):
        super().__init__(provider=provider)
        assert((p_mut is None) != (e_mut is None)) # One and only one is defined
        self.p_mut = p_mut
        self.e_mut = e_mut
        self.recurse = recurse

    def generator(self):
        while 1:
            ind = self.provider.pull()

            # Figure out the mutation probability p_mut
            p_mut = self.p_mut
            if p_mut is None:
                size = self.genome_size(ind.genome, self.recurse)
                p_mut = float(self.e_mut) / size

            # Mutate the individual
            ind.genome = self.mutate_genome(ind.genome, p_mut)

            yield ind


    # XXX Should this be a general utility function?
    def genome_size(self, genome, recurse=True):
        size = 0
        for g in genome:
            if is_iterable(g):
                if recurse:
                    size += genome_size(g)
            else:
                size += 1


    def mutate_genome(self, genome, p_mut):
        """
        Traverses the genome, calling mutate_gene() when appropriate
        """
        for i,g in enumerate(genome):
            if is_iterable(g):
                if self.recurse:
                    genome[i] = self.mutate_genome(g, p_mut)
            else:
                if random.random() <= p_mut:
                    genome[i] = self.mutate_gene(g)

        return genome


    def mutate_gene(self, gene):
        raise NotImplementedError



#############################################################################
#
# BitFlipMutation
#
#############################################################################
class BitFlipMutation(BaseMutationOp):
    def mutate_gene(self, gene):
        return int(not gene)



#############################################################################
#
# GaussianMutation
#
#############################################################################
class GaussianMutation(BaseMutationOp):
    def __init__(self, provider, sigma, p_mut = 1.0):
        super().__init__(provider=provider, p_mut=p_mut, e_mut=None)
        self.sigma = sigma

    def mutate_gene(self, gene):
        gene += random.gauss(0.0, self.sigma)
        return (gene)


#############################################################################
#
# class AdaptiveMutation
#
#############################################################################
class AdaptiveMutation(BaseOp):
    """
    Gaussian mutation operator for real valued genes.  It uses the ES style
    adaptive mutation mechanism described in T. Back and H.-P. Schwefel,
    "An Overview of Evolutionary Algorithms for Parameter Optimization",
    Evolutionary Computation, 1(1):1-23, The MIT Press, 1993.  Each gene is a
    array of 2 numbers.  The first is the actual gene value and the second is
    the standard deviation (sigma) associated with the gene.  The standard
    deviations are are adapted from one generation to the next.

    NOTE: This operator should be used with the AdaptiveFloatCoder.
    """
    def __init__(self, provider, sigma_bounds):
        """
        @param provider: The operator that immediately precedes this one in
                         the pipeline.
        @param sigma_bounds: A tuple containing the minimum and maximum sigma
                             values allowed.  Sigma values that go beyond
                             these bounds will be clipped.
        """
        super().__init__(provider=provider)
        self.sigma_bounds = sigma_bounds
        self.tau = 1.0/math.sqrt(2 * math.sqrt(len(sigma_bounds)))
        self.tau_prime = 1.0/math.sqrt(2 * len(sigma_bounds))

        #if init_sigmas is not None:
        #    self.init_sigmas = init_sigmas
        #elif gene_init_bounds is not None:   # Default calc for init_sigmas
        #    denom = math.sqrt(len(gene_init_bounds))
        #    self.init_sigmas = [(b[1]-b[0]) / denom for b in gene_init_bounds]
        #else:
        #    raise ValueError("init_sigmas or gene_init_bounds must be provided")
        #
        #self.num_sigma_sets = 0

    def generator(self):
        while 1:
            ind = self.provider.pull()

            tau_prime_term = self.tau_prime * random.gauss(0,1)
#            try:
#                sigmas = ind.sigmas
#            except AttributeError:
#                self.num_sigma_sets += 1
#                sigmas = self.init_sigmas
#                ind.sigmas = sigmas
#
#            for i in range(len(ind.sigmas)):
#                ##print("self.tau_prime_term =", self.tau_prime_term)
#                ##print("self.tau =", self.tau)
#                ##print("ind.sigmas[" + str(i) + "]", ind.sigmas[i])
#                ind.sigmas[i] *= math.exp( self.tau_prime_term + self.tau * \
#                                  random.gauss(0,1) )
#                ##print("self.sigma_bounds =", self.sigma_bounds)
#                ind.sigmas[i] = np.clip( ind.sigmas[i],
#                                         self.sigma_bounds[i][0],
#                                         self.sigma_bounds[i][1] )
#
#                ind.genome[i] += ind.sigmas[i] * random.gauss(0,1)

            new_genome = []
            for i in range(len(ind.genome)):
                gene = ind.genome[i][0]
                sigma = ind.genome[i][1]
                bound = self.sigma_bounds[i]
#            for g,sd in ind.genome:
                ##print("self.tau_prime_term =", self.tau_prime_term)
                ##print("self.tau =", self.tau)
                ##print("ind.sigmas[" + str(i) + "]", ind.sigmas[i])
#                ind.sigmas[i] *= math.exp( self.tau_prime_term + self.tau * \
#                                  random.gauss(0,1) )
                sigma *= math.exp(tau_prime_term + self.tau*random.gauss(0,1))
                ##print("self.sigma_bounds =", self.sigma_bounds)
                sigma = np.clip(sigma, bound[0], bound[1])

                gene += sigma * random.gauss(0,1)
                new_genome.append( (gene, sigma) )

            ind.genome = new_genome
            yield ind


#############################################################################
#
# class CMA_Generate
#
#############################################################################
class CMA_Generate(BaseOp):
    def __init__(self, cma_es, problem, coder):
        super().__init__(provider=None)
        self.cma_es = cma_es
        self.problem = problem
        self.coder = coder

    # It would be convenient and easy to put the update here instead of the
    # CMA_Update operator, but that's just not how the ES folks think about
    # things.  In the hopes of not confusing things too much, I'll just make
    # two operators.
    def new_generation(self, prior_generation):
        super().new_generation(prior_generation)

    def generator(self):
        genomes = self.cma_es.ask()
        population = []
        for genome in genomes:
            # XXX This isn't cool.  This class shouldn't be creating
            #     individuals.  Need to pass a prototype individual in, or use
            #     another layer of indirection here.
            ind = Individual(self.problem, self.coder, genome)
            population.append(ind)

        while 1:
            for ind in population:
                yield ind
            random.shuffle(population)


#############################################################################
#
# class CMA_Update
#
#############################################################################
class CMA_Update(BaseOp):
    def __init__(self, provider, cma_es):
        super().__init__(provider=provider)
        self.cma_es = cma_es

    def generator(self):
        popsize = self.cma_es.popsize
        population = [self.provider.pull() for i in range(popsize)]

        self.cma_es.tell([ind.genome for ind in population],
                         [ind.fitness for ind in population])

        while 1:
            for ind in population:
                yield ind
            random.shuffle(population)


#############################################################################
#
# UniformCrossover
#
#############################################################################
class UniformCrossover(BaseOp):
    def __init__(self, provider, p_cross, p_swap=0.5):
        super().__init__(provider=provider)
        self.p_cross = p_cross
        self.p_swap = p_swap

    def generator(self):
        while 1:
            ind1 = self.provider.pull()
            ind2 = self.provider.pull()
            assert(len(ind1.genome) == len(ind2.genome))
            if random.random() <= self.p_cross:
                for i in range(len(ind1.genome)):
                    if random.random() <= self.p_swap:
                        (ind1.genome[i], ind2.genome[i]) = \
                            (ind2.genome[i], ind1.genome[i])

            yield ind1
            yield ind2


#############################################################################
#
# unit_test
#
#############################################################################
if __name__ == "__main__":
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


