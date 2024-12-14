#!/usr/bin/env python

"""
individual.py: defines core classes for Eclypse
"""

import random
import copy


#############################################################################
#
# is_iterable
#
#############################################################################
def is_iterable(gene_or_genome):
    """
    A helper function.
    """
    try:
        genome_iter = iter(gene_or_genome)
    except TypeError:
        return False
    return True


#############################################################################
#
# genome_size
#
#############################################################################
def genome_size(gene_or_genome):
    s = 0
    if is_iterable(gene_or_genome):
        for gene in gene_or_genome:
            s += genome_size(gene)
    elif gene_or_genome is not None:
        s = 1

    return s


#############################################################################
#
# Individual
#
#############################################################################
class Individual():
    def __init__(self, problem, genetic_coder, genome=None):
        self.problem = problem
        self.genetic_coder = genetic_coder
        if genome is None:
            self.genome = genetic_coder.create_random_genome()
        else:
            self.genome = genome
        self.fitness = None
        
    def evaluate(self):
        phenome = self.genetic_coder.decode_genome(self.genome)
        self.fitness = self.problem.evaluate(phenome)
        return self.fitness
        
    def clone(self):
        clone = copy.copy(self)
        clone.genome = self.genetic_coder.copy_genome(self.genome)
        return clone
    
    def size(self):
        return genome_size(self.genome)

    def better_than(self, other_ind):
        return self.problem.better_than(self.fitness, other_ind.fitness)
    
    def equivalent_to(self, other_ind):
        return self.problem.equivalent_to(self.fitness, other_ind.fitness)
    
    def __lt__(self, other):
        return not (self.better_than(other) & self.equivalent_to(other))

    def __str__(self):
        return(str(self.genome) + " " + str(self.fitness))


