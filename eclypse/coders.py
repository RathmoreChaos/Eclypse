#!/usr/bin/env python

"""
coders.py: defines coder classes for Eclypse
"""

import random
import copy
import math



#############################################################################
#
# BaseCoder
#
# A personal note on the name of this class.  I have tried many names for this
# class, and I have never found one that I liked.  I don't think I like Coder
# much either.  In the past I've called it Encoder, Decoder, Encoding,
# Representation, and even more.
#
# So what is this class about?  It is a sort of umbrella class where I store
# functions that do things to a specific kind of genome.  This makes it easy
# to just swap out one type of a coder for another if I want to experiment
# with or compare different representations.  All the rest of the software can
# remain the same.
#
# The name ends with "er" because this class should never contain a genome, it
# just acts on them.  Never store your genomes here.  I use the term "Coder"
# because at times in the past I have had an "encode_genome" method in
# addition to decode_genome, and I may add that again.  Coder seemed ambiguous
# as to the direction (i.e. en- vs de-) that the coding occurs.
#
#############################################################################
class BaseCoder():
    def create_random_genome(self):
        raise NotImplementedError
    
    def decode_genome(self, genome):
        raise NotImplementedError

    def copy_genome(self, genome):
        """
        Returns a copy of the given genome.
        The default here is to use deepcopy, but this can be VERY time
        consuming, so it really pays to override this function with something
        geared towards your specific representation.
        """
        return copy.deepcopy(genome)


#############################################################################
#
# BinaryCoder
#
#############################################################################
class BinaryCoder(BaseCoder):
    def __init__(self, genome_size):
        self.genome_size = genome_size
    
    def create_random_genome(self):
        return([random.choice([0,1]) for i in range(self.genome_size)])
    
    def decode_genome(self, genome):
        return genome   # the genome is the phenome

    def copy_genome(self, genome):
        return genome[:]


#############################################################################
#
# Binary2FloatCoder
#
#############################################################################
class Binary2FloatCoder(BinaryCoder):
    def __init__(self, bits_per_float_list, float_bounds):
        assert(len(bits_per_float_list) == len(float_bounds))
        super().__init__(genome_size=sum(bits_per_float_list))
        self.bits_per_float_list = bits_per_float_list
        self.float_bounds = float_bounds

    def decode_genome(self, genome):
        bit_offset = 0
        phenome = []
        for n_bits, bound in zip(self.bits_per_float_list, self.float_bounds):
            bits = genome[bit_offset : bit_offset + n_bits]
            max_ival = 2 ** len(bits)
            ival = self.binary2int(bits)
            fval = (float(ival) / max_ival) * (bound[1]-bound[0]) + bound[0]

            bit_offset += n_bits
            phenome.append(fval)

        return phenome
            
    def binary2int(self, binary_list):
        binary_str = "".join(str(x) for x in binary_list)
        return int(binary_str, 2)


#############################################################################
#
# GrayBinary2FloatCoder
#
#############################################################################
class GrayBinary2FloatCoder(Binary2FloatCoder):
   def binary2int(self, gray_binary_list):
        ival = 0
        prev_bit = 0
        for gray_bit in gray_binary_list:
            bit = gray_bit ^ prev_bit  # XOR
            ival = (ival << 1) + bit   # Shift left and add bit
            prev_bit = bit

        return ival

#############################################################################
#
# FloatCoder
#
#############################################################################
class FloatCoder(BaseCoder):
    def __init__(self, init_ranges):
        self.init_ranges = init_ranges
    
    def create_random_genome(self):
        return([random.uniform(r[0], r[1]) for r in self.init_ranges])
    
    def decode_genome(self, genome):
        return genome   # the genome is the phenome

    def copy_genome(self, genome):
        return genome[:]


#############################################################################
#
# class AdaptiveFloatCoder
#
#############################################################################
class AdaptiveFloatCoder(FloatCoder):
    """
    Defines each gene as a tuple containing two values: the gene value
    itself, and the mutation sigma (standard deviation) value used by the
    AdaptiveMutation operator.  Thus the genome is defined as an
    array of these tuples.
    """
    def __init__(self, init_ranges, init_sigmas=None):
        """
        The init_ranges parameter is a list of tuples containing a lower and
        upper bound for each gene in a new genome.  The bounds parameter takes
        the same form as init_ranges, but defines the bounds that each gene
        will be "clipped" to.  Clipping is handled by AdaptiveMutation when
        mutation occurs.

        The init_sigmas parameter contains a list of initial sigma values for
        each gene.  If init_sigmas is not set (i.e. if set to None), then
        reasonable values will be calculated from the init_ranges.
        """
        if init_sigmas == None:
            # I borrowed this calculation from Mitch Potter's ECkit code.
            # I don't know for certain if this is how the ES crowd does it.
            init_sigmas = [(ir[1] - ir[0]) / math.sqrt(len(init_ranges))
                           for ir in init_ranges]

        assert(len(init_ranges) == len(init_sigmas))
        self.init_ranges = init_ranges
        self.init_sigmas = init_sigmas

    def create_random_genome(self):
        "Generates a randomized genome for this encoding"
        init_vals = [random.uniform(r[0], r[1]) for r in self.init_ranges]
        genome = list(zip(init_vals, self.init_sigmas))
        return genome

    def decode_genome(self, genome):
        return [g[0] for g in genome]


#############################################################################
#
# unit_test
#
#############################################################################
if __name__ == "__main__":
    from eclypse.problems import SimilarityProblem
    from eclypse.ind import Individual

    problem = SimilarityProblem([1,1,1,1,1])  # OneMax
    coder = BinaryCoder(5)
    ind1 = Individual(problem, coder, [0,0,0,0,0])
    ind2 = Individual(problem, coder, [1,1,1,1,1])

    ind1.evaluate()
    ind2.evaluate()

    assert(ind2.better_than(ind1))
    assert(ind1.better_than(ind2) == False)
    assert(ind1.equivalent_to(ind2) == False)
    print("passed")


