#!/usr/bin/env python

"""
test_pitt.py: tests classes specific to the Pittsburgh (Pitt) approch for
              eclypse.
"""

from eclypse.problems import BaseProblem
from eclypse.coders import FloatCoder
from eclypse.exec.pitt import PittBoundsCoder
from eclypse.exec.pitt import PittPointCoder

#############################################################################
#
# SimpleProblem
#
#  +---+
#  |\ 1|
#  | \ |
#  |0 \|
#  +---+
#
#############################################################################
class SimpleProblem(BaseProblem):
    """
    Essentially a classification problem.  There are two inputs, each ranging
    from 0.0 to 1.0.  Any input in the lower left half of the space will have
    a classification of 0.0.  The upper right half has a classification of
    1.0.
    
    The phenome is evaluated using a number of input points.  The fitness is
    the ratio of the number of correct responses to the total set of inputs.
    """
    def evaluate(self, phenome):
        fitness = 0
        total = 0
        for input1 in [i*0.1 for i in range(11)]:   # loop 0.0 to 1.0 by 0.1
            for input2 in [i*0.1+.05 for i in range(10)]:  # Avoid x+y=1
                answer = [((input1 + input2) >= 1.0) * 1.0]
                output = phenome.execute([input1, input2])
                total += 1
                if answer == output:
                    fitness += 1

        fitness = 1.0 * fitness / total
        return fitness



def test_PittBoundsCoder():
    """
    Test the PittBoundsCoder.
    """
    initRanges = [(0.0, 1.0)] * 3
    ruleCoder = FloatCoder(initRanges)

    coder = PittBoundsCoder(ruleCoder, 10, 10, 1, 1)
    genome = coder.create_random_genome()

    assert(len(genome) == 10)
    assert(len(genome[0]) == 3)



def test_PittPointCoder():
    """
    Test the PittPointCoder.
    """
    numInputs = 2
    numOutputs = 1
    initRanges = [(0.0, 1.0)] * (numInputs + numOutputs)
    ruleCoder = FloatCoder(initRanges)

    coder = PittPointCoder(ruleCoder, 2, 2, numInputs, numOutputs,
                           nearest_neighbor=True)
    genome = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    phenome = coder.decode_genome(genome)

    myProblem = SimpleProblem()
    fitness = myProblem.evaluate(phenome);
    print("fitness =", fitness)
    assert(fitness == 1.0)



if __name__ == "__main__":
    test_PittBoundsCoder()
    test_PittPointCoder()
