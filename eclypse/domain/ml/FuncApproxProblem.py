#! /usr/bin/env python

"""
FuncApproxProblem.py - A function approximation problem.

The basic idea is that the genome contains a set of points (order isn't
important) that can then be sorted and connected linearly to create a
piecewise linear interpolation function.

It may make sense to use a variable length genome to allow the accuracy of the
generated function to increase in order to better approximate the target
function.  If you do this though, you may want to use some sort of parsimony
pressure in case the genomes begin to grow out of control.
"""

import random
import math

from eclypse.problems import BaseProblem


#############################################################################
#
# FuncApproxProblem
#
#############################################################################
class FuncApproxProblem(BaseProblem):
    def __init__(self, target_func, bounds):
        self.groups = []
        self.target_func = target_func

        # Right now this only works with 1 parameter
        #assert(len(bounds) == 1)
        self.bounds = bounds


    def better_than(self, fit1, fit2):
        return fit1 < fit2

    def equivalent_to(self, fit1, fit2):
        return fit1 == fit2

    def quality_of_fit(self, phenome, examples):
        fit = 0
        for example in examples:
            # example[0] is the list of inputs
            result = phenome.execute(example[0])
            # example[1] is the list of outputs.
            # For now assume only 1 output.
            err = result[0] - example[1][0]
            fit += err * err
        #return fit / (len(examples) - 1)
        return fit / len(examples)
        

    def evaluate(self, phenome):
        """
        Evaluate the fitness of an individual by classifying all the training
        examples.

        @param phenome: An ExecutableObject.
        @return: The sum squared error.
        """
        #print "training set:"
        #for example in self.training_set:
        #    print example
        return self.quality_of_fit(phenome, self.training_set)


    def classify_tests(self, phenome):
        """
        Perform an independent evaluation of an individual by having it
        classify a set of examples that were not used during training (the
        test set).

        @param phenome: An executableObject.
        @return: The sum squared error.
        """
        return self.quality_of_fit(phenome, self.test_set)


    def generate_example_groups(self, num_examples, num_groups):
        """
        Generates a set of examples and stores them internally in groups.
        The number of groups is specified by the parameter num_groups.
        """
        examples = self.generate_examples(num_examples)
        random.shuffle(examples)
        self.groups = [[examples[i] for i in range(len(examples)) \
                                    if i % num_groups == j] \
                       for j in range(num_groups)]


    def select_test_set_group(self, groupNum):
        """
        Returns the maximum number of examples available or possible.
        If there is no upper limit, None is returned.
        """
        self.test_set = self.groups[groupNum]

        self.training_set = []
        trainGroups = [self.groups[i] for i in range(len(self.groups)) \
                                          if i != groupNum]
        for group in trainGroups:
            self.training_set = self.training_set + group


    def get_total_examples(self):
        """
        Returns the maximum number of examples available or possible.
        If there is no upper limit, None is returned.
        """
        return None


    def generate_examples(self, num_examples):
        """
        Returns a list of examples for use in either training or testing.
        """
        examples = []
        for i in range(num_examples):
            point = [random.uniform(b[0], b[1]) for b in self.bounds]
            examples.append([point, [self.target_func(point)]])
        return examples



