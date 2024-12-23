#! /usr/bin/env python

##############################################################################
#
#   LEAP - Library for Evolutionary Algorithms in Python
#   Copyright (C) 2004  Jeffrey K. Bassett
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the Free Software
#   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
##############################################################################

from eclypse.problems import BaseProblem


#############################################################################
#
# class ExecutableObject
#
#############################################################################
class ExecutableObject:
    """
    For our purposes, an ExecutablObject simply takes input and returns some
    appropriate output, by way of the execute() method.  This might be done
    just once (such as for a classification task) or repeatedly (such as for
    a robot maneuvering through a maze).

    In practice, an ExecutableObject could be implemented using any one of a
    number of representations, including a rule set, a neural network, a
    finite state machine, a LISP program, or anything else that might make
    sense.  All implementations would be written as subclasses of this class.

    From the EA perspective, the ExecutableObject subclasses are phenotypes
    and would be returned by the decodeGenome() method of an encoder that is
    specific to the representation.

    The purpose of this class is to disassociate the execution semantics from
    the underlying representation.  Thus we can create a variety of domains
    (classification, robot, multi-agent) which are independent of the
    representation chosen.
    """

    def execute(self, input):
        """
        @param input: Domain dependent information appropriate for the task
                      at hand.  For example, sensor input for a robot
                      controller.
        @return: Domain dependent output, such as a set of actions for a
                 robot.
        """
        raise NotImplementedError


#############################################################################
#
# class LearningProblem:
#
#############################################################################
class LearningProblem(BaseProblem):
    """
    A general base class for learning problems.  Phenomes must be
    ExecutableObjects.
    """

    def __init__(self, training_set, test_set, validation_set=[]):
        self.training_set = training_set
        self.test_set = test_set
        self.validation_set = validation_set

    def calc_sample_error(self, phenome, sample_set):
        sampleInput, sampleOutput = sample_set
        results = [phenome.execute(inp) for inp in sampleInput]
        error = sum([abs(results - sampleOutput)])
        return error

    def evaluate(self, phenome):
        return calc_sample_error(phenome, self.training_set)

    def test(self, phenome):
        return calc_sample_error(phenome, self.test_set)

    def validate(self, phenome):
        return calc_sample_error(phenome, self.validation_set)

    def better_than(self, fit1, fit2):
        return fit1 < fit2

    def equivalent_to(self, fit1, fit2):
        return fit1 == fit2


#############################################################################
#
# class MimicProblem:
#
#############################################################################
class MimicProblem(LearningProblem):
    """
    The goal of this problem is to learn to mimic an existing executable
    object.  This makes for a simple way to create problems if you already
    have something that works in a different representation.
    """

    def __init__(self, targetEO, training_input, test_input, \
                 validation_input=[]):
        self.targetEO = targetEO
        training_set = [training_input, [targetEO.execute(inp) for inp in \
                        training_input]]
        test_set = [test_input, [targetEO.execute(inp) for inp in test_input]]
        validation_set = [validation_input, [targetEO.execute(inp) for inp in
                          validition_input]]
        super().__init__(training_set, test_set, validation_set)



