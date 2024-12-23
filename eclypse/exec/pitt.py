#! /usr/bin/env python

##############################################################################
#
#   Eclypse
#   Copyright (C) 2020  Jeffrey K. Bassett
#
##############################################################################

import sys
import random
import math

from eclypse.exec.base import ExecutableObject
from eclypse.coders import BaseCoder
from eclypse.ops import BaseOp


#############################################################################
#
# RuleInterp
#
# This is a basic rule interpreter for Pitt approach systems (or any rule
# system, hopefully).  The goal is to make it versatile enough to handle some
# of the more standard rule representations and generalization techniques.
#
# This interpreter is also capable of keeping internal memory.  Defining the
# init_mem parameter with a list of values will set the initial values for
# memory, and define the number of memory slots based on the length of the
# list.  It will also wire up those memory slots to an equal number of
# condition values and action values in the rules, always the ones at the
# end.  Note: I think the Neural Net community has done some interesting work
# with memory.  It would be interesting to replace these memory units with
# something that more closely resembles and LSTM or GRU unit.  Basically that
# would mean that there are 2 memory actions per unit.  One specifies whether
# or not to make a change, and the other specifies the value to change it to.
#
# ruleset:
#    [ [ c1 c1'  c2 c2' ... cn cn'  a1 a2 ... am rank ] 
#      [ Rule 2 (as above) ]
#      ...
#      [ Rule N ] ]
#
#   cx and cx': are low and high bounds on one condition.
#      cx can be less than, equal to or greater than cx'
#      If binary: cx != cx' is a wildcard  (See Smith LS-1)
#      If float:  cx != cx' is a range     (See SAMUEL)
#         This means that the condition is a point when cx == cx' for all x,
#         which works well with Nearest Neighbor generalization on.
#   ax: is an action.
#   rank: A value indicating which rules should have precedence over other
#         rules during conflict resolution.  Rules with lower rank values will
#         be preferred.  The parameter use_alternate_ranks must be set to True
#         for these to take effect.  Otherwise more specific rules will be
#         preferred during conflict resolution.
# 
# Options:
#    partial_matching (default = False)
#       During the matching phase, if no rules match the given input then all
#       the rules are check again to see how many conditions of each rule
#       match.  Rules with the most matching conditions are the winners.  In
#       cases of ties, either rule specificity or rule ranks are used,
#       depending on the value of RuleRanksOn (see below).  If there are still
#       ties, then one of the remaining rules is chosen randomly.
#
#    nearest_neighbor (default = False)
#       This supercedes/overrides partial_matching (above).  During the
#       matching phase, if no rules match the given input, then a distance
#       calculation is performed between the input and each rule in the
#       rule set, and the closest rule is chosen.  In the case of ties,
#       one of the closest rules is chosen randomly.
#
#    use_alternate_ranks (default = False)
#       When true, conflict resolution among multiple rules that match the
#       input is performed based on a user provided rank value instead of rule
#       specificity (i.e. rule with the smallest condition area).  Note that
#       rule ranks are provided by appending a rank value at the end of each
#       rule.  Rules with smaller rank values are preferred over those with
#       larger values.
#
#############################################################################
class RuleInterp(ExecutableObject):
    """
    Rule interpreter for Pitt approach style rule learning.  
    Rules take the form: input_pairs, memory_input_pairs,
                         output, memory_output
    """

    def __init__(self, ruleset, num_inputs, num_outputs, init_mem=[],
                 partial_matching = False, nearest_neighbor = False,
                 use_alternate_ranks = False, nn_interpolate = False):
        self.ruleset = ruleset
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.memRegs = init_mem
        self.partial_matching = partial_matching
        self.nearest_neighbor = nearest_neighbor
        self.use_alternate_ranks = use_alternate_ranks
        self.nn_interpolate = nn_interpolate

        self.numMemory = len(init_mem)
        self.numConditions = self.num_inputs + self.numMemory
        self.numActions = self.num_outputs + self.numMemory
        if self.use_alternate_ranks:
            self.ruleRanks = self.calc_rule_generality(ruleset)
        else:
            self.ruleRanks = [rule[-1] for rule in ruleset]



    def calc_rule_generality(self, ruleset):
        """
        Calculates how general each rule is.  Rules that cover a larger
        portion of the condition space will have higher values.

        Note that one is added to each dimension when calculating the area.
        This prevents any dimension from equaling zero, which would result in
        the area (generality) equaling zero.  Some of these zero area rules
        are conceptually more general than others.  By always adding one, we
        can properly distinguish rule generalities in these cases.
        """
        areas = []
        for rule in ruleset:
            area = 1
            for c in range(self.numConditions):
                area *= (abs(rule[c*2] - rule[c*2+1]) + 1) # +1 See above
            areas.append(area)
        return areas


    def execute(self, inputValues):
        """
        Selects the appropriate rule and fires it.  The output is returned.
        Python version.
        """
        assert len(inputValues) == self.num_inputs
        #print("type(inputValues) =", type(inputValues))
        #print("inputValues =", inputValues)
        #print("self.memRegs =", self.memRegs)
        allInput = inputValues + self.memRegs
        bestMatchScore = sys.float_info.max  # Minimize
        matchList = []       # indices of rules
        relevant_neighbors = []  # used for interpolation

        # Build match list.  Find all rules that match the input.
        # If 'inexact' matches (partial match, nearest neighbor) requested,
        # then consider those too.
        for r,rule in enumerate(self.ruleset):
            #print("rule #", r, "=", rule)
            numConditionMatches = 0
            distance = 0
            for c in range(self.numConditions):
                #print("condition #", c, "=", (rule[c*2], rule[c*2+1]))
                #print("input #", c ,"=", allInput[c])
                # It might be a good idea to normalize these calculations,
                # especially if the ranges are very different.
                diff1 = rule[c*2] - allInput[c]
                diff2 = rule[c*2+1] - allInput[c]

                if diff1 * diff2 <= 0:      # Check sign
                    diff = 0                # Within the range
                    numConditionMatches += 1
                else:
                    diff = min(abs(diff1), abs(diff2)) 
                    
                distance += diff * diff   # Distance w/o sqrt

                        
                #print("diff1, diff2, diff, matchScore =", \
                #       diff1, diff2, diff, matchScore)

            #print("matchScore =", matchScore)

            # If doing interpolation, record all relevant neighboring rules
            if self.nearest_neighbor == True and self.nn_interpolate == True:
                relevant_neighbors.append([math.sqrt(distance), rule])

            # Should we record this rule, or move on to the next?
            if self.nearest_neighbor == True \
               or (self.partial_matching and numConditionMatches > 0) \
               or numConditionMatches == self.numConditions:  # exact match

                # Figure out what to use for the match score
                if self.nearest_neighbor:
                    matchScore = distance
                else:
                    # Make sure better matches have lower values
                    matchScore = self.numConditions - numConditionMatches

                # Is this rule (one of) the best?
                if matchList == [] or matchScore < bestMatchScore:
                    bestMatchScore = matchScore
                    matchList = [r]
                elif matchScore == bestMatchScore:
                    matchList.append(r)

        #print("matchList =", matchList)

        if not matchList:   # No matching rules
            print("output =", None)
            return None

        # Conflict resolution
        # From existing matches, choose rule(s) with the best (lowest) rank.
        if bestMatchScore == 0:
            bestRank = min([self.ruleRanks[i] for i in matchList])
            #print("bestRank =", bestRank)

            # Cull the matchList based on priority.
            matchList = [i for i in matchList if self.ruleRanks[i] == bestRank]

        #print("matchList =", matchList)

        # More conflict resolution
        # A common approach is to select the output that has the most rules
        # advocating it (i.e. vote).  A simpler approach is to just pick a
        # rule randomly.  For now we'll just pick randomly.
        winner = random.choice(matchList)

        # "Fire" the rule.
        #print("Firing:", winner, self.ruleset[winner])
        win_rule = self.ruleset[winner]
        out_start = self.numConditions*2
        mem_start = out_start + self.num_outputs
        mem_end = mem_start + self.numMemory

        output = win_rule[out_start:mem_start]
        self.memRegs = win_rule[mem_start:mem_end]

        # If doing interpolation, calculate the interpolated values
#        if self.nearest_neighbor == True and self.nn_interpolate == True:
#           d = [neighbor[0] for neighbor in relevant_neighbors]
#           if any([d_i == 0.0 for d_i in d]):
#               d_inv = [1.0 if d_i == 0.0 else 0.0 for d_i in d]
#           else:
#               d_inv = [1/d_i for d_i in d]
#               #d_inv = [1/(d_i * d_i) for d_i in d]
#
#           c = [d_inv_i / sum(d_inv) for d_inv_i in d_inv]
#           if not 0.95 < sum(c) < 1.05:
#               print("c =", c)
#           v = [n[1][out_start] for n in relevant_neighbors]
#           output = [sum([v_i * c_i for v_i, c_i in zip(v, c)])]
#           #print("d =", d)
#           #print("d_inv =", d_inv)
#           #print("c =", c)
#           #print("v =", v)
#           #print("output =", output)
#           #exit()

        if self.nearest_neighbor == True and self.nn_interpolate == True:
            scipy.interpolate.NearestNDInterpolator(x, y)

#        if self.numMemory == 0:
#            self.memRegs = []
#            output = self.ruleset[winner][-self.num_outputs:]
#        else:
#            self.memRegs = self.ruleset[winner][-self.numMemory:]
#            output = self.ruleset[winner][-self.num_outputs - self.numMemory : 
#                                          -self.numMemory]

        #print("output =", output)
        return output
        #return output, self.memRegs


#############################################################################
#
# PittBaseCoder
#
#############################################################################
class PittBaseCoder(BaseCoder):
    """
    A base class encoder for Pitt approach style rule sets.
    The create_random_genome() should be defined by sub-classes.
    """
    def __init__(self, min_rules, max_rules, num_inputs, num_outputs, \
                 init_mem = [], ruleInterpClass = RuleInterp,
                 partial_matching = False, nearest_neighbor = False,
                 use_alternate_ranks = False, nn_interpolate = False):
        super().__init__()

        self.min_rules = min_rules
        self.max_rules = max_rules

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.init_mem = init_mem

        self.partial_matching = partial_matching
        self.nearest_neighbor = nearest_neighbor
        self.use_alternate_ranks = use_alternate_ranks

        self.nn_interpolate = nn_interpolate

        super().__init__()
        self.priorityMetric = None  # XXX Fix me!
        self.ruleInterpClass = ruleInterpClass

        self.numConditions = num_inputs + len(init_mem)
        self.numActions = num_outputs + len(init_mem)


    def decode_genome(self, genome):
        raise NotImplementedError
        # Just make sure you return and executable object when you implement
        # this.


    def create_random_genome(self):
        raise NotImplementedError



#############################################################################
#
# PittBoundsCoder
#
#############################################################################
class PittBoundsCoder(PittBaseCoder):
    """
    Uses another encoder for decoding rules and generating random rules.
    """
    def __init__(self, rule_coder, min_rules, max_rules, \
                 num_inputs, num_outputs, init_mem=[], \
                 ruleInterpClass = RuleInterp,
                 partial_matching = False, nearest_neighbor = False,
                 use_alternate_ranks = False, nn_interpolate = False):
        super().__init__(min_rules, max_rules, num_inputs, num_outputs,
                         init_mem, ruleInterpClass, partial_matching,
                         nearest_neighbor, use_alternate_ranks, nn_interpolate)
        self.rule_coder = rule_coder
        self.priorityMetric = None  # XXX Fix me!


    def create_random_genome(self):
        num_rules = random.randrange(self.max_rules - self.min_rules + 1) \
                   + self.min_rules
        genome = [self.rule_coder.create_random_genome() for i in \
                                                              range(num_rules)]
        return genome


    def decode_genome(self, genome):
        float_genome = [self.rule_coder.decode_genome(rule) for rule in genome]
        return self.ruleInterpClass(float_genome, self.num_inputs, \
                                    self.num_outputs, self.init_mem, \
                                    self.partial_matching, \
                                    self.nearest_neighbor, \
                                    self.use_alternate_ranks, \
                                    self.nn_interpolate)



#############################################################################
#
# PittPointCoder
#
#############################################################################
class PittPointCoder(PittBoundsCoder):
    """
    Uses a single point instead of a hyper-rectangle to represent a rule.  The
    standard rule interpreter should work fine as long as both corners are the
    same for all hyper-rectangles.

    It really doesn't make sense to use this Coder without
    nearest_neighbor=True, except maybe in special cases.
    """
    # All this just to make nearest_neighbor=True the default
    def __init__(self, rule_coder, min_rules, max_rules, \
                 num_inputs, num_outputs, init_mem=[], \
                 ruleInterpClass = RuleInterp, \
                 partial_matching = False, nearest_neighbor = True, \
                 use_alternate_ranks = False, nn_interpolate = False):
        super().__init__(rule_coder, min_rules, max_rules,
                         num_inputs, num_outputs, init_mem, ruleInterpClass,
                         partial_matching, nearest_neighbor,
                         use_alternate_ranks, nn_interpolate)

    def point2box_rule(self, rule):
        """
        Converts a rule where the condition is defined as a point to one where
        the condition is defined as a box.
        """
        newRule = rule[0:0]  # maintain type
        for i in range(0, self.num_inputs):
            newRule += rule[i:i+1]   # maintain type
            newRule += rule[i:i+1]
        newRule += rule[self.num_inputs:]
        return newRule


    def decode_genome(self, genome):
        float_genome = [self.point2box_rule(self.rule_coder.decode_genome(rule))
                        for rule in genome]
        return self.ruleInterpClass(float_genome, self.num_inputs, \
                                    self.num_outputs, self.init_mem, \
                                    self.partial_matching, \
                                    self.nearest_neighbor, \
                                    self.use_alternate_ranks, \
                                    self.nn_interpolate)


#############################################################################
#
# class PittUniformCrossover
#
#############################################################################
class PittUniformCrossover(BaseOp):
    """
    One of the two genomes is traversed and with some probability (p_xfer) a
    gene will be copied to the opposite offspring (i.e. Mother to son),
    otherwise it will be to the corresponding offspring (i.e. Mother to
    daughter).  The same procedure is then performed with the other "parent".

    In other words, instead of traversing both parents simultaneously and
    swapping genes genes occasionally, each parent is traversed separately and
    genes are transfered occasionally.  On average an equal number of
    transfers will occur, but in practice the number of genes transfered will
    often differ.  This allows genomes to grow or shrink in size.

    Note: I've never seen anyone else actually use an operator like this, but
          it seemed like a logical analogy to the other standard Pitt
          crossovers.
    """
    def __init__(self, provider, p_cross, p_xfer=0.5):
        super().__init__(provider=provider)
        self.p_cross = p_cross  # probability of performing crossover at all
        self.p_xfer = p_xfer    # probability of transfering a gene to other


    def recombine(self, child1, child2):
        copy_genome1 = child1.genome[:]
        copy_genome2 = child2.genome[:]
        new_genome1 = child1.genome[0:0]  # empty sequence - maintain type
        new_genome2 = child2.genome[0:0]

        for i in range(len(copy_genome1)):
            g = copy_genome1[i:i+1]
            if random.random() < self.pSwap:
                new_genome2 += g
            else:
                new_genome1 += g

        for i in range(len(copy_genome2)):
            g = copy_genome2[i:i+1]
            if random.random() < self.pSwap:
                new_genome1 += g
            else:
                new_genome2 += g

        child1.genome = new_genome1;
        child2.genome = new_genome2;

        return(child1, child2)


    def generator(self):
        while 1:
            mother = self.provider.pull()
            father = self.provider.pull()
            if random.random() <= self.p_cross:
                daughter_genome = []
                son_genome = []
                for i in range(len(mother.genome)):
                    if random.random() <= self.p_xfer:
                        son_genome.append(mother.genome[i])
                    else:
                        daughter_genome.append(mother.genome[i])

                for i in range(len(father.genome)):
                    if random.random() <= self.p_xfer:
                        daughter_genome.append(father.genome[i])
                    else:
                        son_genome.append(father.genome[i])

                mother.genome = daughter_genome
                father.genome = son_genome

            # XXX Should I randomize the return order?  Ken would say yes.
            if len(mother.genome) > 0:
                yield mother
            if len(father.genome) > 0:
                yield father

