#! /usr/bin/env python

"""
test_ruleInterp.py: tests the Pittsburgh Approach rule interpreter.
"""

from eclypse.exec.pitt import RuleInterp

# Binary w/ wildcards  (bounding box, LS-1 style?)
# Float w/ Bounding box
# Float w/ Nearest Neighbor


#############################################################################
#
# test_RuleInterp_BinaryWildcard
#
#############################################################################
def test_RuleInterp_BinaryWildcard():
    """
    Test the rule interpreter on a binary genome using wildcards for
    generalization.  Wildcards in rules are implemented using 2 bits 
    per input.  If both bits in a condition are the same (0 or 1) then and
    exact match to the input is defined.  If both condition bits are
    different, then it is a wildcard and will match both zeros and ones.
    This is the approach that Smith used in his LS-1 system.  It also
    turns out that this is essentially the same as a bounding box in binary.

    Note that the genome does not need to be setup this way.  If you would
    prefer to have genes with 3 allele values (0, 1 and *) then you're welcome
    to do that in your EA.  You just need to translate to this form when you
    create a rule interpreter.
    """
    #             conditions    act
    ruleset = [[0,0, 0,0, 0,0,  0],
               [0,0, 0,0, 1,1,  1],
               [1,1, 0,1, 0,0,  1]]

    num_inputs = 3
    num_outputs = 1

    interp = RuleInterp(ruleset, num_inputs, num_outputs)

    assert(interp.execute([0,0,0]) == [0])
    assert(interp.execute([0,0,1]) == [1])
    assert(interp.execute([1,0,0]) == [1])
    assert(interp.execute([1,1,0]) == [1])
    assert(interp.execute([1,1,1]) is None)



#############################################################################
#
# test_RuleInterp_BinaryWildcardPM
#
#############################################################################
def test_RuleInterp_BinaryWildcardPM():
    """
    Test the rule interpreter on a binary genome using wildcards for
    generalization and partial matching.
    """
    #             conditions    act
    ruleset = [[0,0, 0,0, 0,0,  0],
               [1,1, 1,1, 1,1,  1]]

    num_inputs = 3
    num_outputs = 1

    interp = RuleInterp(ruleset, num_inputs, num_outputs, partial_matching=True)

    assert(interp.execute([0,0,0]) == [0])
    assert(interp.execute([0,0,1]) == [0])
    assert(interp.execute([0,1,0]) == [0])
    assert(interp.execute([1,0,0]) == [0])
    assert(interp.execute([1,1,1]) == [1])
    assert(interp.execute([1,1,0]) == [1])
    assert(interp.execute([1,0,1]) == [1])
    assert(interp.execute([0,1,1]) == [1])



#############################################################################
#
# test_RuleInterp_BinaryMemory
#
#############################################################################
def test_RuleInterp_BinaryMemory():
    """
    Test the rule interpreter on a binary genome using wildcards for
    generalization and memory registers.
    """
    #          cond  mem act mem_out
    ruleset = [[0,0, 0,0, 0, 0],
               [0,0, 1,1, 1, 1],
               [1,1, 0,0, 1, 1],
               [1,1, 1,1, 0, 0]]

    num_inputs = 1
    num_outputs = 1
    init_mem = [0]

    interp = RuleInterp(ruleset, num_inputs, num_outputs, init_mem)

    output  = interp.execute([1])
    output += interp.execute([1])
    output += interp.execute([0])
    output += interp.execute([1])
  
    mem = interp.memRegs

    print(" output =", output)
    print(" memRegs =", mem)
    assert(output == [1, 0, 0, 1])
    assert(mem == [1])


#############################################################################
#
# test_RuleInterp_FloatBounds
#
#############################################################################
def test_RuleInterp_FloatBounds():
    """
    Test the rule interpreter on a float genome bounding box generalization.
    """
    #             conditions    act
    ruleset = [[0.0, 5.0,  0.0, 5.0,  0],
               [0.0, 5.0,  5.0,10.0,  1],
               [5.0, 0.0,  5.0,10.0,  1],
               [5.0,10.0,  0.0, 5.0,  1],
               [5.0,10.0,  5.0,10.0,  0]]

    num_inputs = 2
    num_outputs = 1

    interp = RuleInterp(ruleset, num_inputs, num_outputs)

    assert(interp.execute([2.5, 2.5]) == [0])
    assert(interp.execute([2.5, 7.5]) == [1])
    assert(interp.execute([7.5, 2.5]) == [1])
    assert(interp.execute([7.5, 7.5]) == [0])
    assert(interp.execute([-1.0, -1.0]) is None)
    assert(interp.execute([11.0, 11.0]) is None)

    assert(interp.execute([5.0, 5.0]) == [0])
    assert(interp.execute([5.0, 5.0]) == [0])
    assert(interp.execute([5.0, 5.0]) == [0])
    assert(interp.execute([5.0, 5.0]) == [0])


#############################################################################
#
# test_RuleInterp_FloatBoundsPM
#
#############################################################################
def test_RuleInterp_FloatBoundsPM():
    """
    Test the rule interpreter on a float genome bounding box generalization
    with partial matching.
    """
    #             conditions    act
    ruleset = [[0.0, 5.0,  0.0, 5.0,  0],
               [5.0,10.0,  5.0,10.0,  1]]

    num_inputs = 2
    num_outputs = 1

    interp = RuleInterp(ruleset, num_inputs, num_outputs, partial_matching=True)

    assert(interp.execute([2.5, 2.5]) == [0])
    assert(interp.execute([7.5, 7.5]) == [1])

    assert(interp.execute([2.5, 7.5]) in [[0], [1]])
    assert(interp.execute([7.5, 2.5]) in [[0], [1]])

    assert(interp.execute([5.0, 5.0]) == [0])
    assert(interp.execute([5.0, 5.0]) == [0])
    assert(interp.execute([5.0, 5.0]) == [0])
    assert(interp.execute([5.0, 5.0]) == [0])

    assert(interp.execute([-1.0, 2.5]) == [0])
    assert(interp.execute([2.5, -1.0]) == [0])
    assert(interp.execute([-1.0, 7.5]) == [1])
    assert(interp.execute([7.5, -1.0]) == [1])
    assert(interp.execute([11.0, 2.5]) == [0])
    assert(interp.execute([2.5, 11.0]) == [0])
    assert(interp.execute([11.0, 7.5]) == [1])
    assert(interp.execute([7.5, 11.0]) == [1])

    assert(interp.execute([-1.0, -1.0]) is None)
    assert(interp.execute([-1.0, 11.0]) is None)
    assert(interp.execute([11.0, -1.0]) is None)
    assert(interp.execute([11.0, 11.0]) is None)


#############################################################################
#
# test_RuleInterp_FloatBoundsNN
#
#############################################################################
def test_RuleInterp_FloatBoundsNN():
    """
    Test the rule interpreter on a float genome bounding box generalization
    with Nearest Neighbor.
    """
    #             conditions    act
    ruleset = [[0.0, 5.0,  0.0, 5.0,  0],
               [0.0, 5.0,  5.0,10.0,  1],
               [5.0,10.0,  0.0, 5.0,  1],
               [5.0,10.0,  5.0,10.0,  0]]

    num_inputs = 2
    num_outputs = 1

    interp = RuleInterp(ruleset, num_inputs, num_outputs, nearest_neighbor=True)

    assert(interp.execute([2.5, 2.5]) == [0])
    assert(interp.execute([2.5, 7.5]) == [1])
    assert(interp.execute([7.5, 2.5]) == [1])
    assert(interp.execute([7.5, 7.5]) == [0])

    assert(interp.execute([5.0, 5.0]) == [0])
    assert(interp.execute([5.0, 5.0]) == [0])
    assert(interp.execute([5.0, 5.0]) == [0])
    assert(interp.execute([5.0, 5.0]) == [0])

    assert(interp.execute([-1.0, 2.5]) == [0])
    assert(interp.execute([2.5, -1.0]) == [0])
    assert(interp.execute([-1.0, 7.5]) == [1])
    assert(interp.execute([2.5, 11.0]) == [1])
    assert(interp.execute([11.0, 2.5]) == [1])
    assert(interp.execute([7.5, -1.0]) == [1])
    assert(interp.execute([11.0, 7.5]) == [0])
    assert(interp.execute([7.5, 11.0]) == [0])

    assert(interp.execute([-1.0, -1.0]) == [0])
    assert(interp.execute([-1.0, 11.0]) == [1])
    assert(interp.execute([11.0, -1.0]) == [1])
    assert(interp.execute([11.0, 11.0]) == [0])


#############################################################################
#
# test_RuleInterp_FloatNN
#
#############################################################################
def test_RuleInterp_FloatNN():
    """
    Test the rule interpreter on a float genome where rules are points
    with Nearest Neighbor generalization.  This is accomplished by making
    the upper and lower bounds identical on each dimension.
    """
    #             conditions    act
    ruleset = [[2.5,2.5,  2.5,2.5,  0],
               [2.5,2.5,  7.5,7.5,  1],
               [7.5,7.5,  2.5,2.5,  1],
               [7.5,7.5,  7.5,7.5,  0]]

    num_inputs = 2
    num_outputs = 1

    interp = RuleInterp(ruleset, num_inputs, num_outputs, nearest_neighbor=True)

    assert(interp.execute([2.5, 2.5]) == [0])

    assert(interp.execute([2.0, 2.0]) == [0])
    assert(interp.execute([2.0, 8.0]) == [1])
    assert(interp.execute([8.0, 2.0]) == [1])
    assert(interp.execute([8.0, 8.0]) == [0])

    assert(interp.execute([5.0, 5.0]) in [[0], [1]])
    assert(interp.execute([5.0, 5.0]) in [[0], [1]])
    assert(interp.execute([5.0, 5.0]) in [[0], [1]])
    assert(interp.execute([5.0, 5.0]) in [[0], [1]])

    assert(interp.execute([-1.0, 2.5]) == [0])
    assert(interp.execute([2.5, -1.0]) == [0])
    assert(interp.execute([-1.0, 7.5]) == [1])
    assert(interp.execute([2.5, 11.0]) == [1])
    assert(interp.execute([11.0, 2.5]) == [1])
    assert(interp.execute([7.5, -1.0]) == [1])
    assert(interp.execute([11.0, 7.5]) == [0])
    assert(interp.execute([7.5, 11.0]) == [0])

    assert(interp.execute([-1.0, -1.0]) == [0])
    assert(interp.execute([-1.0, 11.0]) == [1])
    assert(interp.execute([11.0, -1.0]) == [1])
    assert(interp.execute([11.0, 11.0]) == [0])


#############################################################################
#
# makeMap
#
# Helper function ???
#
#############################################################################
def makeMap(interp):
    f = open("map.dat", "w")
    res = 0.0025
    y = 0.0
    while y <= 1.0:
        x = 0.0
        while x <= 1.0:
            if interp.execute([x, y]) == [1]:
                f.write(str(x) + " " + str(y) + "\n")
            x += res
        y += res
    f.close()


#############################################################################
#
# test_DontKnow
#
# I'm not sure what this was for. ?
#
#############################################################################
def tst_DontKnow():
    #ruleset = [[0.0,0.6, 0.0,0.4, 0],
    #           [0.4,1.0, 0.6,1.0, 1]]
    ruleset = [[0.0,0.0, 0.0,0.0, 0.0],
               [1.0,1.0, 1.0,1.0, 1.0]]

    num_inputs = 2
    num_outputs = 1

    interp = cRuleInterp(ruleset, num_inputs, num_outputs)
    #assert(interp.execute([0.61, 0.0]) == [0])
    #assert(interp.execute([0.5, 0.6]) == [1])

    print("Writing map file...")
    makeMap(interp)

    print("Passed")



#############################################################################
#
# Command line execute
#
#############################################################################
if __name__ == '__main__':
    test_RuleInterp_BinaryMemory()



