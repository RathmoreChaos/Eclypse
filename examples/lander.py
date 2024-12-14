#!/usr/bin/env python

"""
lander.py: Displays a resulting lunar lander control system.
"""

import random
import sys

import gym

from numpy import inf

from eclypse.ind import Individual
from eclypse.problems import BaseProblem
from eclypse.coders import FloatCoder
from eclypse.select import TournamentSelection, select_cmp_default,\
              select_cmp_lexicographic_parsimony, select_cmp_penalty_parsimony
from eclypse.ops import Clone, GaussianMutation, UniformCrossover, Evaluate
from eclypse.survive import Elitism
from eclypse.exec.pitt import PittPointCoder, PittUniformCrossover
from eclypse.env.agent.gym import OpenAIGymProblem


#############################################################################
#
# lander
#
#############################################################################
def lander():
    num_inputs=8
    num_outputs=2
    init_ranges = [[-1.0, 1.0]] * num_inputs + [[-1.0, 1.0]] * num_outputs
    min_rules = 20
    max_rules = 20

    sim_name = "LunarLanderContinuous-v2"
    #sim_name = "LunarLander-v2"
    n_steps = 100
#    problem = OpenAIGymProblem(sim_name, max_timestep=n_steps, num_episodes=5)
#
#    #print(list(problem.input_bounds()))
#    #exit()
#
    rule_coder = FloatCoder(init_ranges)
    coder = PittPointCoder(rule_coder, min_rules, max_rules, \
                           num_inputs, num_outputs)
#    parsimony = select_cmp_default
#    #parsimony = select_cmp_lexicographic_parsimony
#    #parsimony = select_cmp_penalty_parsimony(penalty=10.0, problem=problem)
#
#    pipeline = TournamentSelection(tournament_size=2, select_cmp=parsimony)
#    pipeline = Clone(pipeline)
#    pipeline = UniformCrossover(pipeline, p_cross=0.8, p_swap=0.2)
#    #pipeline = PittUniformCrossover(pipeline, p_cross=0.8, p_xfer=0.2)
#    #pipeline = GaussianMutation(pipeline, sigma=0.1)
#    pipeline = GaussianMutation(pipeline, sigma=0.01)
#    pipeline = Elitism(pipeline, num_elite=1, select_cmp=parsimony)
#    pipeline = Evaluate(pipeline)
#
#    bog = bsf = None  # Best of generation, best so far
#    bsf_last_fitness = -inf
#
#    # Generate initial population
#    pop_size = 100
#    population = [Individual(problem, coder) for _ in range(pop_size)]
#    print("\nGeneration", 0)
#    bog = population[0]
#    bog.evaluate()
#    for ind in population:
#        ind.evaluate()
#        #print(ind)
#        if ind.better_than(bog):
#            bog = ind
#        print(len(ind.genome), ind.fitness)
#    bsf = bog
#    bsf_last_fitness = bsf.fitness
#    print("\nbsf =", bsf, len(bsf.genome), bsf.fitness)
#
#    # Run the EA
#    num_generations = 50
#    for generation in range(1, num_generations+1):
#        pipeline.new_generation(population)
#        new_population = [pipeline.pull() for _ in range(pop_size)]
#        bog = population[0]
#
#        # Print the population
#        print("\nGeneration", generation)
#        for ind in new_population:
#            if ind.better_than(bog):
#                bog = ind
#            #print(ind)
#            print(len(ind.genome), ind.fitness)
#
#        if bog.better_than(bsf) or\
#           (bog.equivalent_to(bsf) and parsimony(bog, bsf) == 1):
#            bsf = bog
#
#        print("bsf =", len(bsf.genome), bsf.fitness)
#        population = new_population
#        bsf_last_fitness = bsf.fitness

    rules = \
[[0.2999749064615262, 0.10300959843263371, 0.45521778351344616,
0.734485310194382, 0.011452320310800087, -0.421812325197331,
0.8313921380118904, 0.34092344670982566, -0.325644982444881,
0.10821349757572461], [0.3959890462124541, 0.06370782257099553,
-0.2638255177289225, -0.9451027217417689, 0.21878283924839206,
0.17206908662748419, -0.33641265252586267, -0.45426894694815956,
0.9092880408427364, 0.11057294559743017], [0.27804961660280797,
0.08968678931364488, 0.7848815296240618, -0.16821763052583916,
-0.5879427348396729, 0.5907451052196145, -0.7315015571306662,
0.8348262361874741, -0.383466112626787, -0.6731700756589449],
[0.27755611973665517, -0.9899236204154408, 0.6460324812079004,
-0.6116788445953985, -0.6506553537832362, 0.6260667816675362,
-0.6400362575155116, 0.19396861607912555, 0.24884202570838795,
-0.34793996465615507], [-0.9546506988532005, 0.22417512814213766,
-0.41990479914512363, 1.0801054802476073, -0.2395463004790922,
-0.040152134773292415, -0.47876961375492283, -0.43290284209547636,
-0.7006884269667932, -0.24443341264017107], [0.9322084714534113,
-0.5490301514487828, 0.7297971619061411, -0.898070433036256,
0.5255065164631897, -0.10153671841376843, -1.0662111123591513,
0.4035855825202811, -0.002478803616611191, 0.19984107313050076],
[0.8844819848468011, -0.3210696741498751, 0.24005172957249507,
-1.017097876096451, -0.20617320585116536, -0.45266719854861587,
-0.08964753473063694, 0.5410613340268328, 0.009113638889057864,
-1.0201702439368303], [0.5847091585350819, 0.37716572532727827,
-0.7036617145562042, 0.7567123231916053, 0.38780458744339613,
-0.8612289467244139, 0.242363491702088, -0.7118322219575701,
0.8228310077649366, 0.9646341085407952], [0.5981286155011014,
0.5948936222350018, -0.1717835501986948, -0.7038338119578746,
-0.875846894424746, -0.9640707645935193, 0.6045836597553642,
0.12004625812946756, 0.8685866307117519, -0.7976501175363987],
[-0.6252159395896218, 0.407930969932253, -0.37824185959954015,
0.4348051275008999, 0.28251307190908886, 1.018178866149931,
1.0386132825688543, -0.9968156167564639, 0.20747287125838804,
-0.017122646591918683], [-0.5930352295783161, 0.2986419086921177,
0.41456204239958644, 0.07904724947252437, 1.0583119338214277,
-0.38234235041731945, 0.8051860380215209, 0.8726266775094604,
-0.906532969400509, -0.7384000962937516], [0.7031417094597399,
0.03153324456763341, -0.5649552623363219, 0.24674955885859132,
0.9076885829814982, 0.5449848748252958, 0.8569206324223668,
-0.5624273510338614, -0.15691027930242812, 0.5400255378342956],
[0.1418856286044718, 0.9439132763107476, -0.32386986511291593,
-0.3865573298563577, 0.04130422519977595, 0.5794430114922445,
-0.6730114614569995, -0.6372306299109438, -0.6305356982455114,
0.6161239950112527], [-1.019115855590343, -0.3515899045418135,
-0.4017791350995056, 0.5139829145220444, -0.4530362082744387,
0.2701852171566839, -0.6467237287382901, 0.6011560715711743,
-0.8491808267729957, -0.8504926275789515], [0.46314850604149504,
-0.27699561564773245, -0.3221724764141098, -0.40250988449790687,
0.28595356463881194, 0.3498347840986311, -0.16014576599142738,
0.667833864481139, -0.40704789490256993, -0.47405945332378885],
[-0.570208765768026, 0.37795162855647474, -0.8398323842422983,
-0.3368551077074108, -0.4402333837010225, 0.6694085997766221,
0.7964125520746916, 0.3515512250762545, -0.5779469245834703,
0.7833012471425489], [0.8824983197881885, -0.41399163853746823,
0.24270708028096344, -0.2780660719861537, 0.8089560001511764,
-0.9843405112804678, 1.1511289892116572, 0.4495539786990717,
0.952792410614807, -0.20634851803294132], [0.46537441282856734,
-0.5892679605678912, 0.21224119242069409, -0.6516219626865964,
-0.20066677216615517, 0.5584085007932991, 0.7258922986512365,
0.05786875817959312, 0.2916133142196026, -0.8293331276912455],
[-0.48740671714400546, 0.9791245810121169, 0.3190293198645255,
-0.5970897207125138, 0.7302413765379239, -0.29211824924954605,
-0.21886319566075696, -0.1895846534724552, -0.5401795656144861,
0.1526484444287203], [-0.7954952328748454, -0.8231516593193978,
0.2877930308463328, 0.51984569648171, -0.6563581703525321,
-0.48854908638558775, -0.01657910698210361, -0.7469275023347717,
-0.7051018696605849, 0.10893345133386723]]

    print(rules)
    exec_phenome = coder.decode_genome(rules)

    # Show the solution in the simulation
    env = gym.make(sim_name)

    observation = env.reset()
    for timestep in range(n_steps * 10):
        env.render()
        action = exec_phenome.execute(observation.tolist())
        observation, reward, done, info = env.step(action)
        if done:
            break

    env.close()


#############################################################################
#
# command line execute
#
#############################################################################
if __name__ == "__main__":
    print(sys.version)
    lander()

