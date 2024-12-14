#!/usr/bin/env python

"""
problems.py: defines common problems for Eclypse
"""

import random
import copy
import numpy

import gym

#############################################################################
#
# OpenAIGymProblem
#
#############################################################################
class OpenAIGymProblem():
    def __init__(self, sim_name, max_timestep, num_episodes=1):
        self.sim_name = sim_name
        self.env=None
        self.env = gym.make(sim_name)
        self.max_timestep = max_timestep
        self.num_episodes = num_episodes  # episodes per eval

    def __del__(self):
        if self.env:
            self.env.close()  # Is this even necessary?

    def input_bounds(self):
        # XXX I don't think this will work in all cases!
        obs = self.env.observation_space
        bounds = zip(obs.low, obs.high)
        return bounds

    def output_bounds(self):
        return([ [0,1] ])   # XXX Fix this!

    def evaluate(self, exec_phenome):
        ep_rewards = [0] * self.num_episodes
        for episode in range(self.num_episodes):
            reward_total = 0
            observation = self.env.reset()
            for timestep in range(self.max_timestep):
                #self.env.render()
                output = exec_phenome.execute(observation.tolist())
                if len(output) == 1:
                    # XXX Should parameterize the rounding
                    action = int(output[0] > 0.5)  # cart-pole
                else:
                    action = output               # lunar lander
                observation, reward, done, info = self.env.step(action)
                reward_total += reward
                if done:
                    break  # Go to next episode
            ep_rewards[episode] = reward_total

        fitness = numpy.mean(ep_rewards)
        return fitness

    def better_than(self, fit1, fit2):
        return fit1 > fit2

    def equivalent_to(self, fit1, fit2):
        return fit1 == fit2


