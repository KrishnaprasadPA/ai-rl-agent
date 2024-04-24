from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util
import numpy as np
import collections
import math


class ActorCriticAgent(ReinforcementAgent):
    def __init__(self, epsilon=0.05, gamma=0.75, alpha=0.01, critic_alpha=0.01, extractor='SimpleExtractor', **args):
        ReinforcementAgent.__init__(self, **args)
        self.featExtractor = util.lookup(extractor, globals())()
        self.gamma = gamma
        self.critic_alpha = critic_alpha
        self.actor_theta = collections.defaultdict(lambda: 0)
        self.critic_theta = collections.defaultdict(lambda: 0)
        self.transition_history = []
        self.actions = ['North', 'West', 'South', 'East']

    def getQValue(self, state, action):
        features = self.featExtractor.getFeatures(state, action)
        value = 0
        for key in features.keys():
            value += self.actor_theta[key] * features[key]
        return value

    def getVValue(self, state, action):
        # Compute value of state-action pair using critic's parameters
        features = self.featExtractor.getFeatures(state, action)
        value = 0
        for key in features.keys():
            value += self.critic_theta[key] * features[key]
        return value

    def actionProbs(self, state):
        legalActions = self.getLegalActions(state)
        max_q_value = max([self.getQValue(state, action) for action in legalActions])
        sum_exp = 0
        probs = {}
        for action in legalActions:
            q_value = self.getQValue(state, action)
            exp_q_value = math.exp(q_value - max_q_value)  # Shift by max_q_value to prevent overflow
            probs[action] = exp_q_value
            sum_exp += exp_q_value
        for action in legalActions:
            probs[action] /= sum_exp  # Normalize by the sum of exponentials
        return probs


    def updateCritic(self, state, action, reward, nextState):
        # Update critic's parameters using TD-error
        difference = reward + self.gamma * self.getVValue(nextState, action) - self.getVValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for key in features.keys():
            self.critic_theta[key] += self.critic_alpha * difference * features[key]



    def updateActor(self, state, action, reward, nextState):
        # Update actor's parameters using advantage (TD-error)
        difference = reward + self.gamma * self.getVValue(nextState, action) - self.getVValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for key in features.keys():
            self.actor_theta[key] += self.alpha * difference * features[key]

    def update(self, state, action, nextState, reward):
        """
        This function is called after observing a transition and reward.
        """
        self.transition_history.append((state, action, reward, nextState))

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        exploration = util.flipCoin(self.epsilon)
        if exploration:
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)
        self.doAction(state, action)
        return action

    def getPolicy(self, state):
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None
        
        probs = self.actionProbs(state)
        distribution = []
        for action, prob in probs.items():
            distribution.append((prob, action))
        action = util.chooseFromDistribution(distribution)
        return action

    def final(self, state):
        G = 0
        for t, (state, action, reward, nextState) in enumerate(reversed(self.transition_history)):
            G = G * self.gamma + reward
            self.updateCritic(state, action, reward, nextState)
            self.updateActor(state, action, G, nextState)
        self.transition_history = []
        ReinforcementAgent.final(self, state)

        








