from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util
import numpy as np
import collections
import math

class ReinforceMCPGAgent(ReinforcementAgent):
    def __init__(self, epsilon=0.05, gamma=0.85, alpha=0.1, extractor='SimpleExtractor', **args):
        ReinforcementAgent.__init__(self, **args)
        self.featExtractor = util.lookup(extractor, globals())()
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        #args['numTraining'] = numTraining
        self.gamma = args['gamma']
        self.theta = collections.defaultdict(lambda : 0)
        self.transition_history = []
        self.actions = ['North','South' ,'East','West','Stop']

    
    def getQValue(self, state, action):
        features = self.featExtractor.getFeatures(state, action)
        value = 0
        for key in features.keys():
            value += self.theta[key] * features[key]
        return value
    
    def actionProbs(self, state):
        legalActions = self.getLegalActions(state)
        sum = 0
        probs = {}
        for action in legalActions: 
            value = math.exp(self.getQValue(state, action))
            probs[action] = value
            sum += value
        for action in legalActions:
            if sum != 0:
                probs[action] /= sum
        return probs
    
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

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        exploration = util.flipCoin(self.epsilon)
        if exploration:
          action = random.choice(legalActions)
        else:
          action = self.getPolicy(state)
        self.doAction(state,action)
        return action

    def update(self,state, action, nextState, reward):
        self.transition_history.append([state, action, reward])


    def computeGvalues(self):
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)  # Insert at the beginning to maintain the order of returns
        for t in range(len(returns)):
            returns[t] = returns[t] * (self.gamma ** t)
        return returns

    def log_gradient(self, state, action):
        action_probs = self.actionProbs(state)
        features = self.featExtractor.getFeatures(state, action)
        grad_log_policy = self.calculate_log_policy_gradient(features, action_probs, state, action)

        return grad_log_policy

    def calculate_log_policy_gradient(self, features, action_probs, state, action):
        grad_log_policy = {}

        for feature, value in features.items():
            # Calculate the gradient of log policy for each feature
            grad_log_policy[feature] = value * (1 - action_probs[action])

            # Subtract the weighted sum of features for all actions
            for a, prob in action_probs.items():
                if a != action:
                    features_a = self.featExtractor.getFeatures(state, a)
                    grad_log_policy[feature] -= features_a.get(feature, 0) * prob

        return grad_log_policy

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        print("ep", self.episodesSoFar)
        G = 0
        for t in reversed(range(0, len(self.transition_history))):
          G = G * self.gamma + self.transition_history[t][2]
          
          log_prob = self.log_gradient(self.transition_history[t][0], self.transition_history[t][1])
          for feature, log_grad in log_prob.items():
            if feature not in self.theta:
                self.theta[feature] = 0
            self.theta[feature] += self.alpha * G * log_grad * np.power(self.gamma, t)
        
        self.transition_history = [] 
        
        ReinforcementAgent.final(self, state)
        

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
