# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util
import numpy as np

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        value = self.values[(state, action)]
        return value
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if len(actions)==0:
          return 0
        
        value = None
        QValues = []
        for action in actions:
          q_val = self.getQValue(state, action)
          QValues.append(q_val)
          
        return max(QValues)
  

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        max_val = 0
        best_action = None
        
        for action in actions:
          q_val = self.getQValue(state, action)
          if best_action == None:
            max_val = q_val
            best_action = action
          if q_val > max_val:
            max_val = q_val
            best_action = action
        
        return best_action
        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        
        exploration = util.flipCoin(self.epsilon)
        if exploration:
          action = random.choice(legalActions)
        else:
          action = self.getPolicy(state)
          
        return action
      
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"     
        q_val = self.getQValue(state, action)
        old_val = (1 - self.alpha) * q_val
        
        actions = self.getLegalActions(nextState)
        if len(actions) == 0:
          new_val = self.alpha * reward
        else:
          new_val = self.alpha * (reward + (self.discount * self.computeValueFromQValues(nextState)))
        self.values[(state, action)] = old_val + new_val

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        self.theta = np.random.rand(5)
        QLearningAgent.__init__(self, **args)


class ReinforceAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.transition_history = []
        self.actions = ['North','South' ,'East','West','Stop']
        self.gamma = 0.99
      
    def softmax(self, x):
        exp = np.exp(x)
        y = exp / np.sum(exp)
        return y
      
    def log_derivative(self, theta, state, action):
        features = self.featExtractor.getFeatures(state, action)
        x = [features["closest-food"] + features["#-of-ghosts-1-step-away"]]
        x = x * theta
        return x - x*self.softmax(x)

    def actionProbs(self, theta, state):
        #x = (state.getPacmanPosition(), tuple(state.getGhostPositions()), state.getScore(), state.getNumFood(), state.getNumAgents(), tuple(state.getCapsules()) )
        i = 0
        x = []
        for i, action in enumerate(self.actions):
          features = self.featExtractor.getFeatures(state, action)
          x.append(features["closest-food"] + features["#-of-ghosts-1-step-away"])
          x[i] = x[i] * theta[i]

        return self.softmax(x)
        
    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        q_value = 0
        features = self.featExtractor.getFeatures(state, action)
        keys = features.keys()
        weights = self.getWeights()
        for key in keys:
            q_value += features[key] * weights[key]
        return q_value

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method. 
        """
        #action = QLearningAgent.getAction(self,state)
        legalActions = self.getLegalActions(state)
        probs = self.actionProbs(self.theta, state)
        #print(probs)
        action = np.random.choice(self.actions, p=probs)
        if action not in legalActions:
          action = np.random.choice(legalActions)
        #print(action)
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action
      
    def update(self, state, action, nextState, reward):
        self.transition_history.append([state, action, reward])

        
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        print("ep")
        G = 0
        
        for t in reversed(range(0, len(self.transition_history))):
          G = G * self.gamma + self.transition_history[t][2]
          log_prob = self.log_derivative(self.theta, self.transition_history[t][0], self.transition_history[t][1])
          self.theta += self.alpha * np.power(self.gamma, t) * log_prob * G
           
        self.transition_history = [] 

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
