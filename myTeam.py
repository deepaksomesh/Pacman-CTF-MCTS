# myTeam.py
# ---------
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
import numpy as np

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'AttackAgent', second = 'DefenseAgent'):

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# MCTS   #
##########
class Node:
    def __init__(self, gameState, agent, action, parent):
        self.gameState = gameState.deepCopy()
        self.action = action
        self.parent = parent
        self.agent = agent
        self.legalActions = [act for act in gameState.getLegalActions(agent.index) if act != Directions.STOP]
        self.unexploredActions = self.legalActions[:]
        self.Q_value = 0
        self.N_visits = 0
        self.child = []
        self.epsilon = 1
        self.reward = 0

    def is_fully_expanded(self):
        return len(self.child) == len(self.legalActions)

    def expand_node(self):
        if self.is_fully_expanded():
            return self
        executed_actions = [children.action for children in self.child]
        actions_available = self.unexploredActions.pop()

        for action in actions_available:
            if action not in executed_actions:
                next_state = self.gameState.generateSuccessor(self.agent.index, action)
                child_node = Node(next_state, self.agent, action, self)
                self.child.append(child_node)
                return child_node

class MCTSAgent:
    def __init__(self, index, simulations=100, exploration=1.4, rollout_depth=10):
        self.C = exploration
        self.simulations = simulations
        self.rollout_depth = rollout_depth

    def selection(self, node):
        best_reward = float('-inf')
        best_child_node = None
        for children in node.child:
            reward = children.node.Q_value / children.node.N_visits
            if reward > best_reward:
                best_reward = reward
                best_child_node = children
        return best_child_node

    def select_and_expand(self,node):
        if util.flipCoin(node.epsilon):
            next_best_node = self.selection(node)
        else:
            next_best_node = random.choice(node.child)
        return next_best_node.node.expand_node()

    def backpropagate(self, node, reward):
        node.N_visits += 1
        node.Q_value += reward
        if node.parent is not None:
            node.parent.backpropagate(node, reward)





##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

