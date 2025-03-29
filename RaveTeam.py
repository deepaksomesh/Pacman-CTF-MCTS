from captureAgents import CaptureAgent
import random
import time
import util
from game import Directions
import game
import numpy as np

import math

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveReflexAgent'):
    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]



#####################
# MCTS with UCT     #
#####################
class MCTS(object):

    def __init__(self, gameState, agent, action, parent, enemy, crossline):
        self.gameState = gameState.deepCopy()
        self.agent = agent
        self.action = action
        self.parent = parent
        self.child = []
        self.enemy = enemy
        self.crossline = crossline

        self.max_node_depth = 10
        self.depth = parent.depth + 1 if parent else 0
        self.rewards = 0

        self.visits = 0
        self.q_value = 0.0
        # RAVE
        self.amaf_visits = {}
        self.amaf_values = {}
        self.action_history = [] if parent is None else parent.action_history + [action]
        self.beta = 0.5

        self.legalActions = [act for act in gameState.getLegalActions(agent.index) if act != Directions.STOP]
        self.available_actions = self.legalActions[:]

    # Select and Expand the best child node
    def select_and_expand(self):
        if self.depth >= self.max_node_depth:
            return self

        if self.available_actions:
            action = self.available_actions.pop()
            next_game_state = self.gameState.generateSuccessor(self.agent.index, action)
            child_node = MCTS(next_game_state, self.agent, action, self, self.enemy, self.crossline)
            self.child.append(child_node)
            return child_node


        if self.child:
            return self.best_child_node().select_and_expand()
        else:
            return self

    # Upped Confidence Bound (UCT)
    def best_child_node(self):
        best_score = float('-inf')
        best_child = None
        c = 1

        for i in self.child:
            if i.visits == 0:
                return i
            # Regular UCT value
            q_value = i.q_value / i.visits
            exploration = math.sqrt(math.log(self.visits) / i.visits)

            # RAVE value
            amaf_visits = self.amaf_visits.get(i.action, 0)
            amaf_value = self.amaf_values.get(i.action, 0) / (amaf_visits if amaf_visits > 0 else 1)

            # Beta parameter for mixing Q and AMAF values
            beta = amaf_visits / (
                    i.visits + amaf_visits + self.beta * i.visits * amaf_visits + 1e-6)

            # Combined estimate
            combined_value = (1 - beta) * q_value + beta * amaf_value
            score = combined_value + c * exploration
            if score > best_score:
                best_score = score
                best_child = i
        return best_child

    def backpropagation(self, reward):
        self.visits += 1
        self.q_value += reward
        # Update AMAF statistics for all actions in the history
        for action in set(self.action_history):
            self.amaf_visits[action] = self.amaf_visits.get(action, 0) + 1
            self.amaf_values[action] = self.amaf_values.get(action, 0) + reward

        if self.parent is not None:
            self.parent.backpropagation(reward)

    def reward(self):
        current_pos = self.gameState.getAgentPosition(self.agent.index)
        # Penalize returning to the start
        if current_pos == self.gameState.getInitialAgentPosition(self.agent.index):
            return -100

        feature = util.Counter()
        weights = {'distance': -1}
        current_pos = self.gameState.getAgentPosition(self.agent.index)
        feature['distance'] = min([self.agent.getMazeDistance(current_pos, crossline_pos) for crossline_pos in self.crossline])
        value = feature * weights
        return value

    # MCTS Search
    def run_mcts(self):
        time_limit = 0.95
        start = time.time()
        end_time = start + time_limit
        while time.time() < end_time:
            node_selected = self.select_and_expand()
            reward = node_selected.reward()
            node_selected.backpropagation(reward)

        return self.best_child_node().action


##########
# Agents #
##########

class OffensiveAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.layout_width = gameState.data.layout.width
        self.layout_height = gameState.data.layout.height
        self.borders = self.detect_borders(gameState)

    def detect_borders(self, gameState):
        """
        Return borders position
        """
        walls = gameState.getWalls().asList()
        if self.red:
            border = self.layout_width // 2 - 1
        else:
            border = self.layout_width // 2
        border_line = [(border, h) for h in range(self.layout_height)]
        return [(x, y) for (x, y) in border_line if (x, y) not in walls and (x + 1 - 2*self.red, y) not in walls]

    def enemy(self, gameState):
        """
        Return Observable Oppo-Ghost Index
        """
        enemyList = []
        for enemy in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(enemy)
            if (not enemyState.isPacman) and enemyState.scaredTimer == 0:
                enemyPos = gameState.getAgentPosition(enemy)
                if enemyPos != None:
                    enemyList.append(enemy)
        return enemyList

    def detect_enemy(self, gameState):
        """
        Return Observable Oppo-Ghost Position Within 5 Steps
        """
        enemyGhosts = []
        ghosts = self.enemy(gameState)
        myPos = gameState.getAgentPosition(self.index)
        for i in ghosts:
            distance = self.getMazeDistance(myPos, gameState.getAgentPosition(i))
            if distance <= 5:
                enemyGhosts.append(i)
        return enemyGhosts

    def chooseAction(self, gameState):
        """
        Picks best actions.
        """
        actions = gameState.getLegalActions(self.index)
        agent_state = gameState.getAgentState(self.index)

        if agent_state.isPacman:
            return self.handleOffense(gameState, actions, agent_state.numCarrying)
        else:
            return self.handleDefense(gameState, actions)

    def handleOffense(self, gameState, actions, carrying):
        """
        Handles offensive strategy when agent is Pacman.
        """
        appr_ghost_pos = [gameState.getAgentPosition(g) for g in self.detect_enemy(gameState)]
        foodList = self.getFood(gameState).asList()

        if not appr_ghost_pos:
            values = [self.evaluate_state_off(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            action_chosen = random.choice(bestActions)

        elif len(foodList) < 2 or carrying > 7:
            Node = MCTS(gameState, self, None, None, appr_ghost_pos, self.borders)
            action_chosen = MCTS.run_mcts(Node)
        else:
            Node = MCTS(gameState, self, None, None, appr_ghost_pos, self.borders)
            action_chosen = MCTS.run_mcts(Node)

        return action_chosen

    def handleDefense(self, gameState, actions):
        """
        Handles defensive strategy when agent is not Pacman.
        """
        ghosts = self.enemy(gameState)
        values = [self.evaluate_state_def(gameState, a, ghosts) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        action_chosen = random.choice(bestActions)

        return action_chosen

    def evaluate_state_off(self, gameState, action):
        """
        Offensive State Evaluation
        """

        features = util.Counter()
        weights = {'minDistToFood': -1, 'getFood': 100}
        next_state = self.get_next_state(gameState, action)
        my_pos = next_state.getAgentPosition(self.index)
        if next_state.getAgentState(self.index).numCarrying > gameState.getAgentState(self.index).numCarrying:
            features['getFood'] = 1
        else:
            if len(self.getFood(next_state).asList()) > 0:
                features['minDistToFood'] = min([self.getMazeDistance(my_pos, f) for f in self.getFood(next_state).asList()])
        return features * weights

    def evaluate_state_def(self, gameState, action, ghosts):
        """
        Defensive State Evaluation
        """
        features = self.get_features(gameState, action)
        weights = self.get_weights(gameState, action)
        return features * weights

    def get_features(self, gameState, action):
        features = util.Counter()
        successor = self.get_next_state(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        if len(foodList) > 0:
            current_pos = successor.getAgentState(self.index).getPosition()
            min_distance = min([self.getMazeDistance(current_pos, food) for food in foodList])
            features['distanceToFood'] = min_distance
        return features

    def get_weights(self, gameState, action):

        return {'successorScore': 100, 'distanceToFood': -1}

    def get_next_state(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        return successor

class DefensiveReflexAgent(OffensiveAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, gameState, action):
        features = util.Counter()
        next_state = self.get_next_state(gameState, action)

        my_state = next_state.getAgentState(self.index)
        my_pos = my_state.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if my_state.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [next_state.getAgentState(i) for i in self.getOpponents(next_state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(my_pos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

