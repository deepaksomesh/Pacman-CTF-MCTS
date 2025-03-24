# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# -----------------------------------------------------------------------------
# Written by: Ruilin Ma, 2023
# Version: 2.0
# Date: 2023/10/15

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import numpy as np
import math

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


#####################
# MCTS Node Class   #
#####################

class MCTSNode:
    def __init__(self, gameState, agent, action=None, parent=None, enemy_pos=None, borderline=None, max_depth=15):
        self.parent = parent
        self.action = action
        self.depth = parent.depth + 1 if parent else 0

        self.children = []
        self.visits = 0
        self.q_value = 0.0

        self.gameState = gameState.deepCopy()
        self.enemy_pos = enemy_pos
        self.legalActions = [act for act in gameState.getLegalActions(agent.index) if act != 'Stop']
        self.unexploredActions = self.legalActions[:]
        self.borderline = borderline

        self.agent = agent
        self.epsilon = 1.414  # Exploration constant for UCB
        self.rewards = 0
        self.max_depth = max_depth  # Maximum depth for rollouts

    def is_fully_expanded(self):
        return len(self.unexploredActions) == 0

    def expand(self):
        if not self.is_fully_expanded():
            action = self.unexploredActions.pop()
            next_state = self.gameState.generateSuccessor(self.agent.index, action)
            child_node = MCTSNode(next_state, self.agent, action, self, self.enemy_pos, self.borderline, self.max_depth)
            self.children.append(child_node)
            return child_node
        return None

    def select_child(self):
        best_score = -np.inf
        best_child = None

        for child in self.children:
            if child.visits == 0:
                return child  # Prioritize unvisited nodes

            exploitation = child.q_value / child.visits
            exploration = self.epsilon * math.sqrt(math.log(self.visits) / child.visits)
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def rollout(self):
        current_state = self.gameState.deepCopy()
        depth = 0
        while depth < self.max_depth:
            if current_state.isOver():
                break

            actions = current_state.getLegalActions(self.agent.index)
            if not actions:  # No legal actions available
                break

            action = self.rollout_policy(actions)
            try:
                current_state = current_state.generateSuccessor(self.agent.index, action)
            except Exception:  # Handle illegal actions
                break
            depth += 1

        return self.evaluate_state(current_state)

    def rollout_policy(self, actions):
        # Heuristic-guided rollout policy
        if random.random() < 0.8:  # 80% chance to use heuristic
            # Filter out actions that lead to illegal states
            valid_actions = []
            for action in actions:
                try:
                    next_state = self.gameState.generateSuccessor(self.agent.index, action)
                    valid_actions.append(action)
                except Exception:
                    continue
            if valid_actions:
                return max(valid_actions, key=lambda a: self.evaluate_state(self.gameState.generateSuccessor(self.agent.index, a)))
            else:
                return random.choice(actions)  # Fallback to random action if no valid actions are found
        else:  # 20% chance to explore randomly
            return random.choice(actions)

    def backpropagate(self, reward):
        self.visits += 1
        self.q_value += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def evaluate_state(self, state):
        # State evaluation function
        features = self.get_features(state)
        weights = self.get_weights()
        return features * weights

    def get_features(self, state):
        features = util.Counter()
        my_pos = state.getAgentPosition(self.agent.index)

        # Distance to closest food in the blue team's territory
        food_list = self.agent.getFood(state).asList()
        if food_list:
            features['minDistToFood'] = min([self.agent.getMazeDistance(my_pos, food) for food in food_list])

        # Distance to closest enemy ghost (filter out None positions)
        if self.enemy_pos:
            valid_enemy_pos = [pos for pos in self.enemy_pos if pos is not None]
            if valid_enemy_pos:
                features['minDistToEnemy'] = min([self.agent.getMazeDistance(my_pos, pos) for pos in valid_enemy_pos])

        # Number of carried food
        features['carriedFood'] = state.getAgentState(self.agent.index).numCarrying

        # Encourage exploration by penalizing staying in the same place
        if self.parent and my_pos == self.parent.gameState.getAgentPosition(self.agent.index):
            features['stuckPenalty'] = 1

        return features

    def get_weights(self):
        return {'minDistToFood': -1, 'minDistToEnemy': -10, 'carriedFood': 100, 'stuckPenalty': -1000}

    def mcts_search(self, timeLimit=0.99):
        """
        Perform MCTS search within the given time limit.
        """
        start = time.time()
        while time.time() - start < timeLimit:
            # Selection
            node = self.select()

            # Expansion
            if not node.is_fully_expanded():
                node = node.expand()

            # Simulation
            reward = node.rollout()

            # Backpropagation
            node.backpropagate(reward)

        # Return the best action
        best_child = self.select_child()
        return best_child.action if best_child else random.choice(self.legalActions)

    def select(self):
        """
        Traverse the tree using UCB until a leaf node is reached.
        """
        current_node = self
        while current_node.is_fully_expanded() and not current_node.gameState.isOver():
            current_node = current_node.select_child()
        return current_node


#####################
# Agents            #
#####################

class OffensiveAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.food_count = len(self.getFood(gameState).asList())
        self.arena_width = gameState.data.layout.width
        self.arena_height = gameState.data.layout.height
        self.friendly_borders = self.detect_border(gameState, is_friendly=True)
        self.hostile_borders = self.detect_border(gameState, is_friendly=False)

    def detect_border(self, gameState, is_friendly):
        walls = gameState.getWalls().asList()
        if is_friendly:
            border_x = self.arena_width // 2 - 1 if self.red else self.arena_width // 2
        else:
            border_x = self.arena_width // 2 if self.red else self.arena_width // 2 - 1
        border_line = [(border_x, h) for h in range(self.arena_height)]
        return [(x, y) for (x, y) in border_line if (x, y) not in walls and (x + 1 - 2*self.red, y) not in walls]

    def chooseAction(self, gameState):
        start = time.time()
        actions = gameState.getLegalActions(self.index)
        agent_state = gameState.getAgentState(self.index)

        if agent_state.isPacman:
            appr_ghost_pos = [gameState.getAgentPosition(g) for g in self.getOpponents(gameState) if not gameState.getAgentState(g).isPacman]
            if appr_ghost_pos:
                rootNode = MCTSNode(gameState, self, None, None, appr_ghost_pos, self.friendly_borders)
                action = rootNode.mcts_search()
            else:
                values = [self.evaluate_offensive(gameState, a) for a in actions]
                maxValue = max(values)
                bestActions = [a for a, v in zip(actions, values) if v == maxValue]
                action = random.choice(bestActions)
        else:
            # When not in Pacman mode, use a simple defensive strategy
            values = [self.evaluate_offensive(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            action = random.choice(bestActions)

        return action

    def evaluate_offensive(self, gameState, action):
        features = self.get_offensive_features(gameState, action)
        weights = self.get_offensive_weights()
        return features * weights

    def get_offensive_features(self, gameState, action):
        features = util.Counter()
        next_state = self.get_next_state(gameState, action)
        my_pos = next_state.getAgentPosition(self.index)

        # Distance to closest food in the blue team's territory
        food_list = self.getFood(next_state).asList()
        if food_list:
            features['minDistToFood'] = min([self.getMazeDistance(my_pos, food) for food in food_list])

        # Number of carried food
        features['carriedFood'] = next_state.getAgentState(self.index).numCarrying

        # Encourage exploration by penalizing staying in the same place
        if my_pos == gameState.getAgentPosition(self.index):
            features['stuckPenalty'] = 1

        return features

    def get_offensive_weights(self):
        return {'minDistToFood': -1, 'carriedFood': 100, 'stuckPenalty': -1000}

    def get_next_state(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor


class DefensiveAgent(CaptureAgent):
    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate_defensive(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

    def evaluate_defensive(self, gameState, action):
        features = self.get_defensive_features(gameState, action)
        weights = self.get_defensive_weights()
        return features * weights

    def get_defensive_features(self, gameState, action):
        features = util.Counter()
        next_state = self.get_next_state(gameState, action)
        my_pos = next_state.getAgentPosition(self.index)

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if next_state.getAgentState(self.index).isPacman:
            features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [next_state.getAgentState(i) for i in self.getOpponents(next_state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(my_pos, a.getPosition()) for a in invaders if a.getPosition() is not None]
            if dists:  # Only compute distance if there are valid positions
                features['invaderDistance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_defensive_weights(self):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

    def get_next_state(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor