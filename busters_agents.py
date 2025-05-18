# busters_agents.py
# ----------------
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


import util
from game import Agent
from game import Actions
from game import Directions
from keyboard_agents import KeyboardAgent
import inference
import busters
from distance_calculator import Distancer

class NullGraphics:
    "Placeholder for graphics"
    def initialize(self, state, is_blue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def update_distributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initialize_uniformly(self, game_state):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legal_positions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe_update(self, observation, game_state):
        noisy_distance = observation
        pacman_position = game_state.get_pacman_position()
        all_possible = util.Counter()
        for p in self.legal_positions:
            true_distance = util.manhattan_distance(p, pacman_position)
            if noisy_distance != None and \
                    busters.get_observation_probability(noisy_distance, true_distance) > 0:
                all_possible[p] = 1.0
        all_possible.normalize()
        self.beliefs = all_possible

    def elapse_time(self, game_state):
        pass

    def get_belief_distribution(self):
        return self.beliefs


class BustersAgent:
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghost_agents = None, observe_enable = True, elapse_time_enable = True):
        try:
            inference_type = util.lookup(inference, globals())
        except Exception:
            inference_type = util.lookup('inference.' + inference, globals())
        self.inference_modules = [inference_type(a) for a in ghost_agents]
        self.observe_enable = observe_enable
        self.elapse_time_enable = elapse_time_enable

    def register_initial_state(self, game_state):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inference_modules:
            inference.initialize(game_state)
        self.ghost_beliefs = [inf.get_belief_distribution() for inf in self.inference_modules]
        self.first_move = True

    def observation_function(self, game_state):
        "Removes the ghost states from the game_state"
        agents = game_state.data.agent_states
        game_state.data.agent_states = [agents[0]] + [None for i in range(1, len(agents))]
        return game_state

    def get_action(self, game_state):
        "Updates beliefs, then chooses an action based on updated beliefs."
        for index, inf in enumerate(self.inference_modules):
            if not self.first_move and self.elapse_time_enable:
                inf.elapse_time(game_state)
            self.first_move = False
            if self.observe_enable:
                inf.observe(game_state)
            self.ghost_beliefs[index] = inf.get_belief_distribution()
        self.display.update_distributions(self.ghost_beliefs)
        return self.choose_action(game_state)

    def choose_action(self, game_state):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghost_agents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghost_agents)

    def get_action(self, game_state):
        return BustersAgent.get_action(self, game_state)

    def choose_action(self, game_state):
        return KeyboardAgent.get_action(self, game_state)


class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def register_initial_state(self, game_state):
        "Pre-computes the distance between every two points."
        BustersAgent.register_initial_state(self, game_state)
        self.distancer = Distancer(game_state.data.layout, False)

    def choose_action(self, game_state):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closest to the closest ghost (according to maze_distance!).
        """
        pacman_position = game_state.get_pacman_position()
        legal_actions = [a for a in game_state.get_legal_pacman_actions()]
        living_ghosts = game_state.get_living_ghosts()
        living_ghost_position_distributions = [beliefs for i, beliefs in enumerate(self.ghost_beliefs) if living_ghosts[i+1]]
        
        most_likely_positions = []
        for dist in living_ghost_position_distributions:
            most_likely_positions.append(dist.argmax())
    
        closest_ghost_pos = None
        closest_dist = float('inf')
        for ghost_pos in most_likely_positions:
            dist = self.distancer.get_distance(pacman_position, ghost_pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_ghost_pos = ghost_pos
    
        best_action = Directions.STOP
        best_action_distance = float('inf')
        for action in legal_actions:
            successor_pos = Actions.get_successor(pacman_position, action)
            dist = self.distancer.get_distance(successor_pos, closest_ghost_pos)
            if dist < best_action_distance:
                best_action_distance = dist
                best_action = action
        return best_action

        

