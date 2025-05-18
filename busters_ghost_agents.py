# busters_ghost_agents.py
# ---------------------
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


import ghost_agents
from game import Directions
from game import Actions
from util import manhattan_distance
import util

class StationaryGhost( ghost_agents.GhostAgent ):
    def get_distribution( self, state ):
        dist = util.Counter()
        dist[Directions.STOP] = 1.0
        return dist

class DispersingGhost( ghost_agents.GhostAgent ):
    "Chooses an action that distances the ghost from the other ghosts with probability spread_prob."
    def __init__( self, index, spread_prob=0.5):
        self.index = index
        self.spread_prob = spread_prob

    def get_distribution( self, state ):
        ghost_state = state.get_ghost_state( self.index )
        legal_actions = state.get_legal_actions( self.index )
        pos = state.get_ghost_position( self.index )
        is_scared = ghost_state.scared_timer > 0

        speed = 1
        if is_scared: speed = 0.5
        action_vectors = [Actions.direction_to_vector( a, speed ) for a in legal_actions]
        new_positions = [( pos[0]+a[0], pos[1]+a[1] ) for a in action_vectors]

        # get other ghost positions
        others = [i for i in range(1,state.get_num_agents()) if i != self.index]
        for a in others: assert state.get_ghost_state(a) != None, "Ghost position unspecified in state!"
        other_ghost_positions = [state.get_ghost_position(a) for a in others if state.get_ghost_position(a)[1] > 1]

        # for each action, get the sum of inverse squared distances to the other ghosts
        sum_of_distances = []
        for pos in new_positions:
            sum_of_distances.append( sum([(1+manhattan_distance(pos, g))**(-2) for g in other_ghost_positions]) )

        best_distance = min(sum_of_distances)
        num_best = [best_distance == dist for dist in sum_of_distances].count(True)
        distribution = util.Counter()
        for action, distance in zip(legal_actions, sum_of_distances):
            if distance == best_distance: distribution[action] += self.spread_prob / num_best
            distribution[action] += (1 - self.spread_prob) / len(legal_actions)
        return distribution
