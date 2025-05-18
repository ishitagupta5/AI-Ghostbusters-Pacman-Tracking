# inference.py
# ------------
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


import itertools
import random
import busters
import game

from util import manhattan_distance, raise_not_defined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argmax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        max_index = values.index(max(values))
        return all[max_index][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        total_sum = self.total()
        if total_sum == 0:
            return
        for key in self.keys():
            self[key] /= total_sum

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        total_sum = self.total()
        if total_sum == 0:
            return None
    
        r = random.random() * total_sum
        running_sum = 0.0
    
        for key, value in self.items():
            running_sum += value
            if running_sum >= r:
                return key
    
        return key


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghost_agent):
        """
        Set the ghost agent for later access.
        """
        self.ghost_agent = ghost_agent
        self.index = ghost_agent.index
        self.obs = []  # most recent observation position

    def get_jail_position(self):
        return (2 * self.ghost_agent.index - 1, 1)

    def get_position_distribution_helper(self, game_state, pos, index, agent):
        try:
            jail = self.get_jail_position()
            game_state = self.set_ghost_position(game_state, pos, index + 1)
        except TypeError:
            jail = self.get_jail_position(index)
            game_state = self.set_ghost_positions(game_state, pos)
        pacman_position = game_state.get_pacman_position()
        ghost_position = game_state.get_ghost_position(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacman_position == ghost_position:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacman_successor_states = game.Actions.get_legal_neighbors(pacman_position, game_state.get_walls())  # Positions Pacman can move to
        if ghost_position in pacman_successor_states:  # Ghost could get caught
            mult = 1.0 / float(len(pacman_successor_states))
            dist[jail] = mult
        else:
            mult = 0.0
        action_dist = agent.get_distribution(game_state)
        for action, prob in action_dist.items():
            successor_position = game.Actions.get_successor(ghost_position, action)
            if successor_position in pacman_successor_states:  # Ghost could get caught
                denom = float(len(action_dist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successor_position] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successor_position] = prob * (1.0 - mult)
        return dist

    def get_position_distribution(self, game_state, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given game_state. You must first place the ghost in the game_state, using
        set_ghost_position below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghost_agent
        return self.get_position_distribution_helper(game_state, pos, index, agent)

    def get_observation_prob(self, noisy_distance, pacman_position, ghost_position, jail_position):
        """
        Return the probability P(noisy_distance | pacman_position, ghost_position).
        """
        if ghost_position == jail_position:
        # If observation is None, probability is 1; else 0.
            return 1.0 if noisy_distance is None else 0.0

        # If ghost is not in jail but observation is None, it's impossible → prob 0
        if noisy_distance is None:
            return 0.0

        # Otherwise, use the busters observation model
        true_distance = manhattan_distance(pacman_position, ghost_position)
        return busters.get_observation_probability(noisy_distance, true_distance)

    def set_ghost_position(self, game_state, ghost_position, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied game_state.

        Note that calling set_ghost_position does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghost_position, game.Directions.STOP)
        game_state.data.agent_states[index] = game.AgentState(conf, False)
        return game_state

    def set_ghost_positions(self, game_state, ghost_positions):
        """
        Sets the position of all ghosts to the values in ghost_positions.
        """
        for index, pos in enumerate(ghost_positions):
            conf = game.Configuration(pos, game.Directions.STOP)
            game_state.data.agent_states[index + 1] = game.AgentState(conf, False)
        return game_state

    def observe(self, game_state):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = game_state.get_noisy_ghost_distances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observe_update(obs, game_state)

    def initialize(self, game_state):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legal_positions = [p for p in game_state.get_walls().as_list(False) if p[1] > 1]
        self.all_positions = self.legal_positions + [self.get_jail_position()]
        self.initialize_uniformly(game_state)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initialize_uniformly(self, game_state):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observe_update(self, observation, game_state):
        """
        Update beliefs based on the given distance observation and game_state.
        """
        raise NotImplementedError

    def elapse_time(self, game_state):
        """
        Predict beliefs for the next time step from a game_state.
        """
        raise NotImplementedError

    def get_belief_distribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initialize_uniformly(self, game_state):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legal_positions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe_update(self, observation, game_state):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.all_positions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.all_positions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        pacman_position = game_state.get_pacman_position()
        jail_position = self.get_jail_position()

        # Update each position's belief using P(observation | position)
        for pos in self.all_positions:
            # Probability that we see 'observation' if ghost is at 'pos'
            prob = self.get_observation_prob(observation, pacman_position, pos, jail_position)
            # Multiply old belief by observation likelihood
            self.beliefs[pos] *= prob
        
        # Code line that is provided for you -- don't forget to normalize at the end!
        self.beliefs.normalize()

    def elapse_time(self, game_state):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        new_beliefs = DiscreteDistribution()

        for old_pos in self.all_positions:
            old_prob = self.beliefs[old_pos]
        
            if old_prob > 0:
                new_pos_dist = self.get_position_distribution(game_state, old_pos)
                for new_pos, trans_prob in new_pos_dist.items():
                    new_beliefs[new_pos] += old_prob * trans_prob

        new_beliefs.normalize()
        self.beliefs = new_beliefs

    def get_belief_distribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghost_agent, num_particles=300):
        InferenceModule.__init__(self, ghost_agent)
        self.set_num_particles(num_particles)

    def set_num_particles(self, num_particles):
        self.num_particles = num_particles

    def initialize_uniformly(self, game_state):
        """
        Initialize a list of particles. Use self.num_particles for the number of
        particles. This particle could is already provided for you.  The value that
        is stored for each particle is the current board position of that particle.
        
        Use self.legal_positions for the legal board positions where a particle
        could be located. Particles should be evenly (not randomly) distributed
        across positions in order to ensure a uniform prior. Use self.particles
        to store the list of particles.
        """
        
        self.particles = []
        
        num_legal = len(self.legal_positions)
    
        # Round-robin / cycling approach
        i = 0
        while len(self.particles) < self.num_particles:
            self.particles.append(self.legal_positions[i % num_legal])
            i += 1

    def observe_update(self, observation, game_state):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initialize_uniformly(). The total method of
        the DiscreteDistribution may be useful.
        """
        pacman_position = game_state.get_pacman_position()
        jail_position = self.get_jail_position()
    
        weights = DiscreteDistribution()
        for particle in self.particles:
            w = self.get_observation_prob(observation, pacman_position, particle, jail_position)
            weights[particle] += w

        if weights.total() == 0:
            self.initialize_uniformly(game_state)
        else:
            new_particles = []
            for _ in range(self.num_particles):
                new_particles.append(weights.sample())
            self.particles = new_particles

    def elapse_time(self, game_state):
        """
        Sample each particle's next state based on its current state and the
        game_state.
        """
        
        new_particles = []
        for old_pos in self.particles:
            new_pos_dist = self.get_position_distribution(game_state, old_pos)
            new_pos = new_pos_dist.sample()
            new_particles.append(new_pos)
    
        self.particles = new_particles

    def get_belief_distribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        dist = DiscreteDistribution()
    
        # Tally up how many times each position appears in self.particles
        for particle in self.particles:
            dist[particle] += 1
    
        # Normalize so it sums to 1
        dist.normalize()
        return dist


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, num_particles=600):
        self.set_num_particles(num_particles)

    def initialize(self, game_state, legal_positions):
        """
        Store information about the game, then initialize particles.
        """
        self.num_ghosts = game_state.get_num_agents() - 1
        self.ghost_agents = []
        self.legal_positions = legal_positions
        self.initialize_uniformly(game_state)

    def initialize_uniformly(self, game_state):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        import itertools

        self.particles = []
        all_possible = list(itertools.product(self.legal_positions, repeat=self.num_ghosts))
    
        random.shuffle(all_possible)
    
        for i in range(self.num_particles):
            self.particles.append(all_possible[i % len(all_possible)])

    def add_ghost_agent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghost_agents.append(agent)

    def get_jail_position(self, i):
        return (2 * i + 1, 1)

    def observe(self, game_state):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = game_state.get_noisy_ghost_distances()
        self.observe_update(observation, game_state)

    def observe_update(self, observation, game_state):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initialize_uniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        
        pacman_position = game_state.get_pacman_position()
        weights = DiscreteDistribution()
        for particle in self.particles:
            joint_weight = 1.0
        
            for i in range(self.num_ghosts):
                ghost_pos = particle[i]
                jail_pos = self.get_jail_position(i)
            
                prob = self.get_observation_prob(observation[i], pacman_position, ghost_pos, jail_pos)
                joint_weight *= prob
        
            weights[particle] += joint_weight

        if weights.total() == 0:
            self.initialize_uniformly(game_state)
        else:
            new_particles = []
            for _ in range(self.num_particles):
                new_particles.append(weights.sample())
            self.particles = new_particles

    def elapse_time(self, game_state):
        """
        Sample each particle's next state based on its current state and the
        game_state.
        """
        new_particles = []
        for old_particle in self.particles:
            new_particle = []

            # now loop through the ghosts and update each entry in new_particle...
            
            for i in range(self.num_ghosts):
                new_pos_dist = self.get_position_distribution(
                    game_state, old_particle, i, self.ghost_agents[i]
                )
                new_pos = new_pos_dist.sample()
                new_particle.append(new_pos)
            
            new_particles.append(tuple(new_particle))
        self.particles = new_particles


# One JointInference module is shared globally across instances of MarginalInference
joint_inference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initialize_uniformly(self, game_state):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            joint_inference.initialize(game_state, self.legal_positions)
        joint_inference.add_ghost_agent(self.ghost_agent)

    def observe(self, game_state):
        """
        Update beliefs based on the given distance observation and game_state.
        """
        if self.index == 1:
            joint_inference.observe(game_state)

    def elapse_time(self, game_state):
        """
        Predict beliefs for a time step elapsing from a game_state.
        """
        if self.index == 1:
            joint_inference.elapse_time(game_state)

    def get_belief_distribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        joint_distribution = joint_inference.get_belief_distribution()
        dist = DiscreteDistribution()
        for t, prob in joint_distribution.items():
            dist[t[self.index - 1]] += prob
        return dist
