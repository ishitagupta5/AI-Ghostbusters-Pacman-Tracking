# distance_calculator.py
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


"""
This file contains a Distancer object which computes and
caches the shortest path between any two points in the maze. It
returns a Manhattan distance between two points if the maze distance
has not yet been calculated.

Example:
distancer = Distancer(gameState.data.layout)
distancer.get_distance( (1,1), (10,10) )

The Distancer object also serves as an example of sharing data
safely among agents via a global dictionary (distance_map),
and performing asynchronous computation via threads. These
examples may help you in designing your own objects, but you
shouldn't need to modify the Distancer code in order to use its
distances.
"""

import threading, sys, time, random, util

class Distancer:
  def __init__(self, layout, background=True, default=10000):
    """
    Initialize with Distancer(layout).  Changing default is unnecessary.

    This will start computing maze distances in the background and use them
    as soon as they are ready.  In the meantime, it returns manhattan distance.

    To compute all maze distances on initialization, set background=False
    """
    self._distances = None
    self.default = default

    # Start computing distances in the background; when the dc finishes,
    # it will fill in self._distances for us.
    dc = DistanceCalculator()
    dc.set_attr(layout, self)
    dc.setDaemon(True)
    if background:
      dc.start()
    else:
      dc.run()

  def get_distance(self, pos1, pos2):
    """
    The get_distance function is the only one you'll need after you create the object.
    """
    if self._distances == None:
      return manhattan_distance(pos1, pos2)
    if is_int(pos1) and is_int(pos2):
      return self.get_distance_on_grid(pos1, pos2)
    pos1_grids = get_grids_2d(pos1)
    pos2_grids = get_grids_2d(pos2)
    best_distance = self.default
    for pos1_snap, snap1_distance in pos1_grids:
      for pos2_snap, snap2_distance in pos2_grids:
        grid_distance = self.get_distance_on_grid(pos1_snap, pos2_snap)
        distance = grid_distance + snap1_distance + snap2_distance
        if best_distance > distance:
          best_distance = distance
    return best_distance

  def get_distance_on_grid(self, pos1, pos2):
    key = (pos1, pos2)
    if key in self._distances:
      return self._distances[key]
    else:
      raise Exception("Positions not in grid: " + str(key))

  def is_ready_for_maze_distance(self):
    return self._distances != None

def manhattan_distance(x, y ):
  return abs( x[0] - y[0] ) + abs( x[1] - y[1] )

def is_int(pos):
  x, y = pos
  return x == int(x) and y == int(y)

def get_grids_2d(pos):
  grids = []
  for x, x_distance in get_grids_1d(pos[0]):
    for y, y_distance in get_grids_1d(pos[1]):
      grids.append(((x, y), x_distance + y_distance))
  return grids

def get_grids_1d(x):
  int_x = int(x)
  if x == int(x):
    return [(x, 0)]
  return [(int_x, x-int_x), (int_x+1, int_x+1-x)]

##########################################
# MACHINERY FOR COMPUTING MAZE DISTANCES #
##########################################

distance_map = {}
distance_map_semaphore = threading.Semaphore(1)
distance_thread = None

def wait_on_distance_calculator(t):
  global distance_thread
  if distance_thread != None:
    time.sleep(t)

class DistanceCalculator(threading.Thread):
  def set_attr(self, layout, distancer, default = 10000):
    self.layout = layout
    self.distancer = distancer
    self.default = default

  def run(self):
    global distance_map, distance_thread
    distance_map_semaphore.acquire()

    if self.layout.walls not in distance_map:
      if distance_thread != None: raise Exception('Multiple distance threads')
      distance_thread = self

      distances = compute_distances(self.layout)
      print('[Distancer]: Switching to maze distances',file=sys.stdout)

      distance_map[self.layout.walls] = distances
      distance_thread = None
    else:
      distances = distance_map[self.layout.walls]

    distance_map_semaphore.release()
    self.distancer._distances = distances

def compute_distances(layout):
    distances = {}
    all_nodes = layout.walls.as_list(False)
    for source in all_nodes:
        dist = {}
        closed = {}
        for node in all_nodes:
            dist[node] = 1000000000
        queue = util.PriorityQueue()
        queue.push(source, 0)
        dist[source] = 0
        while not queue.is_empty():
            node = queue.pop()
            if node in closed:
                continue
            closed[node] = True
            node_dist = dist[node]
            adjacent = []
            x, y = node
            if not layout.is_wall((x,y+1)):
                adjacent.append((x,y+1))
            if not layout.is_wall((x,y-1)):
                adjacent.append((x,y-1) )
            if not layout.is_wall((x+1,y)):
                adjacent.append((x+1,y) )
            if not layout.is_wall((x-1,y)):
                adjacent.append((x-1,y))
            for other in adjacent:
                if not other in dist:
                    continue
                old_dist = dist[other]
                new_dist = node_dist+1
                if new_dist < old_dist:
                    dist[other] = new_dist
                    queue.push(other, new_dist)
        for target in all_nodes:
            distances[(target, source)] = dist[target]
    return distances


def get_distance_on_grid(distances, pos1, pos2):
    key = (pos1, pos2)
    if key in distances:
      return distances[key]
    return 100000

