# "MDPs on Ice - Assignment 5"
# Ported from Java

import random
import numpy as np
import copy
import sys

GOLD_REWARD = 100.0
PIT_REWARD = -150.0
DISCOUNT_FACTOR = 0.5
EXPLORE_PROB = 0.2 # for Q-learning
LEARNING_RATE = 0.1
ITERATIONS = 10000
MAX_MOVES = 1000
ACTIONS = 4
DOWN = 0
UP = 1
RIGHT = 2
LEFT = 3
MOVES = ['D','U','R','L']

# Fixed random number generator seed for result reproducibility --
# don't use a random number generator besides this to match sol
random.seed(5100)

# Problem class:  represents the physical space, transition probabilities, reward locations,
# and approach to use (MDP or Q) - in short, the info in the text file
class Problem:
    # Fields:
    # approach - string, "MDP" or "Q"
    # move_probs - list of doubles, probability of going 1,2,3 spaces
    # map - list of list of strings: "-" (safe, empty space), "G" (gold), "P" (pit)

    # Format looks like
    # MDP    [approach to be used]
    # 0.7 0.2 0.1   [probability of going 1, 2, 3 spaces]
    # - - - - - - P - - - -   [space-delimited map rows]
    # - - G - - - - - P - -   [G is gold, P is pit]
    #
    # You can assume the maps are rectangular, although this isn't enforced
    # by this constructor.

    # __init__ consumes stdin; don't call it after stdin is consumed or outside that context
    def __init__(self):
        self.approach = input('Reading mode...')
        print(self.approach)
        probs_string = input("Reading transition probabilities...\n")
        self.move_probs = [float(s) for s in probs_string.split()]
        self.map = []
        for line in sys.stdin:
            self.map.append(line.split())

    def solve(self, iterations):            
        if self.approach == "MDP":
            return mdp_solve(self, iterations)
        elif self.approach == "Q":
            return q_solve(self, iterations)
        return None
        
# Policy: Abstraction on the best action to perform in each state - just a 2D string list-of-lists
class Policy:
    def __init__(self, problem): # problem is a Problem
        # Signal 'no policy' by just displaying the map there
        self.best_actions = copy.deepcopy(problem.map)

    def __str__(self):
        return '\n'.join([' '.join(row) for row in self.best_actions])

# roll_steps:  helper for try_policy and q_solve -- "rolls the dice" for the ice and returns
# the new location (r,c), taking map bounds into account
# note that move is expecting a string, not an integer constant
def roll_steps(move_probs, row, col, move, rows, cols):
    displacement = 1
    total_prob = 0
    move_sample = random.random()
    for p, prob in enumerate(problem.move_probs):
        total_prob += prob
        if move_sample <= total_prob:
            displacement = p+1
            break
    # Handle "slipping" into edge of map
    new_row = row
    new_col = col
    if not isinstance(move,str):
        print("Warning: roll_steps wants str for move, got a different type")
    if move == "U":
        new_row -= displacement
        if new_row < 0:
            new_row = 0
    elif move == "R":
        new_col += displacement
        if new_col >= cols:
            new_col = cols-1
    elif move == "D":
        new_row += displacement
        if new_row >= rows:
            new_row = rows-1
    elif move == "L":
        new_col -= displacement
        if new_col < 0:
            new_col = 0
    return new_row, new_col


# try_policy:  returns avg utility per move of the policy, as measured by "iterations"
# random drops of an agent onto empty spaces, running until gold, pit, or time limit 
# MAX_MOVES is reached
def try_policy(policy, problem, iterations):
    total_utility = 0
    total_moves = 0
    for i in range(iterations):
        # Resample until we have an empty starting square
        while True:
            row = random.randrange(0,len(problem.map))
            col = random.randrange(0,len(problem.map[0]))
            if problem.map[row][col] == "-":
                break
        for moves in range(MAX_MOVES):
            total_moves += 1
            policy_rec = policy.best_actions[row][col]
            # Take the move - roll to see how far we go, bump into map edges as necessary
            row, col = roll_steps(problem.move_probs, row, col, policy_rec, len(problem.map), len(problem.map[0]))
            if problem.map[row][col] == "G":
                total_utility += GOLD_REWARD
                break
            if problem.map[row][col] == "P":
                total_utility -= PIT_REWARD
                break
    return total_utility / total_moves

# mdp_solve:  use [iterations] iterations of the Bellman equations over the whole map in [problem]
# and return the policy of what action to take in each square
def mdp_solve(problem, iterations):
    policy = Policy(problem)
    MOVES = ['U','R','D','L']
    # TODO
    ##Get reward values
    r_map = []
    for i in range(len(problem.map)):
        rewards = []
        for j in range(len(problem.map[0])):
            if problem.map[i][j] == "-":
                rewards.append(0)
            if problem.map[i][j] == "G":
                rewards.append(GOLD_REWARD)
            if problem.map[i][j] == "P":
                rewards.append(PIT_REWARD)
        r_map.append(rewards)
    ##Do iterations and update utility values
    for x in range(iterations):
        if x == 0:
            utility_map = r_map
        else:
            utility_map = new_utility_map
        new_utility_map = []
        for i in range(len(problem.map)):
            utility = []
            for j in range(len(problem.map[0])):
                if problem.map[i][j] == "G" or problem.map[i][j] == "P":
                    utility.append(r_map[i][j])
                else:
                    u_up,u_down,u_left,u_right = 0,0,0,0
                    for p, prob in enumerate(problem.move_probs):
                        if j - p - 1 < 0:
                            u_left += prob * utility_map[i][0]
                        else:
                            u_left += prob * utility_map[i][j-p-1]
                        
                        if i - p - 1 < 0:
                            u_up += prob * utility_map[0][j]
                        else:
                            u_up += prob * utility_map[i-p-1][j]
                        
                        if j + p + 1 >= len(problem.map[0]):
                            u_right += prob * utility_map[i][len(problem.map[0])-1]
                        else:
                            u_right += prob * utility_map[i][j+p+1]
                        
                        if i + p + 1 >= len(problem.map):
                            u_down += prob * utility_map[len(problem.map)-1][j]
                        else:
                            u_down += prob * utility_map[i+p+1][j]
                    u = [u_up,u_right,u_down,u_left]
                    ##Decide which value to use
                    utility.append(r_map[i][j] + DISCOUNT_FACTOR*max(u))
                    policy.best_actions[i][j] = MOVES[u.index(max(u))]
            new_utility_map.append(utility)
    return policy

def q_solve(problem, iterations):
    policy = Policy(problem)
    # TODO
    ##initialize Q values by list of lists of lists
    q_map = []
    r_map = []
    for i in range(len(problem.map)):
        rewards = []
        q_values = []
        for j in range(len(problem.map[0])):
            if problem.map[i][j] == "-":
                q_values.append([0,0,0,0])
                rewards.append(0)
            if problem.map[i][j] == "G":
                q_values.append([GOLD_REWARD,GOLD_REWARD,GOLD_REWARD,GOLD_REWARD])
                rewards.append(GOLD_REWARD)
            if problem.map[i][j] == "P":
                q_values.append([PIT_REWARD,PIT_REWARD,PIT_REWARD,PIT_REWARD])
                rewards.append(PIT_REWARD)
        q_map.append(q_values)
        r_map.append(rewards)
        
        
    for x in range(iterations):
        ##Get random empty space where a learner starts
        i = random.randint(0,len(problem.map)-1)
        j = random.randint(0,len(problem.map[0])-1)
        
        ##Start the trial
        while problem.map[i][j] == "-":
            ##Decide if go a random direction
            if_best = random.random()
            if if_best <= EXPLORE_PROB:
                ##Move to random direction
                move_dir = random.randint(0,3)
                
            else:
                ##Move according to the best Q values
                qs = q_map[i][j]
                move_dir = qs.index(max(qs))
            new_row,new_col = roll_steps(problem.move_probs,i,j,MOVES[move_dir], len(problem.map),len(problem.map[0]))
            if i != new_row or j != new_col:
                q_map[i][j][move_dir] += LEARNING_RATE*(r_map[i][j] + DISCOUNT_FACTOR * max(q_map[new_row][new_col]) - q_map[i][j][move_dir])
            i = new_row
            j = new_col
            
    
    for i in range(len(problem.map)):
        for j in range(len(problem.map[0])):
            u = q_map[i][j]
            if problem.map[i][j] == "-":
                policy.best_actions[i][j] = MOVES[u.index(max(u))]
    return policy

# Main:  read the problem from stdin, print the policy and the utility over a test run
if __name__ == "__main__":
    problem = Problem()
    policy = problem.solve(ITERATIONS)
    print(policy)
    print("Calculating average utility...")
    print("Average utility per move: {utility:.2f}".format(utility = try_policy(policy, problem,ITERATIONS)))
        
