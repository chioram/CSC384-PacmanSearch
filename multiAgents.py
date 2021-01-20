# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
import util

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        newfoodlist = newFood.asList()
        oldfoodlist = currentGameState.getFood().asList()
        currentPos = currentGameState.getPacmanPosition()

        ghost_pos = [g.getPosition() for g in newGhostStates]
        new_dist_to_ghosts = [(abs(newPos[0] - g[0]) + abs(newPos[1] - g[1])) for g in ghost_pos]
        old_dist_to_ghosts = [(abs(currentPos[0] - g[0]) + abs(currentPos[1] - g[1])) for g in ghost_pos]

        closer = [y<x for x in old_dist_to_ghosts for y in new_dist_to_ghosts]

        new_food_dists = [] #for new food dist
        old_food_dists = [] # for old food dist

        if len(newfoodlist) == 0:
            new_food_dists.append(0)
        else:
            for food in newfoodlist:
                    dist_to_food = abs((newPos[0] - food[0]) + (newPos[1] - food[1]))
                    new_food_dists.append(dist_to_food)

        if len(oldfoodlist) == 0:
            old_food_dists.append(0)
        else:
            for food in oldfoodlist:
                    dist_to_food = abs((currentPos[0] - food[0]) + (currentPos[1] - food[1]))
                    old_food_dists.append(dist_to_food)

        # we only care about if the min distance to food is reduced

        if any([x<=1 for x in new_dist_to_ghosts]) and not(any([x > 6 for x in newScaredTimes])):
            score = -1  # run away from ALL ghosts
        elif any([x > 4 for x in newScaredTimes]) and any(closer):
            score = 777  # Try to eat ghosts if they're scared
        elif currentGameState.getScore() < successorGameState.getScore():
            score = 555 + max(new_food_dists)
        elif currentGameState.getFood().count() - newFood.count() > 0:
            score = 444
        #elif #clause for non-random movement
        else:
            score = 0

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        numAgents = gameState.getNumAgents()
        minimax_move = self.minimax(gameState, (self.depth * numAgents), 0, numAgents)
        return minimax_move[0]

    def minimax(self, gameState, depth, turn, numAgents):

        best_move = None

        # Base Case: (1) state is terminal (2) depth is d
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return best_move, self.evaluationFunction(gameState)
        if turn == 0: # MAX
            value = -9999999
        else: # MIN
            value = 9999999
        for move in gameState.getLegalActions(turn):
            nextState = gameState.generateSuccessor(turn, move)
            result = self.minimax(nextState, depth-1, (turn+1) % numAgents, numAgents)
            if turn == 0 and value < result[1]:
                best_move = move
                value = result[1]
            if turn != 0 and value > result[1]:
                best_move = move
                value = result[1]
        return best_move, value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()
        alpha = -99999999
        beta = 99999999
        ab_move = self.ab(gameState, (self.depth * numAgents), 0, numAgents, alpha, beta)
        return ab_move[0]

    def ab(self, gameState, depth, turn, numAgents, alpha, beta):

        best_move = None

        # Base Case: (1) state is terminal (2) depth is d
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return best_move, self.evaluationFunction(gameState)
        if turn == 0: # MAX
            value = -9999999
        else: # MIN
            value = 9999999
        for move in gameState.getLegalActions(turn):
            nextState = gameState.generateSuccessor(turn, move)
            result = self.ab(nextState, depth-1, (turn+1) % numAgents, numAgents, alpha, beta)
            if turn == 0 and value < result[1]:
                best_move = move
                value = result[1]
                if value >= beta:
                    return best_move, value
                alpha = max(alpha, value)
            if turn != 0 and value > result[1]:
                best_move = move
                value = result[1]
                if value <= alpha:
                    return best_move, value
                beta = min(beta, value)
        return best_move, value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        numAgents = gameState.getNumAgents()
        exmax_move = self.exmax(gameState, (self.depth * numAgents), 0, numAgents)
        return exmax_move[0]

    def exmax(self, gameState, depth, turn, numAgents):

        best_move = None

        # Base Case: (1) state is terminal (2) depth is d
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return best_move, self.evaluationFunction(gameState)
        if turn == 0:  # MAX
            value = -9999999
        else:  # CHANCE
            value = 0
        for move in gameState.getLegalActions(turn):
            nextState = gameState.generateSuccessor(turn, move)
            result = self.exmax(nextState, depth - 1, (turn + 1) % numAgents, numAgents)
            if turn == 0 and value < result[1]: # MAX's TURN
                best_move = move
                value = result[1]
            if turn != 0: # CHANCE's TURN
                n = len(gameState.getLegalActions(turn))
                value = value + ((1.0 / n) * result[1])
        return best_move, value

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Took hint from Assignment and did a linear combo of features.
                Weights set to 1.
                Computed a bunch of things, but did not use all variables.
                Ended up with VAL = Score - distance to closest ghost + 1/distance to closest capsule
                                    + 1/distance to closest food dot
                The closer a ghost, the worse the VAL.
                The higher the game score, the better the VAL

    """
    cgs = currentGameState

    # compute a bunch of important variables

    v_score = cgs.getScore()

    pacman_pos = cgs.getPacmanPosition()
    ghost_pos = [g.getPosition() for g in cgs.getGhostStates()]

    v_ghost_dist = [(abs(pacman_pos[0] - g[0]) + abs(pacman_pos[1] - g[1])) for g in ghost_pos] # NOTE: a list of v's

    v_capsules_dist = [(abs(pacman_pos[0] - c[0]) + abs(pacman_pos[1] - c[1])) for c in cgs.getCapsules()]
    if len(v_capsules_dist) != 0:
        v_mincapsuledist = min(v_capsules_dist)
    else:
        v_mincapsuledist = 0.0000001

    food = cgs.getFood()
    v_foodcount = food.count()

    v_minfooddist = min([(abs(pacman_pos[0] - f[0]) + abs(pacman_pos[1] - f[1])) for f in food])

    scaredtimes = [g.scaredTimer for g in cgs.getGhostStates()]
    goEatGhosts = [time>3 for time in scaredtimes] # should be an array of bools corresponding to each ghost


    # Vectorize

    variableVector = [v_score] + [min(v_ghost_dist)] + [1.0/v_mincapsuledist] + [1.0/v_minfooddist]

    return 1.0*variableVector[0] - 1.0*variableVector[1] + 1.0*variableVector[2] + 1.0*variableVector[3]


# Abbreviation
better = betterEvaluationFunction


