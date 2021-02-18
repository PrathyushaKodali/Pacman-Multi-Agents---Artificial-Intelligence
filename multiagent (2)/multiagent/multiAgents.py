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

        "*** YOUR CODE HERE ***"
        # set the initial distance to food with infinity
        Food_distance = float("inf")
        # define the current score as 0
        current_score = 0

        # loop through food pallets
        for food_pallet in newFood.asList():
            # calculate the manhattan distance between the food pallet and Pacman
            M_distance_food = manhattanDistance(food_pallet,newPos)
            # Find the pallet which is closer to Pacman
            Food_distance = min([M_distance_food , Food_distance])

        # set the initial ghost distance to infinity
        Ghost_distance = float("inf")

        # loop through all the ghosts
        for ghost in newGhostStates:
            # calculate the manhattan distance between the ghost and Pacman
            M_distance_ghost = manhattanDistance(newPos, ghost.getPosition())
            # Find the ghost which is closer to Pacman
            Ghost_distance = min([M_distance_ghost, Ghost_distance])

        if(Ghost_distance<= 1): # if distance between Pacman and ghost is less than or equal to 1
                current_score = current_score - 1000  # to avoid Pacman from losing decrease the current score

        Food_distance_score = 1.0/ Food_distance
        return successorGameState.getScore() + Food_distance_score + current_score


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

        self.get_max_Pacman(gameState, 0, 0)  # Call the get max value function
        return self.bestAction    # returns the best action that Pacman can perform

    def minmax(self,gameState, depthNode, agent_pos):
      # return evaluation function if any of the three conditions are met
        if gameState.isLose() or gameState.isWin() or depthNode == self.depth:
            return self.evaluationFunction(gameState)
        if agent_pos == 0:  # call get max Pacman function
            return self.get_max_Pacman(gameState,depthNode,agent_pos)
        else:  # call min ghost function
            return self.get_min_Ghost(gameState,depthNode,agent_pos)

    def get_min_Ghost(self,gameState, depthNode, agent_pos): # Get Min node value
        start = float("inf"), Directions.STOP  # Set the default value of distance to infinity and STOP as direction
        min_value_list= []
        min_value_list.append(start)     # Add initial node to the list
        legal_actions = gameState.getLegalActions(agent_pos)  # get legal actions of the agent in the present state
        if not legal_actions:
            return self.evaluationFunction(gameState)
        for action in legal_actions:                    # Loop through all legal actions
            successor = gameState.generateSuccessor(agent_pos, action)
            if agent_pos + 1 == gameState.getNumAgents(): # indicates last ghost
                Ghost_min = self.minmax(successor, depthNode+1, 0)
            else:
                Ghost_min = self.minmax(successor, depthNode, agent_pos + 1)
            min_value_list.append((Ghost_min , action))   # Append next state to the list
            (self.Ghost_min, self.bestAction) =  min(min_value_list)
        return self.Ghost_min

    def get_max_Pacman(self, gameState, depthNode, agent_pos):  # Get Max node value
        start = float("-inf"), Directions.STOP  # Set the default value of distance to minus infinity and STOP as direction
        max_value_list = []
        max_value_list.append(start)  # Add initial node to the list
        legal_actions = gameState.getLegalActions(agent_pos)  # get legal actions of the agent in the present state
        if not legal_actions:
            return self.evaluationFunction(gameState)
        for action in legal_actions:                    # Loop through all legal actions
            successor = gameState.generateSuccessor(agent_pos, action)
            Pacman_max = self.minmax(successor, depthNode, 1)
            max_value_list.append((Pacman_max, action))  # Append next state to the list
            (self.Pacman_max , self.bestAction) = max(max_value_list)
        return self.Pacman_max

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.get_max_Pacman(gameState, 0, 0, -float("inf"), float("inf"))
        return self.bestActionforAlphabeta

    def alphabeta(self, gameState, depthNode, agent_pos,alpha, beta):
        # return evaluation function if any of the three conditions are met
        if gameState.isLose() or gameState.isWin() or depthNode == self.depth:
            return self.evaluationFunction(gameState)
        if agent_pos == 0:  # call get max Pacman function
            return self.get_max_Pacman(gameState,depthNode,agent_pos,alpha, beta)
        else:  # call min ghost function
            return self.get_min_Ghost(gameState,depthNode,agent_pos,alpha, beta)


    def get_max_Pacman(self, gameState, depthNode,agent_pos, alpha, beta):
        start = float("-inf"), Directions.STOP  # Set the default value of distance to minus infinity and STOP as direction
        max_value_list = []
        max_value_list.append(start)  # Add initial node to the list
        legal_actions = gameState.getLegalActions(agent_pos)  # get legal actions of the agent in the present state
        if not legal_actions:
            return self.evaluationFunction(gameState)
        for action in legal_actions:                    # Loop through all legal actions
            successor = gameState.generateSuccessor(agent_pos, action)
            Pacman_max = self.alphabeta(successor, depthNode, 1, alpha, beta)
            max_value_list.append((Pacman_max, action))  # Append next state to the list
            (self.maxvalueforAlphabeta, self.bestActionforAlphabeta) = max(max_value_list)
            if(self.maxvalueforAlphabeta > beta):
                return self.maxvalueforAlphabeta  # return the maxvalueforAlphabeta if it is greater than beta
            alpha = max(self.maxvalueforAlphabeta, alpha) # Assign max of the two values to alpha
        return self.maxvalueforAlphabeta

    def get_min_Ghost(self, gameState, depthNode,agent_pos, alpha, beta):
        start = float("inf"), Directions.STOP  # Set the default value of distance to infinity and STOP as direction
        min_value_list = []
        min_value_list.append(start)  # Add initial node to the list
        legal_actions = gameState.getLegalActions(agent_pos)  # get legal actions of the agent in the present state
        if not legal_actions:
            return self.evaluationFunction(gameState)
        for action in legal_actions:                    # Loop through all legal actions
            successor = gameState.generateSuccessor(agent_pos, action)
            if agent_pos + 1 == gameState.getNumAgents(): # if last ghost
                Ghost_min = self.alphabeta(successor, depthNode+1, 0, alpha, beta)
            else:
                Ghost_min = self.alphabeta(successor, depthNode, agent_pos + 1, alpha, beta)

            min_value_list.append((Ghost_min , action))   # Append next state to the list
            (self.minvalueforAlphabeta, self.bestActionforAlphabeta) = min(min_value_list)
            if (self.minvalueforAlphabeta < alpha):
                return self.minvalueforAlphabeta    # return the minvalueforAlphabeta if it is less than alpha
            beta = min(beta, self.minvalueforAlphabeta)  # Assign min of the two values to beta

        return self.minvalueforAlphabeta


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
        "*** YOUR CODE HERE ***"
        self.get_max_Pacman(gameState, 0, 0)  # Call the get max value function
        return self.bestActionforExpectimax    # returns the best action that Pacman can perform

    def expectimax(self, gameState, depthNode, agent_pos):
        # return evaluation function if any of the three conditions are met
        if gameState.isLose() or gameState.isWin() or depthNode == self.depth:
            return self.evaluationFunction(gameState)
        if agent_pos == 0:  # call get max Pacman function
            return self.get_max_Pacman(gameState,depthNode,agent_pos)
        else:  # call min ghost function
            return self.get_expecti_Ghost(gameState,depthNode,agent_pos)


    def get_max_Pacman(self, gameState, depthNode,agent_pos):
        start = float("-inf"), Directions.STOP  # Set the default value of distance to minus infinity and STOP as direction
        max_value_list = []
        max_value_list.append(start)  # Add initial node to the list
        legal_actions = gameState.getLegalActions(agent_pos)  # get legal actions of the agent in the present state
        if not legal_actions:
            return self.evaluationFunction(gameState)
        for action in legal_actions:                    # Loop through all legal actions
            successor = gameState.generateSuccessor(agent_pos, action)
            Pacman_max = self.expectimax(successor, depthNode, 1)
            max_value_list.append((Pacman_max, action))  # Append next state to the list
            (self.maxvalueforExpectimax, self.bestActionforExpectimax) = max(max_value_list)
        return self.maxvalueforExpectimax

    def get_expecti_Ghost(self, gameState, depthNode,agent_pos):
        no_of_states = len(gameState.getLegalActions(agent_pos))
        avg_score = 0
        legal_actions = gameState.getLegalActions(agent_pos)  # get legal actions of the agent in the present state
        for action in legal_actions:                    # Loop through all legal actions
            successor = gameState.generateSuccessor(agent_pos, action)
            if agent_pos + 1 == gameState.getNumAgents(): # if last ghost
                Ghost_expecti_val = self.expectimax(successor, depthNode+1, 0)
            else:
                Ghost_expecti_val = self.expectimax(successor, depthNode, agent_pos + 1) + avg_score
            avg_score = Ghost_expecti_val + avg_score
        return avg_score * (1.0 / no_of_states)



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    eval_score = 0    #  Initialize score value to zero
    # get the current states of Pacman, ghost and food
    agent_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostStates()
    food_position = currentGameState.getFood()
    food_distance = float("inf")

    # increase score by 10 if the Pacman position = food
    if food_position[agent_position[0]][agent_position[1]]:
        eval_score += 10

    #Loop through every food pallet
    for food_pallet in food_position.asList():
        # calculate distance between pacman and food pallet
        M_distance_food= manhattanDistance(food_pallet,agent_position)
        food_distance = min([food_distance, M_distance_food])
        if food_pallet in currentGameState.getCapsules(): # if food in capsule position increase the score
            eval_score = eval_score + 150

    ghost_distance = float("inf")
    remaining_food_pallets = len(food_position.asList())

    for ghost in ghost_positions:
        # calculate distance between pacman and ghost
        M_distance_ghost = manhattanDistance(ghost.getPosition(),agent_position)
        ghost_distance = min([ghost_distance, M_distance_ghost])
        if (ghost_distance <= 1):
            if ghost.scaredTimer:  # increase the score if the ghost is scared , else decrease it
                eval_score = eval_score + 2000
            else:
                eval_score = eval_score - 1000

    if(agent_position == ghost.getPosition):
        return float("-inf")
    return currentGameState.getScore() + (1.0 / food_distance) -2* remaining_food_pallets + eval_score

# Abbreviation
better = betterEvaluationFunction
