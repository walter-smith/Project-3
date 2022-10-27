# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            values = self.values.copy() # effects computeQValue
            for state in self.mdp.getStates():
                end = float('-inf') # init to worst possible value
                for action in self.mdp.getPossibleActions(state):
                    if end <  self.computeQValueFromValues(state, action):
                        end = self.computeQValueFromValues(state, action)
                end = 0 if end == float('-inf') else end
                values[state] = end # ^^ if nothing found, make 0
            self.values = values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0
        for next, prob in self.mdp.getTransitionStatesAndProbs(state,action):
            value += (self.mdp.getReward(state,action,next)+(self.values[next]*self.discount))*prob
        return value
        # self.mdp.getReward(state,action,next) gets the reward value for that state
        # self.values[next] gets the value of that given square, multiply it by the discount
        # multiply everything by the probability

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        value, new_action = float('-inf'), None # value should be worst possible action, as it's null
        myactions = self.mdp.getPossibleActions(state) # get every possible thing
        if len(myactions) == 0:
            return None # if there are no actions, then we should fail
        for action in myactions:
            if value < self.computeQValueFromValues(state, action): # if our action is better...
                value, new_action = self.computeQValueFromValues(state, action), action # update
        return new_action # return best possible from those presented by getPossibleActions()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        value = 0
        while value < self.iterations:
            for state in self.mdp.getStates():
                qvalue = util.Counter()
                if value >= self.iterations:
                    return
                for action in self.mdp.getPossibleActions(state):
                    qvalue[action] = self.computeQValueFromValues(state, action)
                self.values[state], value = qvalue[qvalue.argMax()], value + 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Step 1: Compute predecessors of all states.

        # Initialize empty dict for all states
        predecessors = {}

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                for nextState, nextAction in self.mdp.getTransitionStatesAndProbs(state, action):
                    if nextState not in predecessors.keys():
                        predecessors[nextState] = set()
                    if nextAction != 0:
                        predecessors[nextState].add(state)

        # Step 2: Initialize an empty priority queue.
        pQueue = util.PriorityQueue()

        # Step 3: For each non-terminal state s, do: (to make the autograder work for this question, 
        # you must iterate over states in the order returned by self.mdp.getStates())
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue
            # a. Find the absolute value of the difference between the current value 
            #    of s in self.values and the highest Q-value across all possible actions 
            #    from s (this represents what the value should be); call this number diff. 
            #    Do NOT update self.values[s] in this step. 
            diff = self.getDiff(s)

            # b. Push s into the priority queue with priority -diff (note that this is negative). 
            #    We use a negative because the priority queue is a min heap, but we want to prioritize 
            #    updating states that have a higher error.
            pQueue.push(s, -diff)

        # Step 4: For iteration in 0, 1, 2, ..., self.iterations - 1, do:

        for _ in range(self.iterations):
            # a. If the priority queue is empty, then terminate.
            if pQueue.isEmpty():
                break

            # b. Pop a state s off the priority queue.
            s = pQueue.pop()

            # c. Update the value of s (if it is not a terminal state) in self.values.
            actions = self.mdp.getPossibleActions(s)
            self.values[s] = max([self.getQValue(s, action) for action in actions])
            
            # d. For each predecessor p of s, do:
            for p in predecessors[s]:
                # i. Find the absolute value of the difference between the current value of p in self.values 
                #    and the highest Q-value across all possible actions from p (this represents what the value 
                #    should be); call this number diff. Do NOT update self.values[p] in this step.
                diff = self.getDiff(p)

                # ii. If diff > theta, push p into the priority queue with priority -diff 
                #     (note that this is negative), as long as it does not already exist in the priority queue 
                #     with equal or lower priority. As before, we use a negative because the priority queue 
                #     is a min heap, but we want to prioritize updating states that have a higher error.   
                if diff > self.theta:
                    pQueue.update(p, -diff)

    # Find the absolute value of the difference between the current value 
    # of s in self.values and the highest Q-value across all possible actions 
    # from s (this represents what the value should be); call this number diff. 
    # Do NOT update self.values[s] in this step
    def getDiff(self, state):
        actions = self.mdp.getPossibleActions(state)
        diff = abs(self.values[state] - max([self.getQValue(state, action) for action in actions]))
        return diff
