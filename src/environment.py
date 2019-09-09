'''
Implementation of a basic RL environment.

Rewards are all normal.
Transitions are multinomial.

author: iosband@stanford.edu
'''

import numpy as np

#-------------------------------------------------------------------------------


class Environment(object):
    '''General RL environment'''

    def __init__(self):
        pass

    def reset(self):
        pass

    def advance(self, action):
        '''
        Moves one step in the environment.

        Args:
            action

        Returns:
            reward - double - reward
            newState - int - new state
            pContinue - 0/1 - flag for end of the episode
        '''
        return 0, 0, 0


#-------------------------------------------------------------------------------


class TabularMDP(Environment):
    '''
    Tabular MDP

    R - dict by (s,a) - each R[s,a] = (meanReward, sdReward)
    P - dict by (s,a) - each P[s,a] = transition vector size S
    '''

    def __init__(self, nState, nAction, epLen):
        '''
        Initialize a tabular episodic MDP

        Args:
            nState  - int - number of states
            nAction - int - number of actions
            epLen   - int - episode length

        Returns:
            Environment object
        '''

        self.nState = nState
        self.nAction = nAction
        self.epLen = epLen

        self.timestep = 0
        self.state = 0

        # Now initialize R and P
        self.R = {}
        self.P = {}
        for state in range(nState):
            for action in range(nAction):
                self.R[state, action] = (1, 1)
                self.P[state, action] = np.ones(nState) / nState


    def reset(self):
        '''Reset the environment'''
        self.timestep = 0
        self.state = 0

    def advance(self, action):
        '''
        Move one step in the environment

        Args:
        action - int - chosen action

        Returns:
        reward - double - reward
        newState - int - new state
        pContinue - 0/1 - flag for end of the episode
        '''
        if self.R[self.state, action][1] < 1e-9:
            # Hack for no noise
            reward = self.R[self.state, action][0]
        else:
            reward = np.random.normal(loc=self.R[self.state, action][0],
                                      scale=self.R[self.state, action][1])
        newState = np.random.choice(self.nState, p=self.P[self.state, action])

        # Update the environment
        self.state = newState
        self.timestep += 1

        if self.timestep == self.epLen:
            pContinue = 0
            self.reset()
        else:
            pContinue = 1

        return reward, newState, pContinue

    def compute_qVals(self):
        '''
        Compute the Q values for the environment

        Args:
            NULL - works on the TabularMDP

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        qVals = {}
        qMax = {}

        qMax[self.epLen] = np.zeros(self.nState)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction)

                for a in range(self.nAction):
                    qVals[s, j][a] = self.R[s, a][0] + np.dot(self.P[s, a], qMax[j + 1])

                qMax[j][s] = np.max(qVals[s, j])
        return qVals, qMax


#-------------------------------------------------------------------------------
# Benchmark environments


def make_riverSwim(epLen=20, nState=6):
    '''
    Makes the benchmark RiverSwim MDP.

    Args:
        NULL - works for default implementation

    Returns:
        riverSwim - Tabular MDP environment
    '''
    nAction = 2
    R_true = {}
    P_true = {}

    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[0, 0] = (5. / 1000, 0)
    R_true[nState - 1, 1] = (1, 0)

    # Transitions
    for s in range(nState):
        P_true[s, 0][max(0, s-1)] = 1.

    for s in range(1, nState - 1):
        P_true[s, 1][min(nState - 1, s + 1)] = 0.35
        P_true[s, 1][s] = 0.6
        P_true[s, 1][max(0, s-1)] = 0.05

    P_true[0, 1][0] = 0.4
    P_true[0, 1][1] = 0.6
    P_true[nState - 1, 1][nState - 1] = 0.6
    P_true[nState - 1, 1][nState - 2] = 0.4

    riverSwim = TabularMDP(nState, nAction, epLen)
    riverSwim.R = R_true
    riverSwim.P = P_true
    riverSwim.reset()

    return riverSwim

def make_deterministicChain(nState, epLen):
    '''
    Creates a deterministic chain MDP with two actions.

    Args:
        nState - int - number of states
        epLen - int - episode length

    Returns:
        chainMDP - Tabular MDP environment
    '''
    nAction = 2

    R_true = {}
    P_true = {}

    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[0, 0] = (0, 1)
    R_true[nState - 1, 1] = (1, 1)

    # Transitions
    for s in range(nState):
        P_true[s, 0][max(0, s-1)] = 1.
        P_true[s, 1][min(nState - 1, s + 1)] = 1.

    chainMDP = TabularMDP(nState, nAction, epLen)
    chainMDP.R = R_true
    chainMDP.P = P_true
    chainMDP.reset()

    return chainMDP

def make_stochasticChain(chainLen):
    '''
    Creates a difficult stochastic chain MDP with two actions.

    Args:
        chainLen - int - total number of states

    Returns:
        chainMDP - Tabular MDP environment
    '''
    nState = chainLen
    epLen = chainLen
    nAction = 2
    pNoise = 1. / chainLen

    R_true = {}
    P_true = {}

    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[0, 0] = (0, 1)
    R_true[nState - 1, 1] = (1, 1)

    # Transitions
    for s in range(nState):
        P_true[s, 0][max(0, s-1)] = 1.

        P_true[s, 1][min(nState - 1, s + 1)] = 1. - pNoise
        P_true[s, 1][max(0, s-1)] += pNoise

    stochasticChain = TabularMDP(nState, nAction, epLen)
    stochasticChain.R = R_true
    stochasticChain.P = P_true
    stochasticChain.reset()

    return stochasticChain


def make_stochasticGrid(gridSize,epLen,pNoise,rewardVar):
    '''
    Creates a difficult stochastic grid MDP with four actions.

    Args:
        gridSize - int - the size of the grid (total gridSize x gridSize +1 states - the last state is a terminal state)
        epLen - int - episode length
        pNoise - double - The probability to move randomly instead to the desired location. In [0,1]
        rewardVar - double - The variance of the noise all across the state space

    Returns:
        gridMDP - Tabular MDP environment
    '''
    nState = gridSize*gridSize+1
    epLen = epLen
    nAction = 4

    R_true = {}
    P_true = {}

    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, rewardVar)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    for action in range(nAction):
        R_true[nState - 2, action] = (1, rewardVar) # Reward's location
        R_true[nState - 1, action] = (0, 0) # Terminal state

    # Transitions
    for s in range(nState):
        nextState = [None]*4
        if s >= nState-2: # moving to the terminal state
            for action in range(nAction):
                nextState[action] = nState-1
        else:
            nextState[0] = s-1 if s%gridSize>0 else s #'up'
            nextState[1] = s+1 if (s+1)%gridSize>0 else s  # 'down'
            nextState[2] = s - gridSize if s >= gridSize else s  # 'west'
            nextState[3] = s + gridSize if s+gridSize <= nState-1 else s  # 'east'

        for action in range(nAction):
            for ns in nextState:
                P_true[s, action][ns] += pNoise/4
            P_true[s, action][nextState[action]] += 1-pNoise


    stochasticGrid = TabularMDP(nState, nAction, epLen)
    stochasticGrid.R = R_true
    stochasticGrid.P = P_true
    stochasticGrid.reset()

    return stochasticGrid

def make_stochastic2dChain(gridSize):
    '''
    Creates a difficult stochastic grid MDP with four actions, which is similar to the 1d chain

    Args:
        gridSize - int - the size of the grid (total gridSize x gridSize states

    Returns:
        gridMDP - Tabular MDP environment
    '''
    nState = gridSize*gridSize
    epLen = 2*gridSize-1
    nAction = 4

    pNoise = 1/epLen

    R_true = {}
    P_true = {}

    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # reward for going into the corner at the initial state
    R_true[0, 0] = (0, 1) #'up'
    R_true[0, 2] = (0, 1) #'left'

    # reward for going into the corner at the final state
    R_true[nState - 1, 1] = (1, 1)  # 'down'
    R_true[nState - 1, 3] = (1, 1)  # 'right'

    # Transitions
    for s in range(nState):
        nextState = [None]*4
        if s >= nState-2: # moving to the terminal state
            for action in range(nAction):
                nextState[action] = nState-1
        else:
            nextState[0] = s-1 if s%gridSize>0 else s #'up'
            nextState[1] = s+1 if (s+1)%gridSize>0 else s  # 'down'
            nextState[2] = s - gridSize if s >= gridSize else s  # 'left'
            nextState[3] = s + gridSize if s+gridSize <= nState-1 else s  # 'right'

        for action in range(nAction):
            # adding noise to move up or left
            P_true[s, action][nextState[0]] += pNoise/2 #'up'
            P_true[s, action][nextState[2]] += pNoise/2 # 'left'
            # the desired direction:
            P_true[s, action][nextState[action]] += 1-pNoise


    stochastic2dChain = TabularMDP(nState, nAction, epLen)
    stochastic2dChain.R = R_true
    stochastic2dChain.P = P_true
    stochastic2dChain.reset()

    return stochastic2dChain
def make_bootDQNChain(nState=6, epLen=15, nAction=2):
    '''
    Creates the chain from Bootstrapped DQN

    Returns:
        bootDQNChain - Tabular MDP environment
    '''
    R_true = {}
    P_true = {}

    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[0, 0] = (0.01, 1)
    R_true[nState - 1, 1] = (1, 1)

    # Transitions
    for s in range(nState):
        P_true[s, 0][max(0, s-1)] = 1.

        P_true[s, 1][min(nState - 1, s + 1)] = 0.5
        P_true[s, 1][max(0, s-1)] = 0.5

    bootDQNChain = TabularMDP(nState, nAction, epLen)
    bootDQNChain.R = R_true
    bootDQNChain.P = P_true
    bootDQNChain.reset()

    return bootDQNChain


def make_hardBanditMDP(epLen, gap=0.01, nAction=2, pSuccess=0.5):
    '''
    Creates a difficult bandit-style MDP which is hard to distinguish.

    Args:
        epLen - int
        gap - double - how much better is best arm
        nAction - int - how many actions

    Returns:
        hardBanditMDP - Tabular MDP environment
    '''
    nState = 3

    R_true = {}
    P_true = {}

    for a in range(nAction):
        # Rewards are independent of action
        R_true[0, a] = (0.5, 1)
        R_true[1, a] = (1, 0)
        R_true[2, a] = (0, 0)

        # Transitions are like a bandit
        P_true[0, a] = np.array([0, pSuccess, 1 - pSuccess])
        P_true[1, a] = np.array([0, 1, 0])
        P_true[2, a] = np.array([0, 0, 1])

    # The first action is a better action though
    P_true[0, 0] = np.array([0, pSuccess + gap, 1 - (pSuccess + gap)])

    hardBanditMDP = TabularMDP(nState, nAction, epLen)
    hardBanditMDP.R = R_true
    hardBanditMDP.P = P_true
    hardBanditMDP.reset()

    return hardBanditMDP




def make_stateBanditMDP(stateMul, gap=0.1):
    '''
    Creates a bandit-style MDP which examines dependence on states.

    Args:
        epLen - int
        gap - double - how much better is best arm
        nAction - int - how many actions

    Returns:
        stateBanditMDP - Tabular MDP environment
    '''
    epLen = 2
    nAction = 2
    nState = 1 + 2 * stateMul

    R_true = {}
    P_true = {}

    for a in range(nAction):
        R_true[0, a] = (0, 0)
        P_true[0, a] = np.zeros(nState)

        for k in range(stateMul):
            for i in range(2):
                s = 1 + (2 * k) + i
                P_true[s, a] = np.zeros(nState)
                P_true[s, a][s] = 1
                R_true[s, a] = (1-i, 0)

    # Important piece is where the transitions go
    P_true[0, 0] = np.ones(nState) / (nState - 1)
    P_true[0, 0][0] = 0

    # Rewarding states
    inds = (np.arange(nState) % 2) > 0
    P_true[0, 1][inds] = (0.5 + gap) / stateMul
    P_true[0, 1][-inds] = (0.5 - gap) / stateMul
    P_true[0, 1][0] = 0

    stateBanditMDP = TabularMDP(nState, nAction, epLen)
    stateBanditMDP.R = R_true
    stateBanditMDP.P = P_true
    stateBanditMDP.reset()

    return stateBanditMDP

def make_confidenceMDP(stateMul, gap=0.1):
    '''
    Creates a bandit-style MDP which examines dependence on states.

    Args:
        epLen - int
        gap - double - how much better is best arm
        nAction - int - how many actions

    Returns:
        confidenceMDP - Tabular MDP environment
    '''
    epLen = 2
    nAction = 1
    nState = 1 + 2 * stateMul

    R_true = {}
    P_true = {}

    for a in range(nAction):
        R_true[0, a] = (0, 0)
        P_true[0, a] = np.zeros(nState)

        for k in range(stateMul):
            for i in range(2):
                s = 1 + (2 * k) + i
                P_true[s, a] = np.zeros(nState)
                P_true[s, a][s] = 1
                R_true[s, a] = (1-i, 0)

    # Important piece is where the transitions go
    P_true[0, 0] = np.ones(nState) / (nState - 1)
    P_true[0, 0][0] = 0

    # Rewarding states
    inds = (np.arange(nState) % 2) > 0

    confidenceMDP = TabularMDP(nState, nAction, epLen)
    confidenceMDP.R = R_true
    confidenceMDP.P = P_true
    confidenceMDP.reset()

    return confidenceMDP


def make_HconfidenceMDP(epLen):
    '''
    Creates a H-dependence bandit confidence.

    Args:
        epLen - int
        gap - double - how much better is best arm
        nAction - int - how many actions

    Returns:
        hardBanditMDP - Tabular MDP environment
    '''
    nState = 3

    R_true = {}
    P_true = {}

    # Rewards are independent of action
    R_true[0, 0] = (0.5, 0)
    R_true[1, 0] = (1, 0)
    R_true[2, 0] = (0, 0)

    # Transitions are like a bandit
    P_true[0, 0] = np.array([0, 0.5, 0.5])
    P_true[1, 0] = np.array([0, 1, 0])
    P_true[2, 0] = np.array([0, 0, 1])

    hardBanditMDP = TabularMDP(nState, 1, epLen)
    hardBanditMDP.R = R_true
    hardBanditMDP.P = P_true
    hardBanditMDP.reset()

    return hardBanditMDP
