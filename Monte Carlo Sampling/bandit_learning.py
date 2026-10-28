import numpy as np
import math

class multiArmBandit:

    def __init__(self,true_means,num_steps,method="epsilon_greedy",epsilon=0.1,c=2.0):
        
        """
        true_means: the average reward for each action
        num_steps: the number of time the arm is pulled
        method: 'greedy', 'epsilon-greedy', or 'ucb'
        """

        self.true_means = true_means
        self.num_steps = num_steps
        self.method= method
        self.epsilon = epsilon
        self.c = c

        self.num_actions = len(self.true_means)


    def run_bandit(self):

        Q = np.zeros(self.num_actions)
        N = np.zeros(self.num_actions)

        rewards = []
        actions = []

        for t in range(1, self.num_steps+1):

            if self.method == "greedy":
                a = self.greedy(Q)

            if self.method == "epsilon-greedy":
                a = self.epsilon_greedy_action(Q,self.epsilon)

            if self.method == "ucb":
                a = self.ucb_action(Q,N,len(actions),self.c)

            rng = np.random.default_rng()
            r = rng.normal(loc=self.true_means[a])

            rewards.append(r)
            actions.append(a)

            #Update Q
            N[a] += 1
            alpha = 1/N[a]
            Q[a] = Q[a] + alpha*(r - Q[a])
        
        return Q, N, rewards, actions

    def greedy(self,Q):
        return int(np.argmax(Q))

    def epsilon_greedy_action(self,Q,epsilon):

        best_action = np.argmax(Q)
        
        if np.random.random() < epsilon:
            #other_actions: index of actions that is not the best action
            other_actions = [i for i in range(self.num_actions) if i!= best_action]
            return np.random.choice(other_actions)
        else:
            return int(best_action)

    def ucb_action(self,Q,n,t,c):
        for i in range(len(Q)):
            if n[i] == 0:
                return i
            
        ucb_list = []
        for i in range(len(Q)):
            ucb = Q[i] + c * (math.log(t) / n[i]) ** 0.5
            ucb_list.append(ucb)
        return np.argmax(ucb_list)
    


