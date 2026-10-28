class MonteCarlo:
    def __init__(
        self, generate_episode, actions, num_episodes, gamma=0.95, epsilon=0.1, Q=None
    ):
        self.generate_episode = generate_episode
        self.actions = actions
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = len(actions)
        self.Q = {} if Q is None else Q

    def evaluate_policy_first_visit_mc(self):
        V = {}
        N = {}
        for _ in range(self.num_episodes):
            episode = self.generate_episode(self.Q, epsilon=self.epsilon)
            returns = self.computeReturns(episode)
            visited = set()
            for t, (state, action, reward) in enumerate(episode):
                if state in visited:
                    continue
                visited.add(state)
                if state not in V:
                    V[state] = 0.0
                    N[state] = 0

                N[state] += 1
                V[state] += 1 / N[state] * (returns[t] - V[state])
        return V

    def evaluate_policy_every_visit_mc(self):
        V = {}
        N = {}

        for _ in range(self.num_episodes):
            episode = self.generate_episode(self.Q, epsilon=self.epsilon)
            returns = self.computeReturns(episode)

            for t, (state, action, reward) in enumerate(episode):
                if state not in V:
                    V[state] = 0.0
                    N[state] = 0
                N[state] += 1
                V[state] += 1 / N[state] * (returns[t] - V[state])
        return V

    def computeReturns(self, episode):
        returns = []
        G = 0
        for state, action, reward in reversed(episode):
            G = reward + G * self.gamma
            returns.append(G)
        return returns[::-1]

    # note: We can't reuues policy evaluation unlike in dynamic-programming is because we update before the value converges.
    def monteCarloControl(self):
        """
        This is following first-visit on-policy Monte Carlo Control
        """
        N = {}
        Q = self.Q
        for  _ in range(self.num_episodes):
            episode = self.generate_episode(Q,self.epsilon)
            returns = self.computeReturns(episode)
            visited =set()
            for t,(state,action,reward) in enumerate(episode):
                if (state,action) in visited:
                    continue
                visited.add((state,action))
                if state not in Q:
                    Q[state] = {a:0.0 for a in self.actions}        
                if state not in N:
                    N[state] = {a:0 for a in self.actions}
                
                N[state][action]+=1
                Q[state][action] += 1/N[state][action] * (returns[t] -Q[state][action])
        return Q
