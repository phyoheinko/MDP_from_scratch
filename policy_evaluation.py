import numpy as np

class PolicyEvaluation:

    def __init__(self, policy=None, transition = None, reward = None, gamma = 0.75, epsilon=1e-4):
        self.policy = policy
        self.transition = transition
        self.reward = reward
        self.gamma = gamma
        self.epsilon = epsilon

        self.states = list(self.policy.keys())
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.n_states = len(self.states)

    def calculate_linear_view(self):
        
        transition_matrix = np.zeros((self.n_states, self.n_states))
        reward_matrix = np.zeros(self.n_states)

        for s in self.states:
            i = self.state_to_idx[s]
            a = self.policy[s]

            reward_matrix[i] = self.reward[(s,a)]

            for next_state, prob in self.transition[(s,a)]:
                j = self.state_to_idx[next_state]
                transition_matrix[i,j] += prob
            

        I = np.eye(self.n_states)
        V_array = np.linalg.solve(I - self.gamma * transition_matrix, reward_matrix)
        V_pi = {s: V_array[self.state_to_idx[s]] for s in self.states}
        return V_pi

    def calculate_fixed_point_view(self):
        difference = (1-self.gamma)*self.epsilon
        V = np.zeros(self.n_states)
        delta = float('inf')

        while delta > difference:
            V_new = np.zeros(self.n_states)
            delta = 0.0 

            for s in self.states:
                i = self.state_to_idx[s]
                a = self.policy[s]

                r = self.reward[(s,a)]
                possible_V = []

                value_s = r
                for next_state, prob in self.transition[(s,a)]:
                    j = self.state_to_idx[next_state]
                    value_s_prime = V[j]
                    value_s += self.gamma * prob *  value_s_prime
                V_new[i] = value_s
            delta = max(delta, abs(V_new[i] - V[i]))
            V = V_new

        return V
                
                

