import numpy as np

class PolicyEvaluation:

    def __init__(self, policy=None, transition = None, reward = None, gamma = 0.75):
        self.policy = policy
        self.transition = transition
        self.reward = reward
        self.gamma = gamma

    def calculate(self):
        states = list(self.policy.keys())
        state_to_idx = {s: i for i, s in enumerate(states)}
        n_states = len(states)

        transition_matrix = np.zeros((n_states, n_states))
        reward_matrix = np.zeros(n_states)

        for s in states:
            i = state_to_idx[s]
            a = self.policy[s]

            reward_matrix[i] = self.reward[(s,a)]

            for next_state, prob in self.transition[(s,a)]:
                j = state_to_idx[next_state]
                transition_matrix[i,j] += prob
            

        I = np.eye(n_states)
        V_array = np.linalg.solve(I - self.gamma * transition_matrix, reward_matrix)
        V_pi = {s: V_array[state_to_idx[s]] for s in states}
        return V_pi

