import numpy as np
from policy_evaluation import PolicyEvaluation


class ValueIteration:

    def __init__(self,states,actions,transitions,rewards,gamma=0.95,epsilon=1e-4,policy=None):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy

        self.state_to_idx = {s:i for i,s in enumerate(states)}
        self.n_states = len(states)

        self.actions_to_idx = {a:i for i,a in enumerate(actions)}


    def valueIteration(self):

        V_0 = np.zeros(self.n_states)
        delta = float('inf')

        while delta > (1-self.gamma)* self.epsilon:
            V_new = np.copy(V_0)
            for s in self.states:
                i = self.state_to_idx[s]

                possible_V = 0
                for a in self.actions:
                    reward = self.rewards[(s,a)]
                    q = reward + self.gamma * sum(prob * V_0[self.state_to_idx[possible_state]] for possible_state, prob in self.transitions[(s, a)])
                    if q >= possible_V:
                        possible_V = q
                V_new[i] = possible_V
            delta = np.max(np.abs(V_new - V_0))
            V_0 = V_new

        #Extract a greedy policy
        policy_to_return = {}
        for s in self.states:
            V_max = -float('inf')
            best_action = None
            for a in self.actions:
                q = self.rewards[(s, a)] + self.gamma * sum(prob * V_0[self.state_to_idx[possible_state]] for possible_state, prob in self.transitions[(s, a)])
                if q>= V_max:
                    V_max = q
                    best_action = a
            policy_to_return[s] = best_action
        
        return V_0,policy_to_return

    def policyIteration(self):
        if self.policy == None:
            raise ValueError("There is no valid policy to iterate over")

        # Policy Evaluation
        while True:
            policy_new = {}
            evaluator = PolicyEvaluation(
                        policy=self.policy, 
                        transition= self.transitions,  # assuming env stores it here
                        reward=self.rewards,          # assuming env stores it here
                        gamma=self.gamma
                    )
            value = evaluator.calculate_fixed_point_view()
            for s in self.states:
                best_action = self.policy[s]
                i = self.state_to_idx[s]
                V_max = value[i]

                for a in self.actions:
                    q = self.rewards[(s, a)] + self.gamma * sum(prob * value[self.state_to_idx[possible_state]] for possible_state, prob in self.transitions[(s, a)])
                    if q > V_max:
                        V_max = q
                        best_action = a
                policy_new[s] = best_action

            if policy_new == self.policy:
                break
            self.policy = policy_new

        return policy_new
