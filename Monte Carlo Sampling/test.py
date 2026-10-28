import random

import numpy as np
from bandit_learning import multiArmBandit
from monte_carlo import MonteCarlo

# =====================================
# Multi Arm Bandit Test
# =====================================
true_means = np.array([0, 5, 10, 20])
a = multiArmBandit(true_means, num_steps=10000, method="ucb", epsilon=0.1)
Q, N, rewards, actions = a.run_bandit()
print(Q)
print(N)
print(np.mean(rewards))


# =====================================
# Monte Carlo Test
# =====================================
def generate_episode(Q, epsilon=0.1, terminal_states=None):
    """
    Generates an episode using the current nested Q-table and epsilon-greedy action selection.
    Environment: 10 states (0 to 9), terminal state: 10.
    Start state is randomly generated from non-terminal states.
    """
    actions = ["right", "left"]
    if terminal_states is None:
        terminal_states = {10}

    # Randomly select a starting state from non-terminal states (0 to 9)
    valid_start_states = [s for s in range(10) if s not in terminal_states]
    state = random.choice(valid_start_states)
    episode = []

    while state not in terminal_states:
        # Epsilon-greedy action selection
        if random.random() < epsilon or not Q:
            action = random.choice(actions)
        else:
            # Choose the action that maximizes Q[state][action] safely
            action = max(actions, key=lambda a: Q.get(state, {}).get(a, 0.0))

        # Simple toy transition dynamics
        if action == "right":
            next_state = state + 1
            reward = 1.0 if next_state in terminal_states else 0.0
        else:  # "left"
            next_state = max(0, state - 1)
            reward = -0.1  # Small penalty for moving left/wall bumping

        episode.append((state, action, reward))
        state = next_state

        # Safety break to prevent infinite loops
        if len(episode) > 50:
            break

    return episode


Q = {
    0: {"left": -0.52, "right": 0.15},
    1: {"left": -0.45, "right": 0.28},
    2: {"left": -0.38, "right": 0.42},
    3: {"left": -0.29, "right": 0.58},
    4: {"left": -0.20, "right": 0.71},
    5: {"left": -0.15, "right": 0.82},
    6: {"left": -0.10, "right": 0.90},
    7: {"left": -0.05, "right": 0.95},
    8: {"left": 0.0, "right": 0.98},
    9: {"left": 0.10, "right": 1.00},
}
# 1. Instantiate Monte Carlo
mc = MonteCarlo(
    generate_episode=generate_episode,
    actions=["left", "right"],
    num_episodes=1000,
    gamma=0.9,
    Q=Q,
)

# 2. Test First-Visit MC Policy Evaluation
print("Running First-Visit MC Policy Evaluation...")
v_first = mc.evaluate_policy_first_visit_mc()
print("Estimated V(s) [First-Visit]:", v_first)

print("\n" + "=" * 40 + "\n")

# 3. Test Every-Visit MC Policy Evaluation
print("Running Every-Visit MC Policy Evaluation...")
v_every = mc.evaluate_policy_every_visit_mc()
print("Estimated V(s) [Every-Visit]:", v_every)

# 4. Test
print("Running Monte Carlo Control...")
optimal_q = mc.monteCarloControl()

print("\nLearned Q-Values Q(state, action):")
# Unpack state and the inner action dictionary
for state, action_dict in sorted(optimal_q.items()):
    for action, val in action_dict.items():
        print(f"State {state}, Action '{action}': {val:.4f}")

print("\nDerived Optimal Policy:")
for state, action_dict in sorted(optimal_q.items()):
    # Find the action with the highest Q-value in the inner dictionary
    best_action = max(action_dict, key=action_dict.get)
    print(f"State {state} -> Best Action: '{best_action}'")
