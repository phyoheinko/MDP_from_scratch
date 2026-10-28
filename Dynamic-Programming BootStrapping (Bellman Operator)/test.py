import pygame
from mdp import SurpriseQuizEnv
from policy_evaluation import PolicyEvaluation
from valueIteration import ValueIteration

# Initialize environment with human rendering
env = SurpriseQuizEnv(render_mode="human")
obs, info = env.reset()

# Extract MDP components from the environment
states = env.unwrapped.state_names
actions = env.unwrapped.action_names
transitions = env.unwrapped.transition
rewards = env.unwrapped.reward
gamma = 0.75

print("--- 1. RUNNING VALUE ITERATION ---")
vi_solver = ValueIteration(
    states=states,
    actions=actions,
    transitions=transitions,
    rewards=rewards,
    policy = {'L': 'Relax', 'M': 'Relax', 'P': 'Relax'}
)
optimal_V, optimal_policy_vi = vi_solver.valueIteration()
print("Optimal Values (V):", optimal_V)
print("Optimal Policy (Value Iteration):", optimal_policy_vi)


print("\n--- 2. RUNNING POLICY ITERATION ---")
# Start with an initial naive policy (e.g., picking the first action for all states)
initial_policy = {s: actions[0] for s in states}

pi_solver = ValueIteration(
    states=states,
    actions=actions,
    transitions=transitions,
    rewards=rewards,
    gamma=gamma,
    policy=initial_policy
)
optimal_policy_pi = pi_solver.policyIteration()
print("Optimal Policy (Policy Iteration):", optimal_policy_pi)


# --- Pygame Simulation Loop (Using Value Iteration Optimal Policy) ---
print("\n--- 3. STARTING PYGAME SIMULATION ---")
active_policy = optimal_policy_vi  # You can switch to optimal_policy_pi here

running = True
for step_num in range(10):
    # Handle window events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not running:
        break

    # Take an action based on the optimal policy
    action = active_policy.get(obs, env.action_space.sample())
    obs, reward, terminated, truncated, info = env.step(action)

    # Update the Pygame window
    env.render()

    print(
        f"Step {step_num + 1} | "
        f"Action: {info['action_name']} | "
        f"New State: {info['state_name']} | "
        f"Reward: {reward}"
    )

    # Wait without blocking Pygame's event handling
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < 600:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not running:
            break

        pygame.time.wait(10)
env.close()