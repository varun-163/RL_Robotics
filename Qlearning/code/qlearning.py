import numpy as np
import matplotlib.pyplot as plt
import random

class Gridworld:
    """
    Gridworld environment for the Q-learning agent.
    """
    def __init__(self, penalty=-1.0):
        # Grid dimensions
        self.width = 4
        self.height = 3
        
        # States (using 0-indexed row, col from top-left)
        self.start_state = (2, 0) # Corresponds to (1,1) in problem description
        self.obstacle_state = (1, 1) # Corresponds to (2,2)

        # Corrected 0-indexed coordinates for terminal states
        # (4,3) -> (row=0, col=3), (4,2) -> (row=1, col=3)
        self.terminal_states = [(0, 3), (1, 3)]
        
        # Corrected rewards dictionary
        self.rewards = {
            (0, 3): 1.0,
            (1, 3): penalty,
        }
        self.default_reward = -0.04
        
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.actions = {'N': 0, 'S': 1, 'W': 2, 'E': 3}
        self.action_vectors = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Row, Col: Up, Down, Left, Right
        
        self.action_map = {
            0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'
        }


    def get_state_from_pos(self, pos):
        """Converts (row, col) position to a single integer state."""
        return pos[0] * self.width + pos[1]

    def get_pos_from_state(self, state):
        """Converts integer state back to (row, col) position."""
        return (state // self.width, state % self.width)

    def is_terminal(self, pos):
        """Check if a position is a terminal state."""
        return pos in self.terminal_states

    def is_valid_pos(self, pos):
        """Check if a position is within grid bounds and not an obstacle."""
        r, c = pos
        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return False
        if pos == self.obstacle_state:
            return False
        return True

    def get_reward(self, pos):
        """Get the reward for a given position."""
        return self.rewards.get(pos, self.default_reward)

    def step(self, pos, action):
        """
        Perform a step in the environment with stochastic transitions.
        """
        if self.is_terminal(pos):
            return pos, 0.0

        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        # Correctly define perpendicular actions
        left_of = {0: 2, 1: 3, 2: 1, 3: 0} # Left of Up is Left, Left of Down is Right, etc.
        right_of = {0: 3, 1: 2, 2: 0, 3: 1} # Right of Up is Right, Right of Down is Left, etc.

        rand_val = random.random()
        
        # Determine the actual action based on probabilities
        if rand_val < 0.8:
            final_action = action # Desired direction
        elif rand_val < 0.9:
            final_action = left_of[action] # 10% chance to go left
        else:
            final_action = right_of[action] # 10% chance to go right

        # Calculate new position
        dr, dc = self.action_vectors[final_action]
        new_pos = (pos[0] + dr, pos[1] + dc)

        # Check for collisions
        if not self.is_valid_pos(new_pos):
            new_pos = pos # Stay in the same spot

        reward = self.get_reward(new_pos)
        return new_pos, reward

class QLearningAgent:
    """
    Q-learning agent that learns to navigate the Gridworld.
    """
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9999):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.start_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01

        # Q-table: (height * width) states and 4 actions
        self.q_table = np.zeros((env.height, env.width, 4))

    def choose_action(self, pos):
        """Choose an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Explore
        else:
            return np.argmax(self.q_table[pos[0], pos[1]])  # Exploit

    def update_q_table(self, pos, action, reward, next_pos):
        """Update the Q-value for a given state-action pair."""
        old_value = self.q_table[pos[0], pos[1], action]
        next_max = np.max(self.q_table[next_pos[0], next_pos[1]])
        
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[pos[0], pos[1], action] = new_value

    def decay_epsilon(self):
        """Decay the exploration rate."""
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

def train_agent(episodes, params):
    """Function to train the Q-learning agent and return rewards and Q-value history."""
    env = Gridworld(penalty=params.get('penalty', -1.0))
    agent = QLearningAgent(
        env,
        alpha=params.get('alpha', 0.1),
        gamma=params.get('gamma', 0.99),
        epsilon=1.0,
        epsilon_decay=params.get('epsilon_decay', 0.9999)
    )
    
    total_rewards = []
    q_value_history = [] # To track a specific Q-value
    
    # We will track the Q-value for moving 'Up' from the start state
    q_value_state = env.start_state
    q_value_action = 0 # 0 corresponds to 'Up'
    
    for episode in range(episodes):
        pos = env.start_state
        episode_reward = 0
        steps = 0
        
        while not env.is_terminal(pos) and steps < 200: # Max steps to prevent infinite loops
            action = agent.choose_action(pos)
            next_pos, reward = env.step(pos, action)
            agent.update_q_table(pos, action, reward, next_pos)
            
            pos = next_pos
            episode_reward += reward
            steps += 1
            
        agent.decay_epsilon()
        total_rewards.append(episode_reward)
        q_value_history.append(agent.q_table[q_value_state[0], q_value_state[1], q_value_action])

    return agent, total_rewards, q_value_history

def plot_results(results, title, xlabel, parameter_name):
    """Plot rewards over episodes for different parameter values."""
    plt.figure(figsize=(12, 7))
    plt.title(title, fontsize=16)
    
    for param_val, data in results.items():
        # Smooth the curve for better visualization
        smoothed_rewards = np.convolve(data['rewards'], np.ones(500)/500, mode='valid')
        plt.plot(smoothed_rewards, label=f"{parameter_name} = {param_val}")
        
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Total Reward per Episode (Smoothed)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_q_value_convergence(q_history, title, window_size=500):
    """Plots the convergence of a single Q-value over episodes with a moving average."""
    plt.figure(figsize=(12, 7))
    plt.title(title, fontsize=16)
    
    # Calculate and plot the moving average
    smoothed_q_values = np.convolve(q_history, np.ones(window_size)/window_size, mode='valid')
    plt.plot(smoothed_q_values)
    
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Q-Value (Smoothed)", fontsize=12)
    plt.grid(True)
    plt.show()

def display_policy(agent):
    """Visualize the learned policy."""
    env = agent.env
    # Use a standard list of lists for flexible string lengths and better alignment
    policy_grid = [[' ' for _ in range(env.width)] for _ in range(env.height)]
    
    action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    for r in range(env.height):
        for c in range(env.width):
            pos = (r, c)
            if pos == env.obstacle_state:
                policy_grid[r][c] = 'WALL'
            elif pos in env.terminal_states:
                reward = env.get_reward(pos)
                # Format to integer if it's a whole number, e.g., [1] or [-200]
                policy_grid[r][c] = f"[{int(reward)}]"
            else:
                # If all Q-values are zero (unexplored), show a dot.
                if np.all(agent.q_table[r, c] == 0):
                    policy_grid[r][c] = '.'
                else:
                    action = np.argmax(agent.q_table[r, c])
                    policy_grid[r][c] = action_arrows[action]
                
    print("\nLearned Policy:")
    print("-" * 29)
    # Print grid in natural top-to-bottom order for intuitive viewing
    for i in range(env.height):
        # Use rjust to right-align each cell for a clean grid look
        print(' '.join([cell.rjust(5) for cell in policy_grid[i]]))
    print("-" * 29)


if __name__ == '__main__':
    EPISODES = 20000

    # --- 1. Optimal Run ---
    print("--- Running with Optimal Parameters ---")
    optimal_params = {'alpha': 0.1, 'gamma': 0.99, 'epsilon_decay': 0.9999}
    optimal_agent, optimal_rewards, q_history = train_agent(EPISODES, optimal_params)
    display_policy(optimal_agent)
    
    # Plot Q-value convergence for the optimal run
    plot_q_value_convergence(q_history, "Q-Value Convergence for Start State -> 'Up' (Smoothed)")


    # --- Hyperparameter Tuning Experiments ---

    # a) Learning Rate (alpha)
    print("\n--- Testing different Learning Rates (alpha) ---")
    alpha_results = {}
    alphas = [0.01, 0.1, 0.5, 0.9]
    for alpha in alphas:
        print(f"Training with alpha = {alpha}")
        params = optimal_params.copy()
        params['alpha'] = alpha
        agent, rewards, _ = train_agent(EPISODES, params)
        alpha_results[alpha] = {'agent': agent, 'rewards': rewards}
    plot_results(alpha_results, "Effect of Learning Rate (α) on Performance", "Episodes", "α")

    # b) Discount Factor (gamma)
    print("\n--- Testing different Discount Factors (gamma) ---")
    gamma_results = {}
    gammas = [0.5, 0.9, 0.99, 1.0]
    for gamma in gammas:
        print(f"Training with gamma = {gamma}")
        params = optimal_params.copy()
        params['gamma'] = gamma
        agent, rewards, _ = train_agent(EPISODES, params)
        gamma_results[gamma] = {'agent': agent, 'rewards': rewards}
    plot_results(gamma_results, "Effect of Discount Factor (γ) on Performance", "Episodes", "γ")

    # c) Epsilon Decay
    print("\n--- Testing different Epsilon Decay Rates ---")
    epsilon_decay_results = {}
    decays = [0.999, 0.9999, 0.99999]
    for decay in decays:
        print(f"Training with epsilon_decay = {decay}")
        params = optimal_params.copy()
        params['epsilon_decay'] = decay
        agent, rewards, _ = train_agent(EPISODES, params)
        epsilon_decay_results[decay] = {'agent': agent, 'rewards': rewards}
    plot_results(epsilon_decay_results, "Effect of Epsilon Decay on Performance", "Episodes", "ε-decay")
    
    # --- 3. Test with Increased Penalty ---
    print("\n--- Running with Penalty = -200 ---")
    penalty_params = optimal_params.copy()
    penalty_params['penalty'] = -200.0
    penalty_agent, penalty_rewards, _ = train_agent(EPISODES, penalty_params)
    display_policy(penalty_agent)

