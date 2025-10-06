import numpy as np
import matplotlib.pyplot as plt
import random
import time
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# --- Matplotlib Interactive Mode ---
# Enables the script to continue running while plots are open.
plt.ion()

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

def train_agent(episodes, params, animate=False, fig=None, ax=None):
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
    
    q_value_state = env.start_state
    q_value_action = 0 # 0 corresponds to 'Up'
    
    # Generate more intervals for a detailed animation
    animation_intervals = np.linspace(0, episodes - 1, num=12, dtype=int)

    for episode in range(episodes):
        pos = env.start_state
        episode_reward = 0
        steps = 0
        
        if animate and episode in animation_intervals:
            animate_navigation(agent, episode, episodes, fig, ax)

        while not env.is_terminal(pos) and steps < 200:
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

def plot_results(results, title, xlabel, parameter_name, optimal_value):
    """Plot rewards over episodes for different parameter values."""
    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=16, pad=20)
    
    for param_val, data in results.items():
        smoothed_rewards = np.convolve(data['rewards'], np.ones(500)/500, mode='valid')
        
        # Highlight the optimal parameter's curve
        is_optimal = (param_val == optimal_value)
        linewidth = 3 if is_optimal else 1.5
        alpha = 1.0 if is_optimal else 0.7
        
        plt.plot(smoothed_rewards, label=f"{parameter_name} = {param_val}", linewidth=linewidth, alpha=alpha)
        
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Total Reward per Episode (Smoothed)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_q_value_convergence(q_history, title, window_size=500):
    """Plots the convergence of a single Q-value over episodes with a moving average."""
    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=16, pad=20)
    
    # Plot raw Q-values with transparency
    plt.plot(q_history, color='lightblue', alpha=0.3, label='Raw Q-Value')
    
    # Plot smoothed Q-values
    smoothed_q_values = np.convolve(q_history, np.ones(window_size)/window_size, mode='valid')
    plt.plot(smoothed_q_values, color='darkblue', linewidth=2, label='Smoothed Q-Value (Moving Avg)')
    
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Q-Value", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def animate_navigation(agent, episode, total_episodes, fig, ax):
    """Animate the agent's learned policy and Q-values at a specific episode."""
    ax.cla()
    env = agent.env
    
    # --- Q-Value Heatmap ---
    max_q_values = np.max(agent.q_table, axis=2)
    # Handle case where all Q-values are zero to avoid vmin/vmax error
    vmin = np.min(max_q_values) if np.any(max_q_values != 0) else -1
    vmax = np.max(max_q_values) if np.any(max_q_values != 0) else 1
    norm = Normalize(vmin=vmin - 0.1, vmax=vmax + 0.1)
    cmap = plt.get_cmap('RdYlGn')
    
    ax.set_xticks(np.arange(-.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])

    action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    for r in range(env.height):
        for c in range(env.width):
            pos = (r, c)
            
            # Set background color based on max Q-value
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=cmap(norm(max_q_values[r, c]))))

            if pos == env.obstacle_state:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='black'))
                ax.text(c, r, 'WALL', ha='center', va='center', color='white', fontsize=12, weight='bold')
            elif pos in env.terminal_states:
                reward = env.get_reward(pos)
                facecolor = 'gold' if reward > 0 else 'maroon'
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=facecolor, alpha=0.7))
                ax.text(c, r, f"[{int(reward)}]", ha='center', va='center', fontsize=14, weight='bold')
            else:
                if not np.all(agent.q_table[r, c] == 0):
                    action = np.argmax(agent.q_table[r, c])
                    ax.text(c, r, action_arrows[action], ha='center', va='center', fontsize=24, color='black', weight='bold')
                
                # Display individual Q-values
                q_vals = agent.q_table[r, c]
                ax.text(c, r + 0.4, f"{q_vals[0]:.2f}", ha='center', va='top', fontsize=8, color='black')
                ax.text(c, r - 0.4, f"{q_vals[1]:.2f}", ha='center', va='bottom', fontsize=8, color='black')
                ax.text(c - 0.45, r, f"{q_vals[2]:.2f}", ha='left', va='center', fontsize=8, color='black')
                ax.text(c + 0.45, r, f"{q_vals[3]:.2f}", ha='right', va='center', fontsize=8, color='black')
            
            if pos == env.start_state:
                ax.text(c, r, 'START', ha='center', va='center', color='black', fontsize=9, weight='bold', bbox=dict(boxstyle="round,pad=0.1", fc='yellow', ec='black', lw=1))

    # Correct the visual orientation to match array indexing (0,0 at top-left)
    ax.invert_yaxis()
    ax.set_title(f"Episode {episode}/{total_episodes} | Press any key to continue...", fontsize=14, pad=15)
    
    # Pause execution and wait for a key press to continue
    plt.draw()
    fig.waitforbuttonpress()


def display_policy(agent):
    """Visualize the learned policy."""
    env = agent.env
    policy_grid = [[' ' for _ in range(env.width)] for _ in range(env.height)]
    
    action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    for r in range(env.height):
        for c in range(env.width):
            pos = (r, c)
            if pos == env.obstacle_state:
                policy_grid[r][c] = 'WALL'
            elif pos in env.terminal_states:
                reward = env.get_reward(pos)
                policy_grid[r][c] = f"[{int(reward)}]"
            else:
                if np.all(agent.q_table[r, c] == 0):
                    policy_grid[r][c] = '.'
                else:
                    action = np.argmax(agent.q_table[r, c])
                    policy_grid[r][c] = action_arrows[action]
                
    print("\nLearned Policy:")
    print("-" * 29)
    for i in range(env.height):
        print(' '.join([cell.rjust(5) for cell in policy_grid[i]]))
    print("-" * 29)

def plot_q_value_heatmap(agent, title):
    """Generates a heatmap of the maximum Q-values for each state."""
    env = agent.env
    max_q_values = np.max(agent.q_table, axis=2)
    fig, ax = plt.subplots(figsize=(10, 7))
    
    cax = ax.imshow(max_q_values, cmap='RdYlGn', interpolation='nearest')
    fig.colorbar(cax, label='Max Q-Value')
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xticks(np.arange(env.width))
    ax.set_yticks(np.arange(env.height))
    ax.set_xticklabels(np.arange(1, env.width + 1))
    ax.set_yticklabels(np.arange(1, env.height + 1))

    action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    for r in range(env.height):
        for c in range(env.width):
            pos = (r, c)
            text_color = 'black'
            if pos == env.obstacle_state:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='black'))
                ax.text(c, r, 'WALL', ha='center', va='center', color='white', weight='bold')
            elif pos in env.terminal_states:
                 reward = env.get_reward(pos)
                 ax.text(c, r, f"[{int(reward)}]", ha='center', va='center', color=text_color, weight='bold', fontsize=12)
            else:
                q_val = max_q_values[r, c]
                action = np.argmax(agent.q_table[r, c])
                arrow = action_arrows[action]
                ax.text(c, r, f"{arrow}\n{q_val:.2f}", ha='center', va='center', color=text_color, fontsize=10)

    plt.show()


if __name__ == '__main__':
    EPISODES = 20000
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- 1. Optimal Run with Animation ---
    print("--- Running with Optimal Parameters (with Animation) ---")
    fig, ax = plt.subplots(figsize=(10, 7))
    optimal_params = {'alpha': 0.1, 'gamma': 0.99, 'epsilon_decay': 0.9999}
    optimal_agent, optimal_rewards, q_history = train_agent(
        EPISODES, optimal_params, animate=True, fig=fig, ax=ax
    )
    print("\nAnimation complete. Final policy from animated run:")
    display_policy(optimal_agent)
    plt.close(fig)

    # Plot Q-value convergence for the optimal run
    plot_q_value_convergence(q_history, "Q-Value Convergence (Optimal) for Start State -> 'Up'")

    # --- Hyperparameter Tuning Experiments ---
    print("\n--- Testing different Learning Rates (alpha) ---")
    alpha_results = {}
    alphas = [0.01, 0.1, 0.5, 0.9]
    for alpha in alphas:
        print(f"Training with alpha = {alpha}")
        params = optimal_params.copy()
        params['alpha'] = alpha
        agent, rewards, _ = train_agent(EPISODES, params)
        alpha_results[alpha] = {'agent': agent, 'rewards': rewards}
    plot_results(alpha_results, "Effect of Learning Rate (α) on Performance", "Episodes", "α", optimal_params['alpha'])

    print("\n--- Testing different Discount Factors (gamma) ---")
    gamma_results = {}
    gammas = [0.5, 0.9, 0.99, 1.0]
    for gamma in gammas:
        print(f"Training with gamma = {gamma}")
        params = optimal_params.copy()
        params['gamma'] = gamma
        agent, rewards, _ = train_agent(EPISODES, params)
        gamma_results[gamma] = {'agent': agent, 'rewards': rewards}
    plot_results(gamma_results, "Effect of Discount Factor (γ) on Performance", "Episodes", "γ", optimal_params['gamma'])

    print("\n--- Testing different Epsilon Decay Rates ---")
    epsilon_decay_results = {}
    decays = [0.999, 0.9999, 0.99999]
    for decay in decays:
        print(f"Training with epsilon_decay = {decay}")
        params = optimal_params.copy()
        params['epsilon_decay'] = decay
        agent, rewards, _ = train_agent(EPISODES, params)
        epsilon_decay_results[decay] = {'agent': agent, 'rewards': rewards}
    plot_results(epsilon_decay_results, "Effect of Epsilon Decay on Performance", "Episodes", "ε-decay", optimal_params['epsilon_decay'])
    
    # --- 3. Test with Increased Penalty ---
    print("\n--- Running with Penalty = -200 ---")
    penalty_params = optimal_params.copy()
    penalty_params['penalty'] = -200.0
    penalty_agent, penalty_rewards, penalty_q_history = train_agent(EPISODES, penalty_params)
    display_policy(penalty_agent)
    
    # Plot Q-value convergence for the -200 penalty run
    plot_q_value_convergence(penalty_q_history, "Q-Value Convergence (Penalty = -200) for Start State -> 'Up'")

    # Plot Q-value heatmap for the -200 penalty run
    plot_q_value_heatmap(penalty_agent, "Q-Value Heatmap (Penalty = -200)")

    # --- Turn off interactive mode ---
    plt.ioff()
    print("\nClose all plot windows to exit.")
    plt.show() # Display all generated plots

