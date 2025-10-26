import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import random
from collections import deque
import matplotlib.pyplot as plt
import time
import os

# Suppress TensorFlow warnings and errors about GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF info and warnings
tf.get_logger().setLevel('ERROR') # Suppress Keras-level errors
tf.config.set_visible_devices([], 'GPU') # Explicitly disable GPU

class GridworldEnv:
    """
    Implements the 3x4 Stochastic Gridworld environment as described.
    
    States are mapped to integer indices 0-10.
    Actions are 0: North, 1: South, 2: East, 3: West.
    """
    
    def __init__(self):
        self.grid_size = (4, 3) # (width, height) or (x, y)
        self.start_state = (1, 1)
        self.wall_state = (2, 2)
        self.terminal_states = {
            (4, 3): 1.0,  # Positive terminal state
            (4, 2): -1.0  # Negative terminal state
        }
        self.step_reward = -0.04
        
        # Map (x, y) coordinates to a single integer state index
        self.state_to_index = {}
        self.index_to_state = {}
        idx = 0
        for y in range(1, self.grid_size[1] + 1):
            for x in range(1, self.grid_size[0] + 1):
                state = (x, y)
                if state != self.wall_state:
                    self.state_to_index[state] = idx
                    self.index_to_state[idx] = state
                    idx += 1
                    
        self.num_states = len(self.state_to_index) # 11 valid states
        self.num_actions = 4 # N, S, E, W
        
        self.current_state_coord = self.start_state
        self.current_state_index = self.state_to_index[self.start_state]

    def reset(self):
        """Resets the environment to the start state."""
        self.current_state_coord = self.start_state
        self.current_state_index = self.state_to_index[self.start_state]
        return self.current_state_index

    def _get_stochastic_action(self, action):
        """Returns the actual action based on 80/10/10 probabilities."""
        # Actions: 0:N, 1:S, 2:E, 3:W
        prob = np.random.rand()
        if prob < 0.8:
            return action # Intended direction
        elif prob < 0.9:
            # 90 degrees left
            if action == 0: return 3 # N -> W
            if action == 1: return 2 # S -> E
            if action == 2: return 0 # E -> N
            if action == 3: return 1 # W -> S
        else:
            # 90 degrees right
            if action == 0: return 2 # N -> E
            if action == 1: return 3 # S -> W
            if action == 2: return 1 # E -> S
            if action == 3: return 0 # W -> N
        return action # Should not be reached

    def step(self, action):
        """
        Performs one step in the environment.
        
        Args:
            action (int): The *intended* action (0:N, 1:S, 2:E, 3:W).
            
        Returns:
            tuple: (next_state_index, reward, done)
        """
        
        # If already in a terminal state, stay put, 0 reward, done
        if self.current_state_coord in self.terminal_states:
            return self.current_state_index, 0.0, True

        # Get the actual stochastic action
        actual_action = self._get_stochastic_action(action)
        
        # Get current coordinates
        x, y = self.current_state_coord
        
        # Calculate next coordinates based on actual action
        nx, ny = x, y
        if actual_action == 0: ny += 1 # North
        elif actual_action == 1: ny -= 1 # South
        elif actual_action == 2: nx += 1 # East
        elif actual_action == 3: nx -= 1 # West
            
        # Check for collisions with outer walls or inner obstacle
        if (nx, ny) == self.wall_state or \
           nx < 1 or nx > self.grid_size[0] or \
           ny < 1 or ny > self.grid_size[1]:
            # Collision: stay in the same spot
            nx, ny = x, y
            
        # Get reward and done status
        next_state_coord = (nx, ny)
        
        if next_state_coord in self.terminal_states:
            reward = self.terminal_states[next_state_coord]
            done = True
        else:
            reward = self.step_reward
            done = False
            
        # Update current state
        self.current_state_coord = next_state_coord
        self.current_state_index = self.state_to_index[next_state_coord]
        
        return self.current_state_index, reward, done

class ReplayBuffer:
    """A simple FIFO experience replay buffer."""
    def __init__(self, capacity, num_states):
        self.buffer = deque(maxlen=capacity)
        self.num_states = num_states # Store num_states

    def push(self, state, action, reward, next_state, done):
        """Saves an experience."""
        # Convert state indices to one-hot vectors for storage
        state_one_hot = tf.one_hot(state, self.num_states).numpy()
        next_state_one_hot = tf.one_hot(next_state, self.num_states).numpy()
        self.buffer.append((state_one_hot, action, reward, next_state_one_hot, done))

    def sample(self, batch_size):
        """Randomly samples a batch of experiences."""
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), \
               np.array(next_state), np.array(done, dtype=np.uint8)

    def __len__(self):
        return len(self.buffer)

class DQN(models.Model):
    """Deep Q-Network model."""
    def __init__(self, num_states, num_actions, hidden_units=64):
        super(DQN, self).__init__()
        # Define layers in the constructor
        self.hidden1 = layers.Dense(hidden_units, activation='relu', input_shape=(num_states,))
        self.hidden2 = layers.Dense(hidden_units, activation='relu')
        self.output_layer = layers.Dense(num_actions, activation='linear') # Linear activation for Q-values

    def call(self, inputs):
        """Defines the forward pass."""
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        return self.output_layer(x)

class DQNAgent:
    """DQN Agent that interacts with the Gridworld environment."""
    
    def __init__(self, env, config):
        self.env = env
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.95)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.memory_size = config.get('memory_size', 10000)
        self.target_update_freq = config.get('target_update_freq', 10) # in episodes
        self.hidden_units = config.get('hidden_units', 64)
        
        # ReplayBuffer now correctly receives num_states
        self.memory = ReplayBuffer(self.memory_size, self.num_states)
        
        # Q-Network (Main Model)
        self.model = DQN(self.num_states, self.num_actions, self.hidden_units)
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        # Use the 'losses' module imported from Keras
        self.loss_fn = losses.MeanSquaredError()

        # Target Network
        self.target_model = DQN(self.num_states, self.num_actions, self.hidden_units)
        # Initialize target model weights to be same as main model
        self.update_target_model()

    def update_target_model(self):
        """Copies weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state_index):
        """
        Chooses an action using an epsilon-greedy policy.
        State is provided as an integer index.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        
        # Get Q-values from the main model
        state_one_hot = tf.one_hot([state_index], self.num_states)
        q_values = self.model(state_one_hot)
        return tf.argmax(q_values[0]).numpy()

    def replay(self):
        """Trains the main Q-network using a minibatch from the replay buffer."""
        if len(self.memory) < self.batch_size:
            return 0 # Not enough samples to train
            
        # Sample a minibatch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # states and next_states are already one-hot
        
        # Use the target model to predict Q(s', a')
        # Q-values for all actions in the next states
        q_next = self.target_model(next_states)
        
        # Find max Q-value for each next state
        max_q_next = tf.reduce_max(q_next, axis=1)
        
        # Calculate target Q-values
        # Q_target = r + γ * max_a' Q_target(s', a')
        # If done, Q_target = r
        target_q = rewards + (1.0 - dones) * self.gamma * max_q_next
        
        # We need to train the model only on the Q-value for the action taken.
        with tf.GradientTape() as tape:
            # Get current Q-values from the main model
            q_values = self.model(states)
            
            # Create a mask for the actions taken
            action_mask = tf.one_hot(actions, self.num_actions)
            
            # Get the Q-value for the action taken
            q_value_for_action_taken = tf.reduce_sum(q_values * action_mask, axis=1)
            
            # Calculate loss
            loss = self.loss_fn(target_q, q_value_for_action_taken)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss.numpy()

    def train(self, num_episodes=1000, eval_freq=50, num_eval_episodes=10):
        """Main training loop."""
        print(f"Starting training with config: {self.get_config_str()}")
        
        training_rewards = []
        evaluation_rewards = []
        # Renamed list to avoid conflict with 'losses' module
        loss_history = []
        
        for e in range(num_episodes):
            state_index = self.env.reset()
            done = False
            episode_reward = 0
            episode_loss = 0
            steps = 0
            
            while not done:
                action = self.get_action(state_index)
                next_state_index, reward, done = self.env.step(action)
                
                # Store in memory (using indices)
                self.memory.push(state_index, action, reward, next_state_index, done)
                
                loss = self.replay()
                
                episode_reward += reward
                episode_loss += loss
                steps += 1
                state_index = next_state_index
                
            # End of episode
            training_rewards.append(episode_reward)
            loss_history.append(episode_loss / steps if steps > 0 else 0)

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Update target network
            if (e + 1) % self.target_update_freq == 0:
                self.update_target_model()

            # Print progress more frequently for debugging
            if (e + 1) % 20 == 0:
                print(f"Episode {e+1}/{num_episodes} | Reward: {episode_reward:.2f} | Avg Loss: {loss_history[-1]:.4f} | Epsilon: {self.epsilon:.3f}")

            # Run evaluation
            if (e + 1) % eval_freq == 0:
                avg_eval_reward = self.evaluate(num_eval_episodes)
                evaluation_rewards.append(avg_eval_reward)
                print(f"--- Evaluation at Episode {e+1} | Avg Reward: {avg_eval_reward:.2f} ---")
                
        print("Training finished.")
        return training_rewards, evaluation_rewards, loss_history, list(range(eval_freq, num_episodes + 1, eval_freq))

    def evaluate(self, num_eval_episodes=10):
        """Evaluates the agent's greedy policy."""
        total_reward = 0
        for _ in range(num_eval_episodes):
            state_index = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Choose action greedily (no exploration)
                state_one_hot = tf.one_hot([state_index], self.num_states)
                q_values = self.model(state_one_hot)
                action = tf.argmax(q_values[0]).numpy()
                
                next_state_index, reward, done = self.env.step(action)
                
                episode_reward += reward
                state_index = next_state_index
                
            total_reward += episode_reward
            
        return total_reward / num_eval_episodes

    def print_policy(self):
        """Prints the learned greedy policy for each state."""
        print("\n--- Learned Policy (Greedy) ---")
        action_map = {0: "N", 1: "S", 2: "E", 3: "W"}
        
        # Iterate through states in grid order
        # Correctly reference self.env attributes
        for y in range(self.env.grid_size[1], 0, -1): # 3, 2, 1
            row_str = ""
            for x in range(1, self.env.grid_size[0] + 1): # 1, 2, 3, 4
                state_coord = (x, y)
                # Corrected typo from self.env.wall_.state
                if state_coord == self.env.wall_state:
                    row_str += " [WALL] "
                elif state_coord in self.env.terminal_states:
                    row_str += f" [{self.env.terminal_states[state_coord]:+}] "
                else:
                    state_index = self.env.state_to_index[state_coord]
                    state_one_hot = tf.one_hot([state_index], self.num_states)
                    q_values = self.model(state_one_hot)
                    action = tf.argmax(q_values[0]).numpy()
                    row_str += f" {action_map[action]:^6} "
            print(row_str)
        print("---------------------------------\n")
        
    def get_config_str(self):
        return (f"LR: {self.learning_rate}, Gamma: {self.gamma}, EpsilonDecay: {self.epsilon_decay}, "
                f"Batch: {self.batch_size}, TargetUpdate: {self.target_update_freq}")

def plot_curves(training_rewards, eval_rewards, loss_history, eval_episodes, title="Learning Curves"):
    """Plots the learning curves."""
    
    # Calculate moving average for training rewards
    window_size = 50
    if len(training_rewards) >= window_size:
        moving_avg_rewards = np.convolve(training_rewards, np.ones(window_size)/window_size, mode='valid')
    else:
        moving_avg_rewards = training_rewards # Not enough data for moving avg
        
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
    fig.suptitle(title, fontsize=16)

    # 1. Training Rewards
    ax1.plot(training_rewards, label='Raw Reward per Episode', alpha=0.3)
    if len(training_rewards) >= window_size:
        ax1.plot(range(window_size-1, len(training_rewards)), moving_avg_rewards, 
                 label=f'{window_size}-Episode Moving Avg', color='blue')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards (with exploration)')
    ax1.legend()
    ax1.grid(True)

    # 2. Evaluation Rewards
    ax2.plot(eval_episodes, eval_rewards, marker='o', linestyle='-', 
             label='Avg Reward (Greedy Policy)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Evaluation Rewards (no exploration)')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Average Loss
    # Use the renamed 'loss_history' variable
    ax3.plot(loss_history, label='Average Loss per Episode', color='orange')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Mean Squared Error Loss')
    ax3.set_title('Training Loss')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot to a file
    plot_filename = f"learning_curves_{title.replace(' ', '_').replace(':', '').replace(',', '')}.png"
    plt.savefig(plot_filename)
    print(f"Learning curves saved to {plot_filename}")
    
    # Remove the blocking plt.show() call to prevent hangs
    # plt.show() 
    
    # Close the figure to free up memory
    plt.close(fig)

# --- Main Execution ---

if __name__ == "__main__":
    
    # Baseline Hyperparameters
    baseline_config = {
        'gamma': 0.95,
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'learning_rate': 0.001,
        'batch_size': 32,
        'memory_size': 10000,
        'target_update_freq': 10,
        'hidden_units': 64
    }
    num_episodes_baseline = 1500
    
    # --- 1. Run with baseline hyperparameters ---
    print("=== RUN 1: BASELINE ===")
    # Create a new, isolated env for this run
    env_base = GridworldEnv()
    
    agent = DQNAgent(env_base, baseline_config)
    
    start_time = time.time()
    training_rewards, eval_rewards, loss_history, eval_episodes = agent.train(
        num_episodes=num_episodes_baseline, 
        eval_freq=50, 
        num_eval_episodes=10
    )
    end_time = time.time()
    
    print(f"Baseline training took {end_time - start_time:.2f} seconds.")
    
    # Plot curves for baseline
    plot_curves(training_rewards, eval_rewards, loss_history, eval_episodes, 
                title=f"Baseline Run: {agent.get_config_str()}")
    
    # Print the final policy
    agent.print_policy()
    
    
    # --- 2. Run with different hyperparameters to show effect ---
    
    print("\n\n=== RUN 2: HIGHER LEARNING RATE ===")
    # Example: Faster learning rate
    lr_config = baseline_config.copy()
    lr_config['learning_rate'] = 0.01 # 10x higher
    
    # Create a new, isolated env for this run
    env_lr = GridworldEnv()
    agent_lr = DQNAgent(env_lr, lr_config)
    tr_lr, er_lr, loss_lr, ee_lr = agent_lr.train(num_episodes=num_episodes_baseline, eval_freq=50)
    plot_curves(tr_lr, er_lr, loss_lr, ee_lr, 
                title=f"High LR Run: {agent_lr.get_config_str()}")
    agent_lr.print_policy()


    print("\n\n=== RUN 3: FASTER EPSILON DECAY ===")
    # Example: Faster epsilon decay (less exploration)
    eps_config = baseline_config.copy()
    eps_config['epsilon_decay'] = 0.99 # Faster decay
    
    # Create a new, isolated env for this run
    env_eps = GridworldEnv()
    agent_eps = DQNAgent(env_eps, eps_config)
    tr_eps, er_eps, loss_eps, ee_eps = agent_eps.train(num_episodes=num_episodes_baseline, eval_freq=50)
    plot_curves(tr_eps, er_eps, loss_eps, ee_eps, 
                title=f"Fast Epsilon Decay Run: {agent_eps.get_config_str()}")
    agent_eps.print_policy()

    print("\n\n=== RUN 4: MODERATE TARGET UPDATE ===")
    # Example: More moderate target network update (originally was 50, now 25)
    target_config = baseline_config.copy()
    target_config['target_update_freq'] = 25 # More reasonable update frequency
    
    # Create a new, isolated env for this run
    env_target = GridworldEnv()
    agent_target = DQNAgent(env_target, target_config)
    
    print("Starting RUN 4 with improved configuration...")
    start_time_run4 = time.time()
    
    tr_target, er_target, loss_target, ee_target = agent_target.train(num_episodes=num_episodes_baseline, eval_freq=50)
    
    end_time_run4 = time.time()
    print(f"RUN 4 completed in {end_time_run4 - start_time_run4:.2f} seconds.")
    
    plot_curves(tr_target, er_target, loss_target, ee_target, 
                title=f"Moderate Target Update Run: {agent_target.get_config_str()}")
    agent_target.print_policy()

