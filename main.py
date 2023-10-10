import gym
import numpy as np

# Create the environment
env = gym.make('Taxi-v3')

# Initialize Q-table with zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set hyperparameters
learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
epochs = 1000

for epoch in range(epochs):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy policy for exploration vs. exploitation
        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()  # Exploration
        else:
            action = np.argmax(Q[state, : )  # Exploitation

        # Take the action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)

        # Update Q-table using the Q-learning formula
        Q[state, action] = (1 - learning_rate) * Q[state, action] + \
                           learning_rate * (reward + discount_factor * np.max(Q[next_state, : ]))

        total_reward += reward
        state = next_state

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Total Reward: {total_reward}")

# Test the trained agent
state = env.reset()
done = False
total_reward = 0

while not done:
    action = np.argmax(Q[state, : ])
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    state = next_state

print(f"Test - Total Reward: {total_reward}")

# Close the environment
env.close()
