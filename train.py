from agent import Agent
import gymnasium as gym

env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")

agent = Agent(env=env, max_buffer_size=100000)

agent.train(episodes=3000, batch_size=32)
