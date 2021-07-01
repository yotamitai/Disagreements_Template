import gym
import highway_env

from stable_baselines import DQN


def configure_env():
    env = gym.make('highway-v0')
    env.configure({
        "lanes_count": 4,
        "vehicles_count": 40,
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
        "policy_frequency": 2,
        "duration": 40,
    })
    env.reset()
    return env

env = configure_env()
model = DQN.load('../agents/dqn_highway', env)

# Evaluate the agent
obs = env.reset()
episode_reward = 0
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    episode_reward += reward
    if done or info.get('is_success', False):
        print("Reward:", episode_reward, "Success?", info.get('is_success', False))
        episode_reward = 0.0
        obs = env.reset()
