import gym


env = NotImplemented
agent = NotImplemented

# Evaluate the agent
obs = env.reset()
episode_reward = 0
for _ in range(100):
    action, _ = agent.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    episode_reward += reward
    if done or info.get('is_success', False):
        print("Reward:", episode_reward, "Success?", info.get('is_success', False))
        episode_reward = 0.0
        obs = env.reset()
