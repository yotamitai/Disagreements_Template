import gym
from stable_baselines import DQN

def configure_env():
    env = gym.make('MsPacmanNoFrameskip-v4')
    configure_dict = {
        # add env configuration specifics here
    }
    env.configure(configure_dict)
    env.reset()
    return env

env = configure_env()
model = DQN('MlpPolicy', env)
model.learn(int(1e4))
model.save('agents/dqn_highway')
