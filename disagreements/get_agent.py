import gym

ACTION_DICT = NotImplemented #TODO
# example:{0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}

def get_agent(env=None, config=None):
    """Implement here for specific agent and environment loading scheme"""
    agent =None
    # TODO
    NotImplemented
    return env, agent

def reload_env():
    env = None
    # TODO
    NotImplemented
    return env

def get_agent_q_values_from_state(a, obs):
    NotImplemented #TODO
    q_values = None
    return q_values


def get_agent_action_from_state(a, curr_s):
    NotImplemented  # TODO
    action = None
    return action