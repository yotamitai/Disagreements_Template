import argparse
import gym
import numpy as np

def asses_agents(a1, a2):
    a1_overall = agent_score(a1)
    a2_overall = agent_score(a2)
    if a1_overall < 0 < a2_overall:
        a2_overall += abs(a1_overall)
        a1_overall = 1
    if a2_overall < 0 < a1_overall:
        a1_overall += abs(a2_overall)
        a2_overall = 1
    return a1_overall / a2_overall, a1_overall, a2_overall


def agent_score(config):
    """implement a simulation of the agent and retrieve the in-game score"""
    score  = 0
    NotImplemented #TODO
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='agent configuration')
    args = parser.parse_args()
    agent_score(args.config)