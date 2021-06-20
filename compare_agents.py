import argparse
import logging
import random

import gym
import numpy as np
from datetime import datetime
from os.path import join, basename
from agent_score import asses_agents
from disagreement import save_disagreements, get_top_k_disagreements, disagreement, \
    DisagreementTrace, State
from get_agent import get_agent
from merge_and_fade import merge_and_fade
from utils import load_agent_config, load_agent_aux, mark_agent


def get_logging(args):
    name = '_'.join([basename(args.a1_name), basename(args.a2_name)])
    file_name = '_'.join([name, datetime.now().strftime("%d-%m %H:%M:%S").replace(' ', '_')])
    log_name = join('logs', file_name)
    args.output = join('results', file_name)
    logging.basicConfig(filename=log_name + '.log', filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    return name


def online_comparison(args):
    """Compare two agents running online, search for disagreements"""

    """logging"""
    name = get_logging(args)
    logging.info(f'Comparing Agents: {name}')
    logging.info(f'Disagreement importance by: {args.importance_type}')
    if args.verbose:
        print(f'Comparing Agents: {name}')
        print(f'Disagreement importance by: {args.importance_type}')

    """agent assessment"""
    if args.agent_assesment:
        agent_ratio, a1_overall, a2_overall = asses_agents(args.a1_config, args.a2_config)
        logging.info(
            f'A1 score: {a1_overall}, A2 score: {a2_overall}, agent_ration: {agent_ratio}')
        if args.verbose:
            print(f'A1 score: {a1_overall}, A2 score: {a2_overall}, agent_ration: {agent_ratio}')
    else:
        agent_ratio = 1

    """get agents"""
    # use args.seed to create exactly the same env for both agents
    env1, a1, a1_args = get_agent(args)
    env2, a2, a2_args = get_agent(args)
    a1_Qmax, a2_Qmax = NotImplemented #TODO
    """implement normalizing agent Q values"""
    # divide all Q-values by Qmax for both a1 and a2 #TODO

    """skip first games"""
    # [env1.reset() for _ in range(5)]
    # [env2.reset() for _ in range(5)]

    """Run"""
    traces = []
    for e in range(args.num_episodes):
        logging.info(f'Running Episode number: {e}')
        if args.verbose: print(f'Running Episode number: {e}')
        t = 0
        trace = DisagreementTrace(args.horizon, a1_agent.q, a2_agent.q, agent_ratio, e)
        a1_done = False
        a1_curr_obs = env1.reset()
        _ = env2.reset()
        """first state"""
        a1_curr_s = a1_helper.get_state_from_observation(a1_curr_obs, 0, False)
        frame = env1.render()
        a1_position = [int(x) for x in env1.env.game_state.game.frog.position]
        trace.a1_states.append(
            State(t, e, a1_curr_obs, a1_curr_s, a1_agent.q[a1_curr_s], frame, a1_position))

        while not a1_done:
            a1_a = a1_agent.act(a1_curr_s)
            a2_a = a2_agent.act(a1_curr_s)
            """check for disagreement"""
            if a1_a != a2_a and t != 0:  # dont allow da on first state
                if args.verbose: print(f'\tDisagreement at frame {t}')
                logging.info(f'\tDisagreement at frame {t}')
                preceding_actions = a1_behavior_tracker.s_a[e]
                env2, a2_helper, a2_agent, _ = \
                    disagreement(e, trace, env2, a2_agent, a2_helper, t, a1_curr_s, a2_config,
                                 a2_agent_dir, agent_rng, preceding_actions, args)
                a2_agent.q = a2_agent.q / a2_Qmax
            """observe transition"""
            a1_new_obs, a1_r, a1_done, _ = env1.step(a1_a)
            _ = env2.step(a1_a)  # dont need returned values
            a1_new_s = a1_helper.get_state_from_observation(a1_new_obs, a1_r, a1_done)
            if a1_new_s == 1036:
                trace.lilies_reached += 1
                a1_done = True if trace.lilies_reached == 2 else False  # if lvl completed
            a1_r = a1_helper.get_reward(a1_curr_s, a1_a, a1_r, a1_new_s, a1_done)
            """update agent and stats"""
            a1_agent.update(a1_curr_s, a1_a, a1_r, a1_new_s)
            a2_agent.update(a1_curr_s, a1_a, a1_r, a1_new_s)
            a1_helper.update_stats(e, t, a1_curr_obs, a1_new_obs, a1_curr_s, a1_a, a1_r, a1_new_s)
            a1_behavior_tracker.add_sample(a1_curr_s, a1_a)
            """save state"""
            t += 1
            frame = env1.render()
            a1_curr_s = a1_new_s
            a1_curr_obs = a1_new_obs
            a1_position = [int(x) for x in env1.env.game_state.game.frog.position]
            trace.a1_states.append(
                State(t, e, a1_curr_obs, a1_curr_s, a1_agent.q[a1_curr_s], frame, a1_position))
            trace.a1_scores.append(env1.env.previous_score)

        """end of episode"""
        trace.get_trajectories(e, args.importance_type, args.state_importance,
                               args.trajectory_importance)
        traces.append(trace)
        a1_behavior_tracker.new_episode()

    """top k diverse disagreements"""
    disagreements = get_top_k_disagreements(traces, args)
    if args.verbose: print(f'Obtained {len(disagreements)} disagreements')
    logging.info(f'Obtained {len(disagreements)} disagreements')

    """randomize order"""
    if args.randomized: random.shuffle(disagreements)

    """mark disagreement frames"""
    a1_disagreement_frames, a2_disagreement_frames = [], []
    for d in disagreements:
        d_state = d.a1_states[(args.horizon // 2) - 1]
        d_state.image = mark_agent(d_state.image, d_state.agent_position)
        a1_frames, a2_frames = d.get_frames()
        for i in range(args.horizon // 2, args.horizon):
            a1_position = d.a1_states[i].agent_position
            a2_position = d.a2_states[i].agent_position
            a1_frames[i] = mark_agent(a1_frames[i], position=a1_position, color=(255, 0,))
            a2_frames[i] = mark_agent(a2_frames[i], position=a2_position, color=(0, 0, 0))
        a1_disagreement_frames.append(a1_frames)
        a2_disagreement_frames.append(a2_frames)

    """save disagreement frames"""
    video_dir = save_disagreements(a1_disagreement_frames, a2_disagreement_frames, a1_output_dir,
                                   args.fps)
    if args.verbose: print(f'Disagreements saved')
    logging.info(f'Disagreements saved')

    """generate video"""
    fade_duration = 2
    fade_out_frame = args.horizon - fade_duration
    merge_and_fade(video_dir, args.n_disagreements, fade_out_frame, name)
    if args.verbose: print(f'DAs Video Generated')
    logging.info(f'DAs Video Generated')

    """ writes results to files"""
    a1_agent.save(a1_output_dir)
    if args.verbose: print(f'\nResults written to:\n\t\'{a1_output_dir}\'')
    logging.info(f'\nResults written to:\n\t\'{a1_output_dir}\'')

    """close environments"""
    env1.close()
    env2.close()
    del gym.envs.registration.registry.env_specs[env1.spec.id]
    del gym.envs.registration.registry.env_specs[env2.spec.id]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent Comparisons')
    parser.add_argument('-a1', '--a1_name', help='agent name', type=str, default="Agent-1")
    parser.add_argument('-a2', '--a2_name', help='agent name', type=str, default="Agent-2")
    parser.add_argument('-n', '--num_episodes', help='number of episodes to run', type=int,
                        default=3)
    parser.add_argument('-fps', '--fps', help='summary video fps', type=int, default=1)
    parser.add_argument('-l', '--horizon', help='number of frames to show per highlight',
                        type=int, default=10)
    parser.add_argument('-sb', '--show_score_bar', help='score bar', type=bool, default=False)
    parser.add_argument('-rand', '--randomized', help='randomize order of summary trajectories',
                        type=bool, default=True)
    parser.add_argument('-k', '--n_disagreements', help='# of disagreements in the summary',
                        type=int, default=5)
    parser.add_argument('-overlaplim', '--similarity_limit', help='# overlaping',
                        type=int, default=3)
    parser.add_argument('-impMeth', '--importance_type',
                        help='importance by state or trajectory', default='trajectory')
    parser.add_argument('-impTraj', '--trajectory_importance',
                        help='method calculating trajectory importance', default='last_state_val')
    parser.add_argument('-impState', '--state_importance',
                        help='method calculating state importance', default='bety')
    parser.add_argument('-v', '--verbose', help='print information to the console', default=True)
    parser.add_argument('-ass', '--agent_assesment', help='apply agent ratio by agent score', default=False)
    parser.add_argument('-se', '--seed', help='environment seed', default=0)
    args = parser.parse_args()

    """experiment parameters"""
    args.a1_config = NotImplemented #TODO
    args.a2_config = NotImplemented #TODO

    # args.fps = 1
    # args.horizon = 10
    # args.show_score_bar = False
    # args.n_disagreements = 5
    # args.num_episodes = 10
    # args.randomized = True

    """get more/less trajectories"""
    # args.similarity_limit = 3  # int(args.horizon * 0.66)

    """importance measures"""
    # args.state_importance = "bety"  # "sb" "bety"
    # args.trajectory_importance = "last_state_val"  # last_state_loc,  last_state_val, max_min, max_avg, avg, avg_delta
    # args.importance_type = 'trajectory'  # state/trajectory

    """RUN"""
    args.a1_name = NotImplemented #TODO
    args.a2_name = NotImplemented #TODO
    online_comparison(args)
