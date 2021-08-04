import argparse
import random
from os.path import abspath
from os.path import join

from numpy import argmax

from agent_score import agent_assessment
from disagreement import save_disagreements, get_top_k_disagreements, disagreement, \
    DisagreementTrace, State, make_same_length
from disagreements.get_trajectories import rank_trajectories
from disagreements.logging_info import log, get_logging
from get_agent import get_agent, get_agent_q_values_from_state, get_agent_action_from_state
from side_by_side import side_by_side_video
from utils import load_traces, save_traces
from copy import deepcopy


def online_comparison(args):
    """Compare two agents running online, search for disagreements"""
    """get agents and environments"""
    env1, a1 = get_agent()
    env2, a2 = get_agent()
    env1.args = env2.args = args

    """agent assessment"""
    agent_ratio = 1 if not args.agent_assessment else agent_assessment()  # pass agent configuration parameters

    """Run"""
    traces = []
    for e in range(args.num_episodes):
        log(f'Running Episode number: {e}', args.verbose)
        trace = DisagreementTrace(e, args.horizon, agent_ratio)
        curr_obs, _ = env1.reset(), env2.reset()
        assert curr_obs.tolist() == _.tolist(), f'Nonidentical environment'
        a1.previous_state = a2.previous_state = curr_obs
        t, r, done = 0, 0, False
        while not done:
            curr_s = curr_obs
            a1_s_a_values = get_agent_q_values_from_state(a1, curr_s)
            a2_s_a_values = get_agent_q_values_from_state(a2, curr_s)
            frame = env1.render(mode='rgb_array')
            state = State(t, e, curr_obs, curr_s, a1_s_a_values, frame)
            a1_a = get_agent_action_from_state(a1, curr_s)
            a2_a = get_agent_action_from_state(a2, curr_s)
            """initiate and update trace"""
            trace.update(state, curr_obs, a1_a, a1_s_a_values, a2_s_a_values, r, done, {})
            """check for disagreement"""
            if a1_a != a2_a:
                log(f'\tDisagreement at step {t}', args.verbose)
                disagreement(t, trace, env2, a2, curr_s, a1)
                """return agent 2 to the disagreement state"""
                env2, _ = get_agent()
                [env2.reset() for _ in range(e + 1)]
                [env2.step(a) for a in trace.actions[:-1]]
                env2.args = args
            """Transition both agent's based on agent 1 action"""
            curr_obs, r, done, info = env1.step(a1_a)
            _ = env2.step(a1_a)  # dont need returned values
            assert curr_obs.tolist() == _[0].tolist(), f'Nonidentical environment transition'
            t += 1

        """end of episode"""
        trace.get_trajectories()
        traces.append(deepcopy(trace))

    """close environments"""
    env1.close()
    env2.close()
    return traces


def main(args):
    name = get_logging(args)
    traces = load_traces(args.traces_path) if args.traces_path else online_comparison(args)
    log(f'Obtained traces', args.verbose)

    """save traces"""
    save_traces(traces, args.output)
    log(f'Saved traces', args.verbose)

    """rank disagreement trajectories by importance measures"""
    rank_trajectories(traces, args.importance_type, args.state_importance,
                      args.trajectory_importance)

    """top k diverse disagreements"""
    disagreements = get_top_k_disagreements(traces, args)
    log(f'Obtained {len(disagreements)} disagreements', args.verbose)

    """make all trajectories the same length"""
    disagreements = make_same_length(disagreements, args.horizon, traces)

    """randomize order"""
    if args.randomized: random.shuffle(disagreements)

    """get frames and mark disagreement frame"""
    a1_disagreement_frames, a2_disagreement_frames = [], []
    for d in disagreements:
        t = traces[d.episode]
        relative_idx = d.da_index - d.a1_states[0]
        actions = argmax(d.a1_s_a_values[relative_idx]), argmax(d.a2_s_a_values[relative_idx])
        a1_frames, a2_frames = t.get_frames(d.a1_states, d.a2_states, d.trajectory_index,
                                            actions=actions)
        a1_disagreement_frames.append(a1_frames)
        a2_disagreement_frames.append(a2_frames)

    """save disagreement frames"""
    video_dir = save_disagreements(a1_disagreement_frames, a2_disagreement_frames, args.output,
                                   args.fps)
    log(f'Disagreements saved', args.verbose)

    """generate video"""
    fade_duration = 2
    fade_out_frame = args.horizon - fade_duration + 11  # +11 from pause in save_disagreements
    side_by_side_video(video_dir, args.n_disagreements, fade_out_frame, name)
    log(f'DAs Video Generated', args.verbose)

    """ writes results to files"""
    log(f'\nResults written to:\n\t\'{args.output_dir}\'', args.verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent Comparisons')
    parser.add_argument('-env', '--env_id', help='environment name', default="highway_local-v0")
    parser.add_argument('-a1', '--a1_config', help='agent name', type=str, default="Agent-1")
    parser.add_argument('-a2', '--a2_config', help='agent name', type=str, default="Agent-2")
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
                        help='method calculating trajectory importance', default='last_state')
    parser.add_argument('-impState', '--state_importance',
                        help='method calculating state importance', default='bety')
    parser.add_argument('-v', '--verbose', help='print information to the console', default=True)
    parser.add_argument('-ass', '--agent_assessment', help='apply agent ratio by agent score',
                        default=False)
    parser.add_argument('-se', '--seed', help='environment seed', default=0)
    parser.add_argument('-res', '--results_dir', help='results directory', default='results')
    parser.add_argument('-tr', '--traces_path', help='path to traces file if exists',
                        default=None)
    args = parser.parse_args()

    args.a1_config = {
        "name": NotImplemented,
        "path": NotImplemented,
    }
    args.a2_config = {
        "name": NotImplemented,
        "path": NotImplemented,
    }

    """RUN"""
    main(args)
