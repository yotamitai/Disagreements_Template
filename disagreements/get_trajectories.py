def trajectory_importance_max_min(states_importance):
    """ computes the importance of the trajectory, according to max-min approach:
     delta(max state, min state) """
    return max(states_importance) - min(states_importance)


def trajectory_importance_max_avg(states_importance):
    """ computes the importance of the trajectory, according to max-avg approach:
     delta(max state, avg) """
    avg = sum(states_importance) / len(states_importance)
    return max(states_importance) - avg


def trajectory_importance_avg(states_importance):
    """ computes the importance of the trajectory, according to avg approach """
    avg = sum(states_importance) / len(states_importance)
    return avg


def trajectory_importance_avg_delta(states_importance):
    """ computes the importance of the trajectory, according to the average delta approach """
    sum_delta = 0
    for i in range(len(states_importance)):
        sum_delta += states_importance[i] - states_importance[i - 1]
    avg_delta = sum_delta / len(states_importance)
    return avg_delta
