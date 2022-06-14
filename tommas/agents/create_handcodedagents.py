from tommas.agents.hand_coded_agents import PathfindingAgent
from tommas.agents.destination_selector import DestinationSelector, MultiAgentDestinationSelector, \
    DiscountGoalRanker, ClosestDistanceGoalRanker, DistancePenaltyGoalRanker, HighestGoalRanker, \
    CollaborativeStatePartitionFilter, StaticDiamondFilter, StaticCircleFilter, StaticSquareFilter, \
    PicnicDestinationSelector, SinglePicnicDestinationSelector, RepositionPicnicDestinationSelector

# from tommas.agents.destination_selector import DistanceBasedGoalPreference, HighestGoalPreference, \
#     CollaborativeDistanceBasedGoalPreference, DistanceBasedGoalPreferenceCommunicator, CollaborativeStatePartition, \
#     StatePartitionCommunicator
from tommas.data.agent_ids import get_agent_ids
from typing import Union

agent_types = ['', 'picnic']
goal_ranker_types = ['highest', 'closest_distance', 'discount_distance', 'distance_penalty']
state_filter_types = ['none', 'static_circle', 'static_diamond', 'static_square', "colab_state_partition"]


def _get_goal_ranker(goal_ranker_type, *args):
    if goal_ranker_type == "highest":
        return HighestGoalRanker(*args), goal_ranker_type
    elif goal_ranker_type == "closest_distance":
        return ClosestDistanceGoalRanker(), goal_ranker_type
    elif goal_ranker_type == "discount_distance":
        return DiscountGoalRanker(*args), goal_ranker_type + '=' + str(args[0])
    elif goal_ranker_type == "distance_penalty":
        return DistancePenaltyGoalRanker(*args), goal_ranker_type + '=' + str(args[0])
    else:
        return ValueError("goal_ranker_type: %s, does not exist. Please choose from the following %s" %
                          (goal_ranker_type, goal_ranker_types))


def _get_state_filters(num_agents, state_filter_type, *args):
    if state_filter_type is None or state_filter_type.lower() == "none":
        return [None for _ in range(num_agents)], ''
    elif state_filter_type == "static_circle":
        return [StaticCircleFilter(*args) for _ in range(num_agents)], state_filter_type + "=" + str(args[0])
    elif state_filter_type == "static_diamond":
        return [StaticDiamondFilter(*args) for _ in range(num_agents)], state_filter_type + "=" + str(args[0])
    elif state_filter_type == "static_square":
        return [StaticSquareFilter(*args) for _ in range(num_agents)], state_filter_type + "=" + str(args[0])
    elif state_filter_type == "colab_state_partition":
        state_filter = CollaborativeStatePartitionFilter(*args)
        return [state_filter for _ in range(num_agents)], state_filter_type
    else:
        raise ValueError("state_filter_type: %s, does not exist. Please choose from the following %s" %
                         (state_filter_type, state_filter_types))


def create_agent(reward_funcs, goal_ranker_type, goal_ranker_args: Union[tuple, None] = None,
                 state_filter_type=None, state_filter_args: Union[tuple, None] = None,
                 is_collaborative=False, agent_type='', generate_fake_agent_ids=False):
    def get_agent_name():
        name = ''
        if is_collaborative:
            name = 'colab-'
        if agent_type != '':
            name += agent_type + '-'
        name += goal_ranker_name
        if state_filter_name != '':
            name += '-' + state_filter_name
        return name

    if goal_ranker_args is None:
        goal_ranker_args = tuple()
    if state_filter_args is None:
        state_filter_args = tuple()
    goal_ranker, goal_ranker_name = _get_goal_ranker(goal_ranker_type, *goal_ranker_args)
    state_filters, state_filter_name = _get_state_filters(len(reward_funcs), state_filter_type, *state_filter_args)
    if agent_type == '':
        if is_collaborative:
            agents, agent_ids = _create_collaborative_agents(reward_funcs, goal_ranker, state_filters, generate_fake_agent_ids)
        else:
            agents, agent_ids = _create_independent_agents(reward_funcs, goal_ranker, state_filters, generate_fake_agent_ids)
    elif agent_type == "picnic":
        agents, agent_ids = _create_picnic_agents(reward_funcs, goal_ranker)
    elif agent_type == "single_picnic":
        agents, agent_ids = _create_single_picnic_agents(reward_funcs, goal_ranker)
    elif agent_type == "reposition_picnic":
        agents, agent_ids = _create_reposition_picnic_agents(reward_funcs, goal_ranker)
    else:
        raise ValueError("agent_type: %s, doesn't exist. Please choose from the following %s" %
                         (agent_type, agent_types))
    return agents, agent_ids, [get_agent_name() for _ in range(len(agents))]


def _create_independent_agents(reward_funcs, goal_ranker, state_filters, generate_fake_agent_ids):
    num_agents = len(reward_funcs)
    agents = []
    if generate_fake_agent_ids:
        agent_ids = list(range(num_agents))
    else:
        agent_ids = get_agent_ids(num_agents)
    for agent_idx in range(num_agents):
        destination_selector = DestinationSelector(reward_funcs[agent_idx], agent_idx, goal_ranker,
                                                   state_filters[agent_idx])
        agents.append(PathfindingAgent(agent_ids[agent_idx], reward_funcs[agent_idx], destination_selector))
    return agents, agent_ids


def _create_collaborative_agents(reward_funcs, goal_ranker, state_filters, generate_fake_agent_ids):
    num_agents = len(reward_funcs)
    agents = []
    if generate_fake_agent_ids:
        agent_ids = list(range(num_agents))
    else:
        agent_ids = get_agent_ids(num_agents)
    destination_selectors = [MultiAgentDestinationSelector(reward_funcs[0], 0, goal_ranker, True, state_filters[0])]
    for agent_idx in range(1, num_agents):
        destination_selector = MultiAgentDestinationSelector(reward_funcs[agent_idx], agent_idx, goal_ranker, False,
                                                             state_filters[agent_idx])
        destination_selectors[0].add_destination_selector(destination_selector)
        destination_selectors.append(destination_selector)

    for agent_idx in range(num_agents):
        agents.append(PathfindingAgent(agent_ids[agent_idx], reward_funcs[agent_idx], destination_selectors[agent_idx]))

    return agents, agent_ids


def _create_picnic_agents(reward_funcs, goal_ranker):
    num_agents = len(reward_funcs)
    agents = []
    agent_ids = get_agent_ids(num_agents)
    destination_selectors = [PicnicDestinationSelector(reward_funcs[0], 0, goal_ranker, True)]
    for agent_idx in range(1, num_agents):
        destination_selector = PicnicDestinationSelector(reward_funcs[agent_idx], agent_idx, goal_ranker, False)
        destination_selectors[0].add_destination_selector(destination_selector)
        destination_selectors.append(destination_selector)

    for agent_idx in range(num_agents):
        agents.append(PathfindingAgent(agent_ids[agent_idx], reward_funcs[agent_idx], destination_selectors[agent_idx]))

    return agents, agent_ids


def _create_single_picnic_agents(reward_funcs, goal_ranker):
    num_agents = len(reward_funcs)
    agents = []
    agent_ids = get_agent_ids(num_agents)
    destination_selectors = [SinglePicnicDestinationSelector(reward_funcs[0], 0, goal_ranker, True)]
    for agent_idx in range(1, num_agents):
        destination_selector = SinglePicnicDestinationSelector(reward_funcs[agent_idx], agent_idx, goal_ranker, False)
        destination_selectors[0].add_destination_selector(destination_selector)
        destination_selectors.append(destination_selector)

    for agent_idx in range(num_agents):
        agents.append(PathfindingAgent(agent_ids[agent_idx], reward_funcs[agent_idx], destination_selectors[agent_idx]))

    return agents, agent_ids


def _create_reposition_picnic_agents(reward_funcs, goal_ranker):
    num_agents = len(reward_funcs)
    agents = []
    agent_ids = get_agent_ids(num_agents)
    destination_selectors = [RepositionPicnicDestinationSelector(reward_funcs[0], 0, goal_ranker, True)]
    for agent_idx in range(1, num_agents):
        destination_selector = RepositionPicnicDestinationSelector(reward_funcs[agent_idx], agent_idx, goal_ranker, False)
        destination_selectors[0].add_destination_selector(destination_selector)
        destination_selectors.append(destination_selector)

    for agent_idx in range(num_agents):
        agents.append(PathfindingAgent(agent_ids[agent_idx], reward_funcs[agent_idx], destination_selectors[agent_idx]))

    return agents, agent_ids

