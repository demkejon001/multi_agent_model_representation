from typing import Union
from argparse import Namespace, ArgumentParser

from tommas.agent_modellers.embedding_networks import base_agent_embedding_specs
from tommas.agent_modellers.tommas import TOMMAS, ToMnet, RandomPolicyToMnet
from tommas.agent_modellers.gridworld_tomnet import GridworldToMNet
from tommas.agent_modellers.tommas_transformer import TOMMASTransformer
from tommas.agent_modellers.iterative_action_tommas import IterativeActionTOMMAS, IterativeActionToMnet, \
    IterativeActionLSTM
from tommas.agent_modellers.iterative_action_tommas_transformer import IterativeActionTOMMASTransformer
from tommas.agent_modellers.iterative_action_tomnet import IterativeActionPastCurrentNet
from tommas.data.datamodule_factory import get_datasets_abbreviation_str


MODELS = {"tommas": TOMMAS,
          "ia_tomnet": ToMnet,
          "ja_tomnet": ToMnet,
          "random_policy_tomnet": RandomPolicyToMnet,
          "tommas_transformer": TOMMASTransformer,
          "iterative_action_tommas": IterativeActionTOMMAS,
          "iterative_action_ia_tomnet": IterativeActionToMnet,
          "iterative_action_ja_tomnet": IterativeActionToMnet,
          "iterative_action_tommas_transformer": IterativeActionTOMMASTransformer,
          "iterative_action_lstm": IterativeActionLSTM,
          "iterative_action_past_current": IterativeActionPastCurrentNet,
          "gridworld_tomnet": GridworldToMNet
          }


def make_model(args: Union[Namespace, dict], observation_space, action_space):
    if isinstance(args, dict):
        args = Namespace(**args)
    check_model_type(args)
    model_factory_fn = {"tommas": make_tommas,
                        "ia_tomnet": make_ia_tomnet,
                        "ja_tomnet": make_ja_tomnet,
                        "random_policy_tomnet": make_random_policy_tomnet,
                        "tommas_transformer": make_tommas_transformer,
                        "iterative_action_tommas": make_iterative_action_tommas,
                        "iterative_action_ia_tomnet": make_iterative_action_ia_tomnet,
                        "iterative_action_ja_tomnet": make_iterative_action_ja_tomnet,
                        "iterative_action_tommas_transformer": make_iterative_action_tommas_transformer,
                        "iterative_action_lstm": make_iterative_action_lstm,
                        "iterative_action_past_current": make_iterative_action_past_current,
                        "gridworld_tomnet": make_gridworld_tomnet
                        }
    model_type = args.model
    return model_factory_fn[model_type](args, observation_space, action_space)


def check_model_type(args: Namespace):
    if args.model not in MODELS:
        raise ValueError("model %s is not a known model. Choose from the following models: %s" %
                         (args.model, list(MODELS.keys())))


def get_model_name(args: Namespace):
    model_name_abbreviations = {"tommas": "TMas",
                                "ia_tomnet": "IATNet",
                                "ja_tomnet": "JATNet",
                                "random_policy_tomnet": "RandPolicyTNet",
                                "tommas_transformer": "TTX",
                                "iterative_action_tommas": "IterTMas",
                                "iterative_action_ia_tomnet": "IAIterTNet",
                                "iterative_action_ja_tomnet": "JAIterTNet",
                                "iterative_action_tommas_transformer": "IterTTX",
                                "iterative_action_lstm": "IterLSTM",
                                "iterative_action_past_current": "IterPastCur",
                                "gridworld_tomnet": "GridTNet",
                                }
    return model_name_abbreviations[args.model] + get_datasets_abbreviation_str(args)


def add_model_specific_args(parser: ArgumentParser, args: Namespace) -> ArgumentParser:
    check_model_type(args)
    return MODELS[args.model].add_model_specific_args(parser)


def extract_agent_embedding_specs_from_args(network_type, args: Namespace):
    network_type += "_"
    args_dict = vars(args)
    embedding_specs = base_agent_embedding_specs.copy()
    for spec in base_agent_embedding_specs:
        embedding_specs[spec] = args_dict.get(network_type + spec, base_agent_embedding_specs[spec])
        if not embedding_specs["pre_resnet_layer"]:
            embedding_specs["pre_resnet_layer"] = None
    return embedding_specs


def make_tommas(args: Namespace, observation_space, action_space):
    ia_char_embedding_specs = extract_agent_embedding_specs_from_args("ia_char", args)
    ia_mental_embedding_specs = extract_agent_embedding_specs_from_args("ia_mental", args)
    ja_char_embedding_specs = extract_agent_embedding_specs_from_args("ja_char", args)
    ja_mental_embedding_specs = extract_agent_embedding_specs_from_args("ja_mental", args)

    num_joint_agent_features = observation_space['current_trajectory'][2]
    world_dim = observation_space['current_trajectory'][3:]
    num_independent_features = observation_space["independent_agent_features"][0]

    state_in_channels = observation_space['query_state'][1]
    num_actions = action_space['action'][1]
    num_goals = action_space['goal_consumption'][1]
    pred_net_in_channels = state_in_channels + args.ia_char_embedding_size + args.ja_char_embedding_size \
                           + args.ia_mental_embedding_size + args.ja_mental_embedding_size
    return TOMMAS(model_type=args.model, model_name=get_model_name(args), learning_rate=args.learning_rate,
                  num_independent_agent_features=num_independent_features,
                  num_joint_agent_features=num_joint_agent_features,
                  ia_char_embedding_specs=ia_char_embedding_specs, ia_mental_embedding_specs=ia_mental_embedding_specs,
                  ja_char_embedding_specs=ja_char_embedding_specs, ja_mental_embedding_specs=ja_mental_embedding_specs,
                  pred_net_in_channels=pred_net_in_channels, pred_num_resnet_blocks=args.pred_num_resnet_blocks,
                  pred_resnet_channels=args.pred_resnet_channels,
                  action_pred_hidden_channels=args.action_pred_hidden_channels,
                  goal_pred_hidden_channels=args.goal_pred_hidden_channels,
                  sr_pred_hidden_channels=args.sr_pred_hidden_channels,
                  world_dim=world_dim, num_actions=num_actions, num_goals=num_goals,
                  no_action_loss=args.no_action_loss, no_goal_loss=args.no_goal_loss, no_sr_loss=args.no_sr_loss)


def make_ia_tomnet(args: Namespace, observation_space, action_space):
    ia_char_embedding_specs = extract_agent_embedding_specs_from_args("ia_char", args)
    ia_mental_embedding_specs = extract_agent_embedding_specs_from_args("ia_mental", args)

    num_joint_agent_features = observation_space['current_trajectory'][2]
    world_dim = observation_space['current_trajectory'][3:]
    num_independent_features = observation_space["independent_agent_features"][0]

    state_in_channels = observation_space['query_state'][1]
    num_actions = action_space['action'][1]
    num_goals = action_space['goal_consumption'][1]
    pred_net_in_channels = state_in_channels + args.ia_char_embedding_size + args.ia_mental_embedding_size

    return ToMnet(model_type=args.model, model_name=get_model_name(args), learning_rate=args.learning_rate,
                  num_independent_agent_features=num_independent_features,
                  num_joint_agent_features=num_joint_agent_features,
                  ia_char_embedding_specs=ia_char_embedding_specs, ia_mental_embedding_specs=ia_mental_embedding_specs,
                  ja_char_embedding_specs=None, ja_mental_embedding_specs=None,
                  pred_net_in_channels=pred_net_in_channels, pred_num_resnet_blocks=args.pred_num_resnet_blocks,
                  pred_resnet_channels=args.pred_resnet_channels,
                  action_pred_hidden_channels=args.action_pred_hidden_channels,
                  goal_pred_hidden_channels=args.goal_pred_hidden_channels,
                  sr_pred_hidden_channels=args.sr_pred_hidden_channels,
                  world_dim=world_dim, num_actions=num_actions, num_goals=num_goals,
                  no_action_loss=args.no_action_loss, no_goal_loss=args.no_goal_loss, no_sr_loss=args.no_sr_loss)


def make_ja_tomnet(args: Namespace, observation_space, action_space):
    ja_char_embedding_specs = extract_agent_embedding_specs_from_args("ja_char", args)
    ja_mental_embedding_specs = extract_agent_embedding_specs_from_args("ja_mental", args)

    num_joint_agent_features = observation_space['current_trajectory'][2]
    world_dim = observation_space['current_trajectory'][3:]
    num_independent_features = observation_space["independent_agent_features"][0]

    state_in_channels = observation_space['query_state'][1]
    num_actions = action_space['action'][1]
    num_goals = action_space['goal_consumption'][1]
    pred_net_in_channels = state_in_channels + args.ja_char_embedding_size + args.ja_mental_embedding_size

    return ToMnet(model_type=args.model, model_name=get_model_name(args), learning_rate=args.learning_rate,
                  num_independent_agent_features=num_independent_features,
                  num_joint_agent_features=num_joint_agent_features,
                  ia_char_embedding_specs=None, ia_mental_embedding_specs=None,
                  ja_char_embedding_specs=ja_char_embedding_specs, ja_mental_embedding_specs=ja_mental_embedding_specs,
                  pred_net_in_channels=pred_net_in_channels, pred_num_resnet_blocks=args.pred_num_resnet_blocks,
                  pred_resnet_channels=args.pred_resnet_channels,
                  action_pred_hidden_channels=args.action_pred_hidden_channels,
                  goal_pred_hidden_channels=args.goal_pred_hidden_channels,
                  sr_pred_hidden_channels=args.sr_pred_hidden_channels,
                  world_dim=world_dim, num_actions=num_actions, num_goals=num_goals,
                  no_action_loss=args.no_action_loss, no_goal_loss=args.no_goal_loss, no_sr_loss=args.no_sr_loss)


def make_random_policy_tomnet(args: Namespace, observation_space, action_space):
    return RandomPolicyToMnet(model_type=args.model, model_name=get_model_name(args),
                              learning_rate=args.learning_rate, no_action_loss=args.no_action_loss,
                              no_goal_loss=args.no_goal_loss, no_sr_loss=args.no_sr_loss)


def make_tommas_transformer(args: Namespace, observation_space, action_space):
    num_joint_agent_features = observation_space['trajectory'][2]
    world_dim = observation_space['trajectory'][3:]

    state_in_channels = observation_space['query_state'][1]
    num_actions = action_space['action'][1]
    num_goals = action_space['goal_consumption'][1]
    pred_net_in_channels = state_in_channels + args.embedding_size

    gridworld_embedding_specs = dict()
    gridworld_embedding_specs["num_resnet_blocks"] = args.num_resnet_blocks
    gridworld_embedding_specs["resnet_channels"] = args.resnet_channels
    gridworld_embedding_specs["pooling"] = args.pooling
    gridworld_embedding_specs["pre_resnet_layer"] = args.pre_resnet_layer
    if not gridworld_embedding_specs["pre_resnet_layer"]:
        gridworld_embedding_specs["pre_resnet_layer"] = None

    return TOMMASTransformer(model_type=args.model, model_name=get_model_name(args),
                             learning_rate=args.learning_rate,
                             num_joint_agent_features=num_joint_agent_features,
                             gridworld_embedding_specs=gridworld_embedding_specs,
                             embedding_size=args.embedding_size, n_layer=args.n_layer, n_head=args.n_head,
                             pred_net_in_channels=pred_net_in_channels,
                             pred_num_resnet_blocks=args.pred_num_resnet_blocks,
                             pred_resnet_channels=args.pred_resnet_channels,
                             action_pred_hidden_channels=args.action_pred_hidden_channels,
                             goal_pred_hidden_channels=args.goal_pred_hidden_channels,
                             sr_pred_hidden_channels=args.sr_pred_hidden_channels,
                             world_dim=world_dim, num_actions=num_actions, num_goals=num_goals,
                             no_action_loss=args.no_action_loss, no_goal_loss=args.no_goal_loss,
                             no_sr_loss=args.no_sr_loss)


def make_iterative_action_tommas(args: Namespace, observation_space, action_space):
    num_joint_agent_features = observation_space['current_trajectory'][2]
    num_independent_features = observation_space["independent_agent_features"][0]

    num_actions = action_space['action'][1]

    return IterativeActionTOMMAS(model_type=args.model, model_name=get_model_name(args),
                                 learning_rate=args.learning_rate,
                                 num_independent_agent_features=num_independent_features,
                                 num_joint_agent_features=num_joint_agent_features,
                                 ia_char_hidden_layer_features=args.ia_char_hidden_layer_features,
                                 ia_char_embedding_size=args.ia_char_embedding_size,
                                 ia_mental_hidden_layer_features=args.ia_mental_hidden_layer_features,
                                 ia_mental_embedding_size=args.ia_mental_embedding_size,
                                 ja_char_hidden_layer_features=args.ja_char_hidden_layer_features,
                                 ja_char_embedding_size=args.ja_char_embedding_size,
                                 ja_mental_hidden_layer_features=args.ja_mental_hidden_layer_features,
                                 ja_mental_embedding_size=args.ja_mental_embedding_size,
                                 num_actions=num_actions)


def make_iterative_action_ia_tomnet(args: Namespace, observation_space, action_space):
    num_joint_agent_features = observation_space['current_trajectory'][2]
    num_independent_features = observation_space["independent_agent_features"][0]

    num_actions = action_space['action'][1]

    return IterativeActionToMnet(model_type=args.model, model_name=get_model_name(args),
                                 learning_rate=args.learning_rate,
                                 num_independent_agent_features=num_independent_features,
                                 num_joint_agent_features=num_joint_agent_features,
                                 ia_char_hidden_layer_features=args.ia_char_hidden_layer_features,
                                 ia_char_embedding_size=args.ia_char_embedding_size,
                                 ia_mental_hidden_layer_features=args.ia_mental_hidden_layer_features,
                                 ia_mental_embedding_size=args.ia_mental_embedding_size,
                                 ja_char_hidden_layer_features=None,
                                 ja_char_embedding_size=-1,
                                 ja_mental_hidden_layer_features=None,
                                 ja_mental_embedding_size=-1,
                                 num_actions=num_actions)


def make_iterative_action_ja_tomnet(args: Namespace, observation_space, action_space):
    num_joint_agent_features = observation_space['current_trajectory'][2]
    num_independent_features = observation_space["independent_agent_features"][0]

    num_actions = action_space['action'][1]

    return IterativeActionToMnet(model_type=args.model, model_name=get_model_name(args),
                                 learning_rate=args.learning_rate,
                                 num_independent_agent_features=num_independent_features,
                                 num_joint_agent_features=num_joint_agent_features,
                                 ia_char_hidden_layer_features=None,
                                 ia_char_embedding_size=-1,
                                 ia_mental_hidden_layer_features=None,
                                 ia_mental_embedding_size=-1,
                                 ja_char_hidden_layer_features=args.ja_char_hidden_layer_features,
                                 ja_char_embedding_size=args.ja_char_embedding_size,
                                 ja_mental_hidden_layer_features=args.ja_mental_hidden_layer_features,
                                 ja_mental_embedding_size=args.ja_mental_embedding_size,
                                 num_actions=num_actions)


def make_iterative_action_tommas_transformer(args: Namespace, observation_space, action_space):
    num_joint_agent_features = observation_space['trajectory'][2]
    num_actions = action_space['action'][1]
    return IterativeActionTOMMASTransformer(model_type=args.model, model_name=get_model_name(args),
                                            learning_rate=args.learning_rate, optimizer_type=args.optimizer,
                                            num_joint_agent_features=num_joint_agent_features,
                                            hidden_layer_features=args.hidden_layer_features,
                                            embedding_size=args.embedding_size,
                                            n_head=args.n_head, n_layer=args.n_layer, num_actions=num_actions)


def make_iterative_action_lstm(args: Namespace, observation_space, action_space):
    num_joint_agent_features = observation_space['trajectory'][2]
    num_actions = action_space['action'][1]
    return IterativeActionLSTM(model_type=args.model, model_name=get_model_name(args),
                               learning_rate=args.learning_rate,
                               num_joint_agent_features=num_joint_agent_features,
                               hidden_layer_features=args.hidden_layer_features,
                               embedding_size=args.embedding_size, num_actions=num_actions, n_layer=args.n_layer)


def make_iterative_action_past_current(args: Namespace, observation_space, action_space):
    num_joint_agent_features = observation_space['current_trajectory'][2]
    num_actions = action_space['action'][1]
    return IterativeActionPastCurrentNet(model_type=args.model, model_name=get_model_name(args),
                                         learning_rate=args.learning_rate, optimizer_type=args.optimizer,
                                         num_joint_agent_features=num_joint_agent_features,
                                         lstm_char=args.lstm_char, lstm_mental=args.lstm_mental,
                                         char_hidden_layer_features=args.char_hidden_layer_features,
                                         char_embedding_size=args.char_embedding_size,
                                         char_n_head=args.char_n_head, char_n_layer=args.char_n_layer,
                                         mental_hidden_layer_features=args.mental_hidden_layer_features,
                                         mental_embedding_size=args.mental_embedding_size,
                                         mental_n_head=args.mental_n_head, mental_n_layer=args.mental_n_layer,
                                         num_actions=num_actions
                                         )


def make_gridworld_tomnet(args: Namespace, observation_space, action_space):
    num_gridworld_features = observation_space['current_trajectory'][2]

    num_actions = action_space['action'][1]
    num_goals = action_space['goal_consumption'][1]
    return GridworldToMNet(model_type=args.model, model_name=get_model_name(args), learning_rate=args.learning_rate,
                           optimizer_type=args.optimizer,
                           num_gridworld_features=num_gridworld_features,
                           gridworld_embedding_size=args.gridworld_embedding_size,
                           action_embedding_size=args.action_embedding_size,
                           lstm_char=args.lstm_char, lstm_mental=args.lstm_mental,
                           char_hidden_layer_features=args.char_hidden_layer_features,
                           char_embedding_size=args.char_embedding_size,
                           char_n_head=args.char_n_head, char_n_layer=args.char_n_layer,
                           mental_hidden_layer_features=args.mental_hidden_layer_features,
                           mental_embedding_size=args.mental_embedding_size,
                           mental_n_head=args.mental_n_head, mental_n_layer=args.mental_n_layer,
                           pred_net_features=args.pred_net_features, num_agents=4,
                           num_actions=num_actions, num_goals=num_goals,
                           no_action_loss=args.no_action_loss, no_goal_loss=args.no_goal_loss,
                           no_sr_loss=args.no_sr_loss)
