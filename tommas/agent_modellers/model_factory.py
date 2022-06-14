from typing import Union
from argparse import Namespace, ArgumentParser

from tommas.agent_modellers.gridworld_tomnet import GridworldToMNet
from tommas.agent_modellers.iterative_action_tomnet import IterativeActionPastCurrentNet
from tommas.data.datamodule_factory import get_datasets_abbreviation_str


MODELS = {"iterative_action_past_current": IterativeActionPastCurrentNet,
          "gridworld_tomnet": GridworldToMNet
          }


def make_model(args: Union[Namespace, dict], observation_space, action_space):
    if isinstance(args, dict):
        args = Namespace(**args)
    check_model_type(args)
    model_factory_fn = {"iterative_action_past_current": make_iterative_action_past_current,
                        "gridworld_tomnet": make_gridworld_tomnet
                        }
    model_type = args.model
    return model_factory_fn[model_type](args, observation_space, action_space)


def check_model_type(args: Namespace):
    if args.model not in MODELS:
        raise ValueError("model %s is not a known model. Choose from the following models: %s" %
                         (args.model, list(MODELS.keys())))


def get_model_name(args: Namespace):
    model_name_abbreviations = {"iterative_action_past_current": "IterPastCur",
                                "gridworld_tomnet": "GridTNet",
                                }
    return model_name_abbreviations[args.model] + get_datasets_abbreviation_str(args)


def add_model_specific_args(parser: ArgumentParser, args: Namespace) -> ArgumentParser:
    check_model_type(args)
    return MODELS[args.model].add_model_specific_args(parser)


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
