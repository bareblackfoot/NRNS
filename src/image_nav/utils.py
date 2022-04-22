import torch
import torch.nn as nn
import numpy as np
from src.functions.target_func.switch_model import SwitchMLP
from src.functions.target_func.goal_model import GoalMLP
from src.functions.feat_pred_fuc.deepgcn import TopoGCN

"""Evaluate Episode"""


def evaluate_episode(agent, args, length_shortest):
    success = False
    dist_thres = 1.0
    dist_to_goal = np.linalg.norm(
        np.asarray(agent.goal_pos.tolist()) - np.asarray(agent.current_pos.tolist())
    )
    if dist_to_goal <= dist_thres and agent.steps < args.max_steps:
        success = True
    episode_spl = calculate_spl(success, length_shortest, agent.length_taken)
    return dist_to_goal, episode_spl, success


def calculate_spl(success, length_shortest, length_taken):
    spl = (length_shortest * 1.0 / max(length_shortest, length_taken)) * success
    return spl


"""Load Models"""


def load_models(args):
    """Action Pred function"""
    if args.bc_type == "map":
        from src.functions.bc_func.bc_map_network import ActionNetwork
    else:
        from src.functions.bc_func.bc_gru_network import ActionNetwork
    model_action = ActionNetwork()
    model_action.load_state_dict(torch.load(args.model_dir + args.bc_model_path))
    model_action.to(args.device)
    model_action.eval()

    """Load Switch function"""
    """stop 할지 안할지 결정"""
    model_switch = SwitchMLP()
    model_switch.load_state_dict(torch.load(args.model_dir + args.switch_model_path))
    print(sum(p.numel() for p in model_switch.parameters()))
    model_switch.to(args.device)
    model_switch.eval()

    """Load Target function"""
    """local goal 위치 예측 (rotation, location)"""
    model_goal = GoalMLP()
    model_goal.load_state_dict(torch.load(args.model_dir + args.goal_model_path))
    model_goal.to(args.device)
    model_goal.eval()

    """Load Distance function"""
    """갈 수 있는지 없는지 확인(explorable 한지 아닌지)"""
    # model_feat_pred = TopoGCN()
    # model_feat_pred.load_state_dict(torch.load(args.model_dir + args.distance_model_path))#.module
    # model_goal.to(args.device)
    model_feat_pred = TopoGCN()
    model_feat_pred = torch.load(args.model_dir + args.distance_model_path)#.module
    print(sum(p.numel() for p in model_feat_pred.parameters()))

    model_feat_pred.to(args.device)
    model_feat_pred.eval()

    return model_switch, model_goal, model_feat_pred, model_action
