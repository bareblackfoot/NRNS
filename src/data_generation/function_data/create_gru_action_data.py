import numpy as np
import quaternion
import torch
import msgpack_numpy
import os
from tqdm import tqdm
import argparse
from src.utils.cfg import input_paths
from src.functions.validity_func.local_nav import LocalAgent, map_from_actions
from src.utils.sim_utils import set_up_habitat
import gzip, json


"""V2: Gets the data to train Behavioral Cloning from the trajectories"""
parser = argparse.ArgumentParser()
parser = input_paths(parser)   
args = parser.parse_args() 
args.sim_dir += f"{args.dataset}/"
args.base_dir += f"{args.dataset}/"
args.data_splits += f"{args.dataset}/"

def get_angle_diff(states, n1, n2):
    # Get rotational difference between two nodes
    quat1 = quaternion.from_float_array(states[n1][1])
    agent_rotation1 = -quaternion.as_rotation_vector(quat1)[1] * 180 / np.pi
    quat2 = quaternion.from_float_array(states[n2][1])
    agent_rotation2 = -quaternion.as_rotation_vector(quat2)[1] * 180 / np.pi
    if agent_rotation1 < 0:
        agent_rotation1 = 360 + agent_rotation1
    if agent_rotation2 < 0:
        agent_rotation2 = 360 + agent_rotation2
    d = agent_rotation1 - agent_rotation2
    if d > 180:
        d -= 360
    if d < -180:
        d += 360
    return d, agent_rotation1, agent_rotation2

def get_scenes(scenes, scanName):
    if args.dataset == "mp3d":
        scene = "{}{}/{}.glb".format(args.sim_dir, scanName, scanName)
    else:
        scene = "{}{}.glb".format(args.sim_dir, scanName)
    sim, _ = set_up_habitat(scene)
    pairs = []
    traj_feats = {}
    infoFile = trajectory_data_dir + "train_instances/" + scanName + ".json.gz"
    with gzip.open(infoFile, "r") as fin:
        info = json.loads(fin.read().decode("utf-8"))
    for scene in tqdm(scenes):
        """Load trajectory data"""
        scan_name = scene.split("_")[0]
        floor = scene.split("_")[1]
        states = info[int(scene.split("_")[1])]
        featFile = trajectory_data_dir + "train_instances/feats/" + scene + ".pt"
        feats = torch.load(featFile).squeeze(-1).squeeze(-1)
        actions = states['actions']
        # infoFile = trajectory_data_dir + "trajectoryInfo/" + scene + ".msg"
        # info = msgpack_numpy.unpack(open(infoFile, "rb"), raw=False)
        # states = info["states"]
        # actions = info["actions"]
        # featFile = trajectory_data_dir + "trajectoryFeats/" + scene + ".pt"
        # feats = torch.load(featFile).squeeze(-1).squeeze(-1)
        # try:
        #     assert len(states) == feats.shape[0]
        # except:
        #     import ipdb
        #
        #     ipdb.set_trace()
        #     print("error")
        #     continue

        """Loop over trajectory"""
        start_index = 0
        goal_index = min(len(actions) - 1, 25)
        traj_feats[scene] = {}
        while goal_index < len(actions):
            next_actions = actions[start_index:goal_index]
            next_actions.append(0)
            start_pos = states['poses'][start_index]
            start_rot = states['rotations'][start_index]
            # start_rot = -quaternion.as_euler_angles(start_rot)[1] * 180 / np.pi
            # start_rot = states[start_index][1]
            local_agent = LocalAgent(
                actuation_noise=False,
                pose_noise=False,
                curr_pos=start_pos,
                curr_rot=start_rot,
                map_size_cm=1200,
                map_resolution=5,
            )
            maps = map_from_actions(
                sim, local_agent, start_pos, start_rot, next_actions[:-1]
            )

            quat1 = quaternion.from_float_array(states['rotations'][start_index])
            quat2 = quaternion.from_float_array(states['rotations'][goal_index])
            agent_rotation1 = -quaternion.as_euler_angles(quat1)[1] * 180 / np.pi
            agent_rotation2 = -quaternion.as_euler_angles(quat2)[1] * 180 / np.pi
            # Save info and feats
            pairs.append(
                {
                    "goal_index": goal_index,
                    "agent_position1": states['poses'][start_index][0],
                    "agent_rotation1": agent_rotation1,
                    "agent_position2": states['poses'][goal_index][0],
                    "agent_rotation2": agent_rotation2,
                    "next_actions": next_actions,
                    "traj": scene,
                    "floor": floor,
                    "scan_name": scan_name,
                    "maps": maps,
                }
            )
            traj_feats[scene][str(goal_index)] = (
                feats[start_index : goal_index + 1].detach().numpy()
            )
            start_index += 10
            goal_index += 10

    msgpack_numpy.pack(
        pairs,
        open(action_dir + scanName + "_gru_action_data.msg", "wb"),
        use_bin_type=True,
    )
    print("saved abridged node data.")
    msgpack_numpy.pack(
        traj_feats,
        open(action_dir + scanName + "_gru_action_feats.msg", "wb"),
        use_bin_type=True,
    )
    print("saved abridged feat data.")
    sim.close()


def run_house(scanName):
    scenes = []
    for s in os.listdir(trajectory_feats_dir):
        if scanName in s:
            scenes.append(s[:-3])
    get_scenes(scenes, scanName)


if __name__ == "__main__":
    action_dir = args.base_dir + "behavioral_cloning/"
    # trajectory_data_dir = args.base_dir + "trajectory_data/"
    # trajectory_feats_dir = trajectory_data_dir + "trajectoryFeats/"
    # passive_scene_file = args.base_dir + "scenes_passive.txt"
    trajectory_data_dir = args.base_dir + "trajectory_data/"
    distance_data_dir = args.base_dir + "distance_data_straight/"
    trajectory_feats_dir = trajectory_data_dir + "train_instances/feats/"
    # passive_scene_file = args.base_dir + "scenes_train.txt"
    data_splits = f"/home/blackfoot/codes/NRNSD/data/data_splits/{args.dataset}/"
    passive_scene_file = data_splits + "scenes_passive.txt"
    with open(passive_scene_file) as f:
        passive_scenes = sorted([line.rstrip() for line in f])
    houseList = sorted(passive_scenes)
    for enum, scanName in enumerate(houseList):
        print(f"Current Scan {scanName}")
        run_house(scanName)