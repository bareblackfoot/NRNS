import sys
sys.path.append("/home/blackfoot/codes/NRNSD")
import numpy as np
import quaternion
import torch
import msgpack_numpy
from numpy.linalg import inv, norm
import gzip, json
from src.data_generation.graph_data.graph import GraphMap, affinity_cluster
from src.data_generation.graph_data.points import (
    generate_new_points,
    se3_to_mat,
    gen_points,
)
from src.utils.sim_utils import set_up_habitat_panoramic as set_up_habitat
from src.utils.cfg import input_paths
import argparse

"""Builds graph from all the passive videos, expects feats to already be calculated"""


def get_scenes(pathfinder, sim, episodes, scan_name):
    graphs = []
    avg_edge_len = []
    for episode in episodes:
        """load data"""
        episode_id = episode["episode_id"]
        # featFile = trajectory_data_dir + "train_instances/feats/" + episode_id + ".pt"
        featFile = trajectory_data_dir + "trajectoryFeats/" + episode_id + ".pt"
        feats = torch.load(featFile).squeeze(-1).squeeze(-1)

        """build graph"""
        # G = GraphMap(params={"feat_size": 512, "frame_size": [480, 640, 3]})
        G = GraphMap(params={"feat_size": feats.shape[-1], "frame_size": list(sim.sensor_suite.observation_spaces.spaces['rgb'].shape)})
        G.build_graph(episode, feats)

        """cluster graph via affinity clustering"""
        try:
            a_clusteredGraph = affinity_cluster(G)
        except:
            continue

        """making sure valid points work via map"""
        # valid_points = generate_new_points(a_clusteredGraph, pathfinder)
        # invalid_points = generate_invalid_points(a_clusteredGraph, pathfinder)

        """gather valid points via depth map"""
        valid_points = gen_points(a_clusteredGraph, sim)

        """save affinty clustered graph data so we can input it to gnn"""
        nodes = {
            "ids": [],
            "feat": [],
            "pos": [],
            "rot": [],
        }
        for n in sorted(a_clusteredGraph.nodes):
            nodes["ids"].append(int(n.nodeid))
            nodes["feat"].append(n.feat[0].detach().numpy())
            nodes["pos"].append(n.pos)
            nodes["rot"].append(n.rot)

        edges = []
        edge_attrs = []
        for e in a_clusteredGraph.edges:
            edges.append([float(e.ids[0]), float(e.ids[1])])
            Anode = a_clusteredGraph.node_by_id[e.ids[0]]
            Bnode = a_clusteredGraph.node_by_id[e.ids[1]]
            A = se3_to_mat(
                quaternion.from_float_array(Anode.rot), np.asarray(Anode.pos)
            )
            B = se3_to_mat(
                quaternion.from_float_array(Bnode.rot), np.asarray(Bnode.pos)
            )
            pos_delta = inv(A) @ B
            edge_attrs.append(pos_delta)
            edges.append([float(e.ids[1]), float(e.ids[0])])
            pos_delta = inv(B) @ A
            edge_attrs.append(pos_delta)

            avg_edge_len.append(norm(np.asarray(Anode.pos) - np.asarray(Bnode.pos)))

        graphs.append(
            {
                "nodes": nodes,
                "edges": edges,
                "edge_attrs": edge_attrs,
                "valid_points": valid_points,
                "traj_name": episode_id,
                "scan_name": scan_name,
            }
        )
    msgpack_numpy.pack(
        graphs,
        open(clustered_graph_dir + scan_name + "_graphs.msg", "wb"),
        use_bin_type=True,
    )
    print("saved at " + clustered_graph_dir + scan_name + "_graphs.msg")


def run_house(house):
    if args.dataset == "mp3d":
        scene = "{}{}/{}.glb".format(sim_dir, house, house)
    else:
        scene = "{}/{}.glb".format(sim_dir, house)
    sim, pathfinder = set_up_habitat(scene)
    infoFile = trajectory_data_dir + "train_instances/" + house + ".json.gz"
    with gzip.open(infoFile, "r") as fin:
        episodes = json.loads(fin.read().decode("utf-8"))
    get_scenes(pathfinder, sim, episodes, house)
    pathfinder = None
    sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = input_paths(parser)
    args = parser.parse_args()
    args.base_dir += f"{args.dataset}/"
    args.data_splits += f"{args.dataset}/"
    args.trajectory_data_dir = f"{args.base_dir}{args.trajectory_data_dir}"

    # data_splits = f"data_splits/{args.dataset}/"
    sim_dir = "./data/scene_datasets/"
    if args.dataset == "mp3d":
        sim_dir += "mp3d/"
    else:
        sim_dir += "gibson_train_val"
    # base_dir = f"./data/topo_nav/{args.dataset}/"
    visualization_dir = args.base_dir + "visualizations/visualized_graphs/"
    if args.noise:
        trajectory_data_dir = args.base_dir + "noise/trajectory_data/"
        clustered_graph_dir = args.base_dir + "noise/clustered_graph/"
        print("using noise")
    else:
        trajectory_data_dir = args.base_dir + "no_noise/trajectory_data/"
        clustered_graph_dir = args.base_dir + "no_noise/clustered_graph/"

    passive_scene_file = args.data_splits + "scenes_passive.txt"
    with open(passive_scene_file) as f:
        houseList = sorted([line.rstrip() for line in f])
    for enum, house in enumerate(houseList):
        print(house)
        run_house(house)
