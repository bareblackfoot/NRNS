import torch
import numpy as np
import quaternion
import networkx as nx
from torch_geometric.data import Data
from habitat_sim import ShortestPath
from src.utils.sim_utils import diff_rotation


def gather_graph(agent):
    node_list = [n for n in agent.graph]
    adj = [list(e) for e in agent.graph.edges]
    for i in range(len(adj)):
        adj[i][0] = node_list.index(adj[i][0])
        adj[i][1] = node_list.index(adj[i][1])
    adj = torch.tensor(adj).T
    edge_attr = torch.stack([e[2]["attr"] for e in agent.graph.edges.data()])

    origin_nodes = []
    edge_rot = []
    unexplored_nodes = [
        n[0] for n in agent.graph.nodes.data() if n[1]["status"] == "unexplored"
    ]
    for e in agent.graph.edges.data():
        if e[1] in unexplored_nodes:
            origin_nodes.append(node_list.index(e[0]))
            edge_rot.append(e[2]["rotation"])

    feats = agent.node_feats.clone().detach()
    unexplored_indexs = []
    # connected_nodes = {}
    for i, n in enumerate(agent.graph):
        if agent.graph.nodes.data()[n]["status"] == "unexplored":
            unexplored_indexs.append(i)
            # connected_nodes[i] = [edge[0] for edge in adj.T.cpu().detach().numpy() if edge[1]==i] + [edge[1] for edge in adj.T.cpu().detach().numpy() if edge[0]==i]

    geo_data = Data(
        goal_feat=agent.goal_feat.clone().detach(),
        x=feats[node_list],
        edge_index=adj.cuda(),
        edge_attr=edge_attr.cuda(),
        ue_nodes=torch.tensor(unexplored_indexs).cuda(),
        num_nodes=len(node_list),
    )
    # geo_data = Data(
    #     goal_feat=agent.goal_feat.clone().detach().cpu(),
    #     x=agent.node_feats.clone().detach()[node_list].cpu(),
    #     edge_index=adj.detach().cpu(),
    #     edge_attr=edge_attr.detach().cpu(),
    #     ue_nodes=torch.tensor(unexplored_indexs).detach().cpu(),
    #     num_nodes=len(node_list),
    # )
    # geo_data.x = geo_data.x.cuda()
    # geo_data.goal_feat = geo_data.goal_feat.cuda()
    return geo_data, unexplored_indexs


def predict_distances(agent):
    if agent.use_gt_distances:
        pred_dists = gt_distances(agent)
        total_cost = add_travel_distance(agent, pred_dists, rot_thres=0.25)
        next_node = agent.unexplored_nodes[total_cost.argmin()]
        pred_dists = np.asarray(pred_dists)
        if total_cost[total_cost.argmin()] - pred_dists[pred_dists.argmin()] >= 10:
            next_node = agent.unexplored_nodes[pred_dists.argmin()]
    else:
        with torch.no_grad():
            geo_data, ue_nodes = gather_graph(agent)
            # agent.feat_model.module(geo_data)
            output = agent.feat_model([geo_data]).detach().cpu().squeeze(1)
            pred_dists = 10 * (1 - output)[ue_nodes] #(1 - x)*10  = min(valid_geo, 10.0)
            pred_dists = pred_dists.sqrt()
            total_cost = add_travel_distance(agent, pred_dists, rot_thres=0.25)
            next_node = agent.unexplored_nodes[total_cost.argmin()]
    return next_node


def add_travel_distance(agent, pred_dists, rot_thres):
    total_cost = []
    for n, goal_dist in zip(agent.unexplored_nodes, pred_dists):
        travel_dist = 0.25 * len(
            nx.shortest_path(agent.graph, source=agent.current_node, target=n)
        )
        quat1 = quaternion.from_float_array(agent.current_rot)
        quat2 = quaternion.from_float_array(agent.node_rots[n])
        rot_diff = rot_thres * diff_rotation(quat1, quat2) / 15
        total_cost.append(travel_dist + goal_dist + rot_diff)
    return np.asarray(total_cost)

def gt_distances(agent):
    distances = []
    for node_id in agent.unexplored_nodes:
        path = ShortestPath()
        path.requested_start = agent.node_poses[node_id].numpy().tolist()
        path.requested_end = agent.goal_pos.tolist()
        agent.pathfinder.find_path(path)
        distances += [path.geodesic_distance]
    return distances
