import sys
import msgpack_numpy
import gzip, json

"""Builds graph from all the passive videos, expects feats to already be calculated"""


def make_traj_msg(episodes, scan_name):
    for episode in episodes:
        """load data"""
        episode_id = episode["episode_id"]
        episode.update({'states': []})
        for pose, rot in zip(episode['poses'], episode['rotations']):
            episode['states'].append((pose, rot))
        msgpack_numpy.pack(
            episode,
            open(save_dir + episode_id + "_graphs.msg", "wb"),
            use_bin_type=True,
        )
        print("saved at " + save_dir + episode_id + "_graphs.msg")


def run_house(house):
    infoFile = trajectory_data_dir + "train_instances/" + house + ".json.gz"
    with gzip.open(infoFile, "r") as fin:
        episodes = json.loads(fin.read().decode("utf-8"))
    make_traj_msg(episodes, house)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise Exception("missing dataset argument-- Options: 'gibson' or 'mp3d'")
    print("dataset:", sys.argv[1])
    dataset = sys.argv[1]

    if len(sys.argv) < 3:
        raise Exception("missing noise argument-- Options: 'no_noise' or 'noise'")
    noise = False
    print("noise on:", sys.argv[2])
    if sys.argv[2] == "noise":
        noise = True

    data_splits = f"./data/data_splits/{dataset}/"
    sim_dir = "./data/scene_datasets/"
    if dataset == "mp3d":
        sim_dir += f"{dataset}/"
    else:
        sim_dir += "gibson_train_val/"
    base_dir = f"./data/topo_nav/{dataset}/"
    visualization_dir = base_dir + "visualizations/visualized_graphs/"
    if noise:
        trajectory_data_dir = base_dir + "noise/trajectory_data/"
        save_dir = base_dir + "noise/trajectory_data/trajectoryInfo/"
        print("using noise")
    else:
        trajectory_data_dir = base_dir + "no_noise/trajectory_data/"
        save_dir = base_dir + "no_noise/trajectory_data/trajectoryInfo/"

    passive_scene_file = data_splits + "scenes_passive.txt"
    with open(passive_scene_file) as f:
        houseList = sorted([line.rstrip() for line in f])
    for enum, house in enumerate(houseList):
        print(house)
        run_house(house)
