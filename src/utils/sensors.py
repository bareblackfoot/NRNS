from typing import Any
import numpy as np
from gym import spaces
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, RGBSensor, DepthSensor, SemanticSensor
import torch
import habitat_sim


@registry.register_sensor(name="PanoramicPartRGBSensor")
class PanoramicPartRGBSensor(RGBSensor):
    def __init__(self, config, **kwargs: Any):
        self.config = config
        self.angle = config.ANGLE
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "rgb_" + self.angle

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, 3),
            dtype=np.uint8,
        )

    def get_observation(self, obs, *args: Any, **kwargs: Any) -> Any:
        obs = obs.get(self.uuid, None)
        return obs

@registry.register_sensor(name="PanoramicPartSemanticSensor")
class PanoramicPartSemanticSensor(RGBSensor):
    def __init__(self, config, **kwargs: Any):
        self.config = config
        self.angle = config.ANGLE
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "semantic_" + self.angle

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=np.Inf,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.uint8,
        )

    def get_observation(self, obs, *args: Any, **kwargs: Any) -> Any:
        obs = obs.get(self.uuid, None)
        return obs

@registry.register_sensor(name="PanoramicPartDepthSensor")
class PanoramicPartDepthSensor(DepthSensor):
    def __init__(self, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self.angle = config.ANGLE
        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "depth_" + self.angle

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.DEPTH

    def get_observation(self, obs,*args: Any, **kwargs: Any):
        obs = obs.get(self.uuid, None)
        if isinstance(obs, np.ndarray):
            obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)
            obs = np.expand_dims(
                obs, axis=2
            )
        else:
            obs = obs.clamp(self.config.MIN_DEPTH, self.config.MAX_DEPTH)

            obs = obs.unsqueeze(-1)

        if self.config.NORMALIZE_DEPTH:
            obs = (obs - self.config.MIN_DEPTH) / (
                self.config.MAX_DEPTH - self.config.MIN_DEPTH
            )
        return obs

@registry.register_sensor(name="PanoramicRGBSensor")
class PanoramicRGBSensor(Sensor):
    def __init__(self, config, **kwargs: Any):
        self.config = config
        self.torch = False #config.HABITAT_SIM_V0.GPU_GPU
        self.num_camera = config.NUM_CAMERA
        self.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        super().__init__(config=config)

        # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        # return "panoramic_rgb"
        return "rgb"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, 3),
            dtype=np.uint8,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations,*args: Any, **kwargs: Any):
        if isinstance(observations['rgb_0'][:,:,:3], torch.Tensor):
            rgb_list = [observations['rgb_%d' % (i)][:, :, :3] for i in range(self.num_camera)]
            rgb_array = torch.cat(rgb_list, 1)
        else:
            rgb_list = [observations['rgb_%d' % (i)][:, :, :3] for i in range(self.num_camera)]
            rgb_array = np.concatenate(rgb_list, 1)
        if rgb_array.shape[1] > self.config.HEIGHT*4:
            left = rgb_array.shape[1] - self.config.HEIGHT*4
            slice = left//2
            rgb_array = rgb_array[:,slice:slice+self.config.HEIGHT*4]
        return rgb_array
        #return make_panoramic(observations['rgb_left'],observations['rgb'],observations['rgb_right'], self.torch)

@registry.register_sensor(name="PanoramicDepthSensor")
class PanoramicDepthSensor(DepthSensor):
    def __init__(self, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        #self.agent_id = config.AGENT_ID
        if config.NORMALIZE_DEPTH: self.depth_range = [0,1]
        else: self.depth_range = [config.MIN_DEPTH, config.MAX_DEPTH]
        self.min_depth_value = config.MIN_DEPTH
        self.max_depth_value = config.MAX_DEPTH
        self.num_camera = config.NUM_CAMERA
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.depth_range[0],
            high=self.depth_range[1],
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        # return "panoramic_depth"
        return "depth"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.DEPTH

    def get_observation(self, observations,*args: Any, **kwargs: Any):
        depth_list = [observations['depth_%d' % (i)] for i in range(self.num_camera)]

        if isinstance(depth_list[0], np.ndarray):
            obs = np.concatenate(depth_list, 1)
            obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)
            obs = np.expand_dims(
                obs, axis=2
            )
        else:
            obs = torch.cat(depth_list, 1)
            obs = obs.clamp(self.config.MIN_DEPTH, self.config.MAX_DEPTH)

            obs = obs.unsqueeze(-1)

        if self.config.NORMALIZE_DEPTH:
            obs = (obs - self.config.MIN_DEPTH) / (
                self.config.MAX_DEPTH - self.config.MIN_DEPTH
            )

        if obs.shape[1] > self.config.HEIGHT*4:
            left = obs.shape[1] - self.config.HEIGHT*4
            slice = left//2
            obs = obs[:,slice:slice+self.config.HEIGHT*4]

        return obs

@registry.register_sensor(name="PanoramicSemanticSensor")
class PanoramicSemanticSensor(SemanticSensor):
    def __init__(self, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self.torch = False#sim.config.HABITAT_SIM_V0.GPU_GPU
        self.num_camera = config.NUM_CAMERA

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=np.Inf,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "panoramic_semantic"
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC
    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations,*args: Any, **kwargs: Any):
        depth_list = [observations['semantic_%d'%(i)] for i in range(self.num_camera)]
        if isinstance(depth_list[0], torch.Tensor):
            return torch.cat(depth_list, 1)
        else:
            return np.concatenate(depth_list,1)