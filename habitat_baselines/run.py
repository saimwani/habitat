#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from habitat.utils.visualizations import maps
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config


from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_rotate_vector,
)


from typing import Any
import habitat
from gym import spaces
import math

# @habitat.registry.register_sensor(name="Agent_Orientation")
# class AgentOrientationSensor(habitat.Sensor):
#     def __init__(self, sim, config, **kwargs: Any):
#         super().__init__(config=config)

#         self._sim = sim

#     def _get_uuid(self, *args: Any, **kwargs: Any):
#         return "agent_orientation"

#     def _get_sensor_type(self, *args: Any, **kwargs: Any):
#         return habitat.SensorTypes.HEADING

#     # Defines the size and range of the observations of the sensor
#     def _get_observation_space(self, *args: Any, **kwargs: Any):
#         return spaces.Box(
#             low=np.finfo(np.float32).min,
#             high=np.finfo(np.float32).max,
#             shape=(3,),
#             dtype=np.float32,
#         )

#     def _quat_to_xy_heading(self, quat):
#         direction_vector = np.array([0, 0, -1])

#         heading_vector = quaternion_rotate_vector(quat, direction_vector)

#         phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
#         return np.array(phi)

#     def get_observation(
#         self, observations, *args: Any, episode, **kwargs: Any
#     ):
#         return self._quat_to_xy_heading(self._sim.get_agent_state().rotation)



# def get_get_topdown_map(sim):
#     top_down_map = maps.get_topdown_map(sim, map_resolution=(500, 500))
#     recolor_map = np.array(
#         [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
#     )
#     range_x = np.where(np.any(top_down_map, axis=1))[0]
#     range_y = np.where(np.any(top_down_map, axis=0))[0]
#     padding = int(np.ceil(top_down_map.shape[0] / 125))
#     range_x = (
#         max(range_x[0] - padding, 0),
#         min(range_x[-1] + padding + 1, top_down_map.shape[0]),
#     )
#     range_y = (
#         max(range_y[0] - padding, 0),
#         min(range_y[-1] + padding + 1, top_down_map.shape[1]),
#     )
#     top_down_map = top_down_map[
#         range_x[0] : range_x[1], range_y[0] : range_y[1]
#     ]
#     top_down_map = recolor_map[top_down_map][:,:,0]
#     left=(1500-top_down_map.shape[1])//2
#     right=1500-top_down_map.shape[1]-left
#     top=(1500-top_down_map.shape[0])//2
#     bot=1500-top_down_map.shape[0]-top
#     top_down_map=np.pad(top_down_map,((top, bot), (left, right)), 'constant', constant_values=((255,255), (255,255)))
#     return top_down_map

#############################################################
nothin = torch.rand(100,100)

@habitat.registry.register_sensor(name="Agent_Position")
class AgentPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)

        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "agent_position"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        return self._sim.get_agent_state().position



@habitat.registry.register_sensor(name="Agent_Map")
class AgentMapSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "agent_map"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.MAP

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(100,100),
            # shape=(, get_get_topdown_map(self._sim).shape[1]),
            dtype=np.float32,
        )

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        return torch.zeros(100,100)

@habitat.registry.register_measure
class EpisodeInfoExample(habitat.Measure):
    def __init__(self, sim, config, **kwargs: Any):
        # This measure only needs the config
        self._config = config
        super().__init__()

    # Defines the name of the measure in the measurements dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "episode_info"

    # This is called whenver the environment is reset
    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        # Our measure always contains all the attributes of the episode
        self._metric = vars(episode).copy()
        # But only on reset, it has an additional field of my_value
        # self._metric["my_value"] = self._config.VALUE

    # This is called whenver an action is taken in the environment
    def update_metric(self, *args: Any, episode, action, **kwargs: Any):
        # Now the measure will just have all the attributes of the episode
        self._metric = vars(episode).copy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    config.defrost()

     # Add things to the config to for the measure
    config.TASK_CONFIG.TASK.EPISODE_INFO_EXAMPLE = habitat.Config()
    # The type field is used to look-up the measure in the registry.
    # By default, the things are registered with the class name
    config.TASK_CONFIG.TASK.EPISODE_INFO_EXAMPLE.TYPE = "EpisodeInfoExample"
    # config.TASK_CONFIG.TASK.EPISODE_INFO_EXAMPLE.VALUE = 5
    # Add the measure to the list of measures in use
    config.TASK_CONFIG.TASK.MEASUREMENTS.append("EPISODE_INFO_EXAMPLE")


    config.TASK_CONFIG.TASK.AGENT_MAP_SENSOR = habitat.Config()  ###
    config.TASK_CONFIG.TASK.AGENT_MAP_SENSOR.TYPE = "Agent_Map" ###
    config.TASK_CONFIG.TASK.SENSORS.append("AGENT_MAP_SENSOR")
    config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")



    config.TASK_CONFIG.TASK.AGENT_POSITION_SENSOR = habitat.Config()  ###
    config.TASK_CONFIG.TASK.AGENT_POSITION_SENSOR.TYPE = "Agent_Position" ###
    config.TASK_CONFIG.TASK.SENSORS.append("AGENT_POSITION_SENSOR")


###
    
##

    config.TASK_CONFIG.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


if __name__ == "__main__":
    main()