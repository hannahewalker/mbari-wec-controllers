# Copyright 2022 Open Source Robotics Foundation, Inc. and Monterey Bay Aquarium Research Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


package_name = 'rolling_stats'

# launch args that override controller.yaml when given, e.g.
#   ros2 launch rolling_stats controller.launch.py scale_factor:=1.2 retract_factor:=0.8
OVERRIDES = {'scale_factor': float, 'retract_factor': float, 'pbloghome': str}


def launch_node(context):
    config = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'controller.yaml'
        )
    overrides = {name: convert(context.launch_configurations[name])
                 for name, convert in OVERRIDES.items() if context.launch_configurations[name]}

    node = Node(
        package=package_name,
        name='controller',
        executable='controller',
        parameters=[config, overrides]
    )
    return [node]


def generate_launch_description():
    ld = LaunchDescription()
    for name in OVERRIDES:
        ld.add_action(DeclareLaunchArgument(
            name, default_value='', description=f'{name} (default: value in controller.yaml)'))
    ld.add_action(OpaqueFunction(function=launch_node))

    return ld
