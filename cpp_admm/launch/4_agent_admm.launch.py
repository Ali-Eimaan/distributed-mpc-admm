# Copyright (c) 2026, Ali-Eimaan. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Launch four ConsensusNode processes on a cycle topology, square formation.

    ros2 launch cpp_admm 4_agent_admm.launch.py

One process per agent -- not one process with four nodes. The whole claim of this package
is that the computation is distributed, so the demo has to be distributable; running them
in a single container would make a shared-memory bug invisible.

Arguments
---------
n_agents          int     4
topology          str     "cycle" | "complete" | "path" | "star"
formation         str     "square" | "line" | "v"
formation_scale   float   1.0
rho               float   1.0
max_iterations    int     50
horizon           int     15
control_rate_hz   float   10.0
leader            int     -1       (-1 for no leader / pure rendezvous)
rviz              bool    false
"""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# The launch file and launch_utils.py live in the same directory; make the import work
# whether ros2 launch runs this from the install space or the source tree.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch_utils  # noqa: E402


def _neighbors(edges: list[tuple[int, int]], n_agents: int) -> list[list[int]]:
    """Open neighbourhood per agent, sorted ascending (the kernel's contract)."""
    result: list[list[int]] = [[] for _ in range(n_agents)]
    for i, j in edges:
        result[i].append(j)
        result[j].append(i)
    for nbr in result:
        nbr.sort()
    return result


def launch_setup(context, *args, **kwargs):
    """Resolve LaunchConfigurations and emit one Node action per agent.

    An OpaqueFunction is required because the edge list depends on the *value* of
    ``n_agents``, which is not available at description-build time.
    """
    n_agents = int(LaunchConfiguration("n_agents").perform(context))
    topology = LaunchConfiguration("topology").perform(context)
    formation = LaunchConfiguration("formation").perform(context)
    formation_scale = float(LaunchConfiguration("formation_scale").perform(context))
    rho = float(LaunchConfiguration("rho").perform(context))
    max_iterations = int(LaunchConfiguration("max_iterations").perform(context))
    horizon = int(LaunchConfiguration("horizon").perform(context))
    control_rate_hz = float(LaunchConfiguration("control_rate_hz").perform(context))
    leader = int(LaunchConfiguration("leader").perform(context))
    rviz = LaunchConfiguration("rviz").perform(context).lower() in ("true", "1")

    edges = launch_utils.build_topology(topology, n_agents)
    offsets = launch_utils.build_offsets(formation, n_agents, formation_scale)
    neighbors = _neighbors(edges, n_agents)

    nodes = []
    for i in range(n_agents):
        params = {
            "agent_id": i,
            "n_agents": n_agents,
            "neighbors": neighbors[i],
            "horizon": horizon,
            "dt": 1.0 / control_rate_hz,  # control_rate_hz MUST equal 1/dt
            "control_rate_hz": control_rate_hz,
            "rho": rho,
            "max_iterations": max_iterations,
            "formation_offsets": launch_utils.flatten_neighbor_offsets(
                i, neighbors[i], offsets
            ),
            "is_leader": (i == leader),
            "reference_topic": f"/agent_{leader}/reference" if i == leader else "",
        }
        nodes.append(
            Node(
                package="cpp_admm",
                executable="consensus_node",
                name=f"agent_{i}",
                namespace=f"/agent_{i}",
                output="screen",
                parameters=[params],
                # ~/state resolves to /agent_<i>/agent_<i>/state; remap it here to
                # whatever the simulator publishes (there is no simulator in this repo).
            )
        )

    if rviz:
        nodes.append(
            Node(package="rviz2", executable="rviz2", name="rviz2", output="screen")
        )

    return nodes


def generate_launch_description() -> LaunchDescription:
    """Declare the arguments in the module docstring, then defer to launch_setup."""
    args = [
        DeclareLaunchArgument("n_agents", default_value="4"),
        DeclareLaunchArgument("topology", default_value="cycle"),
        DeclareLaunchArgument("formation", default_value="square"),
        DeclareLaunchArgument("formation_scale", default_value="1.0"),
        DeclareLaunchArgument("rho", default_value="1.0"),
        DeclareLaunchArgument("max_iterations", default_value="50"),
        DeclareLaunchArgument("horizon", default_value="15"),
        DeclareLaunchArgument("control_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("leader", default_value="-1"),
        DeclareLaunchArgument("rviz", default_value="false"),
    ]
    return LaunchDescription([*args, OpaqueFunction(function=launch_setup)])
