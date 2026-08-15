# Copyright (c) 2026, Ali-Eimaan. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Four agents plus a topology publisher that switches the graph on a schedule.

    ros2 launch cpp_admm time_varying_graph.launch.py schedule:=split_merge

This produces ``media/topology_switch.gif``. It is the demo that actually distinguishes
this repo: anyone can run ADMM on a fixed graph, and the interesting behaviour -- the
transient at each switch and the drift while the graph is split -- only shows up here.

Arguments
---------
n_agents      int     4
schedule      str     "alternate" | "split_merge" | "random_failure"
dwell_time    float   2.5    seconds between switches
loss_prob     float   0.0    Bernoulli loss injected by the topology publisher
record        bool    false  start a rosbag2 recording of /admm/** and /agent_*/cmd
seed          int     0

The graph publisher is a small Python node defined inline below. It publishes
``/admm/graph`` (Int32MultiArray flat edge list); every ConsensusNode recomputes its own
neighborhood from it. Note that this makes the *schedule* centralised while the *control*
stays distributed -- that is a deliberate and legitimate split (the schedule stands in for
physical link availability), but say so in the README rather than letting a reader assume
otherwise.
"""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Single source of truth for topology/formation/schedule definitions.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch_utils  # noqa: E402
from launch_utils import build_topology, schedule_edges  # noqa: E402


def _neighbors(edges: list[tuple[int, int]], n_agents: int) -> list[list[int]]:
    """Open neighbourhood per agent, sorted ascending (the kernel's contract)."""
    result: list[list[int]] = [[] for _ in range(n_agents)]
    for i, j in edges:
        result[i].append(j)
        result[j].append(i)
    for nbr in result:
        nbr.sort()
    return result


# A self-contained rclpy node, embedded so the schedule stays in this file rather than in
# a separate package. It imports the *same* schedule_edges the launch file uses (via the
# injected launch directory), so the schedule can never silently diverge from the launch
# arguments.
_GRAPH_PUBLISHER_CODE = r'''
import sys

sys.path.insert(0, __LAUNCH_DIR__)
from launch_utils import schedule_edges  # noqa: E402

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray


def parse_args(argv):
    kwargs = {}
    it = iter(argv)
    for token in it:
        if token == "--schedule":
            kwargs["schedule"] = next(it)
        elif token == "--n":
            kwargs["n"] = int(next(it))
        elif token == "--dwell":
            kwargs["dwell"] = float(next(it))
        elif token == "--loss":
            kwargs["loss"] = float(next(it))
        elif token == "--seed":
            kwargs["seed"] = int(next(it))
    return (
        kwargs["schedule"], kwargs["n"], kwargs["dwell"], kwargs["loss"], kwargs["seed"]
    )


def main():
    schedule, n_agents, dwell, loss_prob, seed = parse_args(sys.argv[1:])
    rclpy.init()
    node = Node("graph_scheduler")
    pub = node.create_publisher(Int32MultiArray, "/admm/graph", 1)

    index = 0

    def tick():
        nonlocal index
        edges = schedule_edges(schedule, n_agents, index, loss_prob, seed)
        msg = Int32MultiArray()
        msg.data = [int(v) for edge in edges for v in edge]
        node.get_logger().info("publishing /admm/graph: %s" % (msg.data,))
        pub.publish(msg)
        index += 1

    tick()
    node.create_timer(dwell, tick)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
'''.replace("__LAUNCH_DIR__", repr(os.path.dirname(os.path.abspath(__file__))))


def launch_setup(context, *args, **kwargs):
    """Emit the agent nodes, the graph publisher, and (optionally) the rosbag recorder."""
    n_agents = int(LaunchConfiguration("n_agents").perform(context))
    schedule = LaunchConfiguration("schedule").perform(context)
    dwell_time = float(LaunchConfiguration("dwell_time").perform(context))
    loss_prob = float(LaunchConfiguration("loss_prob").perform(context))
    seed = int(LaunchConfiguration("seed").perform(context))
    record = LaunchConfiguration("record").perform(context).lower() in ("true", "1")

    # Agents start on the cycle (the default topology) and correct themselves on the first
    # /admm/graph message. Rendezvous (no formation offsets) is the cleanest demonstration
    # of topology switching: split components each converge internally, drift apart, then
    # re-converge when the graph reconnects.
    neighbors = _neighbors(build_topology("cycle", n_agents), n_agents)
    nodes = []
    for i in range(n_agents):
        nodes.append(
            Node(
                package="cpp_admm",
                executable="consensus_node",
                name=f"agent_{i}",
                namespace=f"/agent_{i}",
                output="screen",
                parameters=[
                    {
                        "agent_id": i,
                        "n_agents": n_agents,
                        "neighbors": neighbors[i],
                        "formation_offsets": [],
                    }
                ],
            )
        )

    nodes.append(
        ExecuteProcess(
            cmd=[
                sys.executable,
                "-c",
                _GRAPH_PUBLISHER_CODE,
                "--schedule",
                schedule,
                "--n",
                str(n_agents),
                "--dwell",
                str(dwell_time),
                "--loss",
                str(loss_prob),
                "--seed",
                str(seed),
            ],
            output="screen",
        )
    )

    if record:
        nodes.append(
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "bag",
                    "record",
                    "-o",
                    "topology_switch",
                    "-e",
                    r"/admm/.*|/agent_.*/cmd",
                ],
                output="screen",
            )
        )

    return nodes


def generate_launch_description() -> LaunchDescription:
    """Declare the arguments in the module docstring, then defer to launch_setup."""
    args = [
        DeclareLaunchArgument("n_agents", default_value="4"),
        DeclareLaunchArgument("schedule", default_value="alternate"),
        DeclareLaunchArgument("dwell_time", default_value="2.5"),
        DeclareLaunchArgument("loss_prob", default_value="0.0"),
        DeclareLaunchArgument("record", default_value="false"),
        DeclareLaunchArgument("seed", default_value="0"),
    ]
    return LaunchDescription([*args, OpaqueFunction(function=launch_setup)])
