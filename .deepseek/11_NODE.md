# §11 · The ROS 2 node and launch files

**Governs:** `cpp_admm/include/cpp_admm/consensus_node.hpp`, `cpp_admm/src/consensus_node.cpp`,
`cpp_admm/launch/4_agent_admm.launch.py`, `cpp_admm/launch/time_varying_graph.launch.py`
**Milestone:** M8
**Done when:** `ros2 launch cpp_admm 4_agent_admm.launch.py` drives four **separate processes** to a
formation with no arguments.

---

## §11.1 One process per agent

Not one process with four nodes, and not one node with four kernels. The claim of this package is
that the computation is distributed, so the demo has to be distributable — a single container would
make a shared-memory bug invisible, and a reviewer who reads the launch file will check.

### Topics

| Direction | Topic | Type |
| --- | --- | --- |
| sub | `~/state` | `geometry_msgs/PoseStamped` (or `nav_msgs/Odometry`, see §11.4) |
| sub | `/admm/copy/<subject>` | `std_msgs/Float64MultiArray` |
| sub | `/admm/consensus/<neighbor>` | `std_msgs/Float64MultiArray` |
| sub | `/admm/graph` | `std_msgs/Int32MultiArray` |
| pub | `/admm/copy/<neighbor>` | `std_msgs/Float64MultiArray` |
| pub | `/admm/consensus/<agent_id>` | `std_msgs/Float64MultiArray` |
| pub | `~/cmd` | `geometry_msgs/AccelStamped` |
| pub | `~/diagnostics` | `std_msgs/Float64MultiArray` |

`Float64MultiArray` avoids a message-generation dependency; `layout.dim` carries
`(subject, admm_iteration, control_step)`. **This is a deliberate trade and it has a cost**: an
untyped array has no schema and will eventually be mismatched silently. Say so in the header comment,
and replace it with a typed `.msg` if this graduates into `transition-viable-swarm`.

### QoS

ADMM traffic: **best-effort, volatile, depth 1.**

Reliable QoS is the wrong choice here. A retransmitted stale iterate is worse than a dropped one,
because the kernel handles a miss explicitly (`messages_missing`, staleness tracking) but cannot tell
that a late arrival is late. Depth 1 plus best-effort is the transport-level equivalent of
`ZMQ_CONFLATE` ([09_CPP_KERNEL.md §9.9](09_CPP_KERNEL.md)).

### Threading

Single-threaded executor, one timer at `control_rate_hz`. The ADMM loop spins inside the timer
callback.

## §11.2 Parameters

Declared in the constructor, all overridable from launch. These keys are the contract with the launch
files — change them in the same commit or not at all.

| Parameter | Type | Default |
| --- | --- | --- |
| `agent_id` | int | required |
| `n_agents` | int | required |
| `neighbors` | int[] | required — open neighbourhood |
| `horizon` | int | 15 |
| `dt` | double | 0.1 |
| `control_rate_hz` | double | 10.0 |
| `rho` | double | 1.0 |
| `max_iterations` | int | 50 |
| `alpha` | double | 1.6 |
| `adaptive_rho` | bool | false |
| `max_staleness` | int | 5 |
| `formation_offsets` | double[] | flattened `(neighbor, dx, dy)` triples |
| `is_leader` | bool | false |
| `reference_topic` | string | `""` |
| `u_max`, `v_max` | double | 3.0, 2.0 |
| `q_position` … `w_formation` | double | see `AgentConfig` |

`control_rate_hz` MUST equal `1/dt` unless there is a stated reason. A mismatch means the plant
advances by a different amount than the prediction assumed, and every trajectory is then optimal for
a system you are not controlling ([§16.3](16_CONVENTIONS.md)).

`max_iterations` defaults to 50 because this is a **real-time budget, not a convergence target**.
With warm starting, a warm-started 10–20 iterations beats a cold 200
([08_CLOSED_LOOP.md §8.3](08_CLOSED_LOOP.md)).

`loadConfig()` calls `config.validate()` and lets the exception propagate
([09_CPP_KERNEL.md §9.2](09_CPP_KERNEL.md)).

## §11.3 `RosTransport`

Owned by `ConsensusNode`, declared in the header so it can be tested against a mock node.

Holds one publisher per neighbour for local copies, one for its own consensus value, matching
subscriptions, and a depth-1 inbox per `(kind, subject)` guarded by a mutex — the callbacks run on
the executor thread.

> **`poll()` must pump the executor, not sleep.**
>
> ROS 2 subscription callbacks only run while the executor is spinning. A `poll` that sleeps
> guarantees an empty inbox, and the kernel then degrades to a fully asynchronous run **while
> appearing to work** — the demo still moves, the formation still forms, and every staleness counter
> is wrong. Call `rclcpp::spin_some(node)` in a loop until the timeout expires or every expected
> subject has arrived.

`reconfigure(neighbors)` creates publishers and subscriptions for added neighbours, resets those for
removed ones, and clears their inbox entries.

## §11.4 `ConsensusNode::controlStep`

```
1. if now − last_state_stamp > 3 × control period  →  enterSafeState("stale state")
2. transport->flush(); kernel->setControlStep(++control_step)
3. kernel->setInitialState(x0); setReference(reference) for leaders
4. stats = kernel->solve()
5. if !stats.converged  →  RCLCPP_WARN_THROTTLE, but still apply the input
6. publishCommand(kernel->firstInput()); kernel->shiftWarmStart(); publishDiagnostics(stats)
```

Step 5 is the distinction that matters, and it is easy to get backwards:

| Situation | Response |
| --- | --- |
| ADMM hit the iteration cap | **apply the input.** An unconverged ADMM iterate is a suboptimal *feasible* input. |
| The QP failed outright, or staleness exceeded `max_staleness` | `enterSafeState`. |

`enterSafeState` publishes zero acceleration, sets the flag, warns throttled, and recovers only on
fresh state — **resetting the kernel warm start on recovery**, since the stored iterate is stale by
an unknown amount.

`flush()` at every control-step boundary stops a slow agent poisoning the next step with packets
from the previous one.

`onState` prefers `nav_msgs/Odometry` if the estimator publishes it. Numerically differentiating a
`PoseStamped` at 10 Hz gives a velocity too noisy for a `v_max` constraint, and the symptom is a
controller that saturates its velocity box on sensor noise.

`onGraphUpdate` logs the old and new neighbourhoods at INFO. These events are what the thesis
analyses; they must be reconstructable from a bag file alone. It calls `transport->reconfigure()`
then `kernel->setNeighbors()`, both of which reallocate — rate-limit upstream, do not call this at
the control rate.

`publishDiagnostics` packs iterations, local residuals, `rho`, the `qp_time_ms` / `comm_time_ms`
split, message counts and staleness into a `Float64MultiArray` with **named dims**, so
`ros2 topic echo` is readable without a decoder.

Nothing unthrottled in the control loop ([§16.6](16_CONVENTIONS.md)). A `printf` at 10 Hz × 4
processes is a latency bug, not a debugging aid.

`main()` catches `std::exception` around construction, logs it, and exits non-zero. A launch file
starting four agents must make a bad config obvious rather than leaving one agent silently dead —
and a 4-agent formation with 3 live agents converges to something, which is exactly why this matters.

## §11.5 `4_agent_admm.launch.py`

Arguments: `n_agents` (4), `topology` (`cycle` | `complete` | `path` | `star`), `formation`
(`square` | `line` | `v`), `formation_scale`, `rho`, `max_iterations`, `horizon`,
`control_rate_hz`, `leader` (−1 for none), `rviz`.

An `OpaqueFunction` is required: the edge list depends on the *value* of `n_agents`, which is not
available at description-build time.

**`build_topology` MUST produce edge lists identical to `CommunicationGraph`** — same node numbering,
same cycle orientation ([04_GRAPH.md §4.2](04_GRAPH.md)). The C++ and Python demos are compared
against each other, and a differently-numbered cycle makes that comparison meaningless while looking
fine.

`flatten_neighbor_offsets` emits `[j, dx, dy, …]` sorted by neighbour id, so the parameter dump is
diffable.

## §11.6 `time_varying_graph.launch.py`

Arguments: `n_agents`, `schedule` (`alternate` | `split_merge` | `random_failure`), `dwell_time`,
`loss_prob`, `record`, `seed`.

| Schedule | Behaviour |
| --- | --- |
| `alternate` | cycle ↔ path; connectivity stays positive, a visible transient at each switch |
| `split_merge` | cycle → two disconnected pairs → cycle; the components drift apart, then re-converge |
| `random_failure` | each edge present with probability `1 − loss_prob`, reseeded per switch |

`schedule_edges` MUST be deterministic given `seed` — the split/merge GIF has to be reproducible.

A small inline Python node publishes `/admm/graph`; every `ConsensusNode` recomputes its own
neighbourhood from it. **This makes the schedule centralised while the control stays distributed.**
That is a legitimate split — the schedule stands in for physical link availability — but say so in
the README ([13_DOCS.md §13.5](13_DOCS.md)) rather than letting a reader assume otherwise and find
it in the code.

## §11.7 Shared launch helpers

`build_topology` and `build_offsets` MUST live in **one** place —
`cpp_admm/launch/launch_utils.py`, imported by both launch files. Two divergent copies of the
topology definition is the one bug this pair of files cannot afford, and it is the default outcome
of copy-pasting the first file into the second.

Install the `launch/` directory as a whole (the skeleton's `CMakeLists.txt` already does) so the
import works from the install space, not only from the source tree.
