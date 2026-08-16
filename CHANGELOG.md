# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-16

First public release. Everything below is new; the sections record what the release
contains rather than what changed since a previous tag.

### Added — Python reference (`python/distributed_mpc_admm/`)

- `communication_graph` — `CommunicationGraph` with complete / cycle / path / star /
  random-connected factories, graph Laplacian and algebraic connectivity;
  `TimeVaryingGraph` for switching schedules with dwell time and joint-connectivity
  queries; `LossyChannel` with Bernoulli loss and bounded delay.
- `per_agent_solver` — discrete double-integrator model with condensed prediction
  matrices, and `CvxpyAgentSolver`, a DPP-compliant local QP compiled once and re-solved
  through parameters.
- `formation_constraints` — `FormationSpec` (polygon, line, V, grid, rendezvous),
  rigidity matrix and margin, `LeaderFollowerSpec`, formation-error metrics, formation
  interpolation for morph events.
- `consensus_admm` — `ConsensusADMM` implementing general-form consensus ADMM in scaled
  dual form (over-relaxation, residual balancing, packet-loss handling), and
  `DistributedMPC`, the receding-horizon driver with solver caching and warm starting.
- `dual_decomposition` — a dual-decomposition baseline solving the same problems, used
  for the measured comparison in `docs/COMPARISON_VS_DUAL_DECOMP.md`.
- `plotting` — trajectory, convergence, formation-error, topology and bandwidth figures,
  plus the closed-loop animation used for the README media.

### Added — C++ / ROS 2 kernel (`cpp_admm/`)

- `PerAgentQp` — OSQP-backed local QP with a fixed sparsity pattern and value-only data
  updates, so a change in `rho`, `z`, `lambda` or the formation offsets never
  re-factorises the problem structure.
- `AdmmKernel` — allocation-free steady-state ADMM iteration behind a transport-agnostic
  interface, with staleness tracking and an early exit when neighbour data goes stale.
- Transports — `InProcessTransport` (deterministic, for tests), `ZeroMqTransport`
  (optional, requires cppzmq) and `RosTransport` inside the node.
- `ConsensusNode` — one ROS 2 node per agent, best-effort depth-1 QoS for ADMM traffic,
  a documented safe state, and runtime topology updates.
- Launch files for a fixed 4-agent formation and for a switching-topology demo.

### Added — Evidence

- Test suites: 45 Python tests and 20 gtest cases, including a C++/Python parity test
  that pins the two implementations to 1e-8 on the first 20 iterates.
- Notebooks `01`–`05` (ADMM intuition through convergence analysis) and three
  standalone studies under `analysis/`, all executed in CI.
- Derivations under `docs/derivations/` (augmented Lagrangian, consensus ADMM,
  convergence and where it stops applying) and a notation reference in
  `docs/README_math.md`.
- `docs/COMPARISON_VS_DUAL_DECOMP.md` with measured ADMM-versus-dual-decomposition
  numbers under a single protocol.

### Known limitations

Stated in full in the README's "What this does not do". In short: no recursive-feasibility
or stability guarantee, no collision avoidance, double-integrator dynamics only, and the
ADMM convergence guarantee holds only for a fixed connected graph with synchronous updates
and exact local solves — the switching and packet-loss results are measurements, not
theorems.

[0.1.0]: https://github.com/Ali-Eiman/distributed-mpc-admm/releases/tag/v0.1.0
