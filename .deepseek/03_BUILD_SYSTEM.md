# §3 · Build system

**Governs:** `python/pyproject.toml`, `cpp_admm/CMakeLists.txt`, `cpp_admm/package.xml`,
`python/distributed_mpc_admm/__init__.py`
**Milestone:** throughout — these files are complete in the skeleton; keep them correct.

Two independent build systems, deliberately. The Python package is installable and testable with no
ROS on the machine at all, which is what makes M1–M6 a fast loop.

---

## §3.1 Two trees, one algorithm

| Tree | Built by | Depends on |
| --- | --- | --- |
| `python/` | `pip install -e ".[dev]"` | NumPy, SciPy, CVXPY, OSQP, NetworkX, Matplotlib |
| `cpp_admm/` | `colcon build` | ROS 2, Eigen, OSQP, optionally libzmq |

**Neither depends on the other at build time.** The only coupling is the parity fixture
`cpp_admm/test/data/four_agent_reference.json`, which the Python side writes and the C++ side reads
([§10.5](10_TESTS.md)).

`AdmmKernel` MUST NOT include a ROS header. That is not stylistic: the kernel is destined for the
author's `transition-viable-swarm`, and a ROS type on its interface would drag the whole middleware
in. The transport abstraction ([09_CPP_KERNEL.md §9.1](09_CPP_KERNEL.md)) is what keeps that true.

## §3.2 `cpp_admm/CMakeLists.txt`

Complete in the skeleton. The one `TODO(deepseek §3.2)` asks you to verify it configures once the
sources compile, then delete the marker.

Structure, and why each piece is the way it is:

- `cmake_minimum_required(VERSION 3.22...3.31)` — the range form. CMake 4.x warns on bare
  compatibility levels below 3.5 and the upper bound stops a future CMake from applying new
  policies silently.
- `CMAKE_CXX_STANDARD 20`, sources C++17-compatible ([§16.7](16_CONVENTIONS.md)). Risk V3.
- `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`. `-Wconversion` is on deliberately: this code
  mixes `int` indices, `Eigen::Index` and `std::size_t`, and a narrowing conversion in a block
  offset is exactly the class of bug §16.1 is about. **Fix the warnings; do not silence them.**
- `admm_kernel` is a `SHARED` library built from `admm_kernel.cpp` and `per_agent_qp.cpp`, linking
  `Eigen3::Eigen` and `osqp::osqp` **publicly** (both appear in the installed headers).
- `consensus_node` is an executable linking `admm_kernel` plus the ROS targets.

## §3.3 Optional ZeroMQ

`libzmq` is found through `pkg_check_modules(ZMQ QUIET libzmq)` and gates
`CPP_ADMM_WITH_ZMQ`. When absent, the build still succeeds and `ZeroMqTransport`'s constructor
throws `std::runtime_error` naming the missing dependency
([09_CPP_KERNEL.md §9.9](09_CPP_KERNEL.md)).

**It must not silently degrade into a no-op transport.** A transport that accepts publishes and
never delivers turns every agent into an isolated solver, and the resulting run looks like a slow
convergence rather than a broken build.

## §3.4 Dependency linkage — not `ament_target_dependencies`

Deprecated from Kilted onward (risk V6). Use namespaced targets:

```cmake
target_link_libraries(consensus_node PRIVATE
  ${rclcpp_TARGETS} ${geometry_msgs_TARGETS} ${std_msgs_TARGETS})
```

`find_package(<pkg> REQUIRED)` still comes first — it is what defines `${<pkg>_TARGETS}`.

## §3.5 The Python package

`pyproject.toml` is complete. Two things to keep true:

- `requires-python = ">=3.12"` and the classifier list MUST match the CI matrix in
  [14_CI.md §14.1](14_CI.md). Three places, one fact.
- `license = "BSD-3-Clause"` is the PEP 639 SPDX form and needs `setuptools>=77`
  ([§2.6](02_ENVIRONMENT.md)).

`distributed_mpc_admm/__init__.py` carries `TODO(deepseek §3.5)`. Replace it with plain top-level
imports of every name already listed in `__all__`:

```python
from .communication_graph import CommunicationGraph, LossyChannel, Message, TimeVaryingGraph
from .consensus_admm import ADMMHistory, ADMMOptions, ADMMResult, ConsensusADMM, DistributedMPC, MPCOptions, SimulationLog
from .formation_constraints import FormationSpec, LeaderFollowerSpec
from .per_agent_solver import AgentCostWeights, AgentLimits, CvxpyAgentSolver, DoubleIntegrator, LocalProblemData, LocalSolution, PerAgentSolver
```

Keep `__all__` and the imports in sync — a name in one and not the other is a lint failure waiting
to happen, and `ruff`'s `F401` will catch the reverse.

**Guard nothing behind `try`/`except ImportError`.** If CVXPY is missing the package should fail to
import with a clear error, not half-work. A package that imports and then raises deep inside a
solve is far harder to diagnose than one that refuses at the door.

## §3.6 `package.xml`

Complete. `<license>BSD-3-Clause</license>` MUST agree with `LICENSE` and `pyproject.toml`.

`osqp_vendor` is declared as a `<depend>` and may not resolve on `lyrical` (risk V9). If it does
not, keep the declaration, build OSQP from source in the workflow, and record the tag in
[§2.2](02_ENVIRONMENT.md) — do not delete the dependency to make `rosdep install` quiet.

`cppzmq` is deliberately **not** declared: it is optional (§3.3) and a hard dependency on it would
make the package unbuildable on machines that only need the ROS transport.

## §3.7 Registering a new test

Adding a C++ test file means adding its block:

```cmake
ament_add_gtest(test_<name> test/test_<name>.cpp)
if(TARGET test_<name>)
  target_link_libraries(test_<name> admm_kernel)
endif()
```

**Forgetting this is silent** — the file sits in the tree, never compiles, and `colcon test` stays
green. The `if(TARGET …)` guard matters: `ament_add_gtest` quietly does nothing when gtest is
unavailable, and an unguarded `target_link_libraries` then fails at configure with a confusing
message about an unknown target.

Python tests need no registration; `pytest` discovers `python/tests/test_*.py` through the
`testpaths` setting.

## §3.8 What is deliberately not here

- **No `setup.py`.** `pyproject.toml` is sufficient and a second source of packaging truth is a
  drift risk.
- **No `src/disturbance_*`-style split of `admm_kernel.cpp`** yet. Split `InProcessTransport` and
  `ZeroMqTransport` into their own translation unit when `admm_kernel.cpp` passes ~600 lines; until
  then, one file keeps the phase ordering readable in one place.
- **No Python C extension, no pybind11 binding.** The parity check goes through a JSON fixture
  ([§10.5](10_TESTS.md)), which is slower and enormously simpler, and it keeps the two
  implementations genuinely independent — a binding would let one silently become the other.
