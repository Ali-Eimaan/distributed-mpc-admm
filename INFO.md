# GitHub Portfolio — Detailed Specification & PhD-Readiness Analysis

**Owner:** Pakistani robotics PhD candidate (MS Mechatronics, 5 yrs embedded/robotics, ROS 2 Jazzy + PX4 + Gazebo)
**Thesis target:** *Transition-Viable Distributed MPC-CBF for Aerial-Ground Swarms: A Hybrid-Systems Theory of Asynchronous Split, Merge, and Morph Events with Set-Valued Resets*
**Application target:** Fall 2027 PhD intake at hardware-equipped labs (Tier A: KTH, ETH, TU Delft, TU/e; Tier B: Waterloo, UTIAS, Polytechnique Montréal, NTNU; Tier C: HKUST, HKU, NTU, UNIST)
**Document purpose:** Define the full repository structure, file-level contents, and identify any gaps that need closing before December 2026 application submissions.

---

## How to read this document

For each repo I specify:
1. **Identity** — name, one-line tagline, primary language, lines-of-code estimate at v1.0 release.
2. **Purpose** — what it proves to a hardware-equipped advisor reading your GitHub.
3. **File-level structure** — every directory and the files in it, with one-sentence descriptions of what each non-boilerplate file contains.
4. **Demo deliverables** — videos, GIFs, plots that go in the README.
5. **CI / quality signals** — what GitHub Actions runs and what badges go on the README.
6. **Citation hook** — how this repo links back to the TVF/RRC-CBF/AHTD thesis.

At the end I do a hard portfolio gap analysis against what hardware-equipped PhD admissions committees actually look for.

# Repo 6 — `distributed-mpc-admm`
 
**Tagline:** Distributed MPC via consensus ADMM for 4-to-8 double-integrator agents in 2D, with formation control and time-varying graphs.
**Language:** Python (60%) + C++ (40%, the production ADMM kernel).
**Estimated size at v1.0:** ~4,000 LOC.
**Why this exists:** Proves you can distribute computation across agents, not just simulate centrally. ADMM is the workhorse of distributed control; if you can't write one, your whole thesis is suspect.
 
## Directory structure
 
```
distributed-mpc-admm/
├── .github/workflows/test.yml
├── python/
│   ├── distributed_mpc_admm/
│   │   ├── __init__.py
│   │   ├── consensus_admm.py
│   │   ├── per_agent_solver.py
│   │   ├── communication_graph.py
│   │   ├── formation_constraints.py
│   │   └── plotting.py
│   ├── notebooks/
│   │   ├── 01_admm_intuition.ipynb
│   │   ├── 02_4_agent_consensus.ipynb
│   │   ├── 03_formation_control.ipynb
│   │   ├── 04_switching_topology.ipynb
│   │   └── 05_convergence_analysis.ipynb
│   ├── tests/
│   │   ├── test_admm_convergence.py
│   │   └── test_formation_consensus.py
│   └── pyproject.toml
├── cpp_admm/
│   ├── CMakeLists.txt
│   ├── package.xml                  (ROS 2 package)
│   ├── include/cpp_admm/
│   │   ├── admm_kernel.hpp
│   │   ├── per_agent_qp.hpp
│   │   └── consensus_node.hpp
│   ├── src/
│   │   ├── admm_kernel.cpp
│   │   ├── per_agent_qp.cpp
│   │   └── consensus_node.cpp
│   ├── launch/
│   │   ├── 4_agent_admm.launch.py
│   │   └── time_varying_graph.launch.py
│   └── test/
│       └── test_admm_kernel.cpp
├── analysis/
│   ├── iterations_to_consensus.ipynb
│   ├── communication_load_study.ipynb
│   └── topology_robustness.ipynb
├── docs/
│   ├── derivations/
│   │   ├── consensus_admm_derivation.tex
│   │   ├── augmented_lagrangian.tex
│   │   └── convergence_proof.tex
│   ├── README_math.md
│   └── COMPARISON_VS_DUAL_DECOMP.md
├── media/
│   ├── 4_agent_formation.gif
│   ├── topology_switch.gif
│   └── convergence_curves.png
├── LICENSE
└── README.md
```
 
## File-level descriptions
 
- **`python/distributed_mpc_admm/consensus_admm.py`** — clean reference implementation of consensus ADMM for separable quadratic objectives over a graph. Pedagogy-first; not optimized.
- **`python/distributed_mpc_admm/per_agent_solver.py`** — per-agent local QP solver (CVXPY backend); takes neighbor states as ADMM dual variables.
- **`python/distributed_mpc_admm/communication_graph.py`** — `CommunicationGraph` class supporting time-varying topologies, packet loss, and delay simulation.
- **`python/distributed_mpc_admm/formation_constraints.py`** — encodes rigid formations, leader-follower, and consensus-on-relative-positions as ADMM-compatible constraints.
- **`cpp_admm/include/cpp_admm/admm_kernel.hpp`** — production-grade ADMM kernel using OSQP for the per-agent QP and ZeroMQ for inter-agent communication. **This is the kernel you actually drop into `transition-viable-swarm`.**
- **`cpp_admm/src/consensus_node.cpp`** — ROS 2 node that runs one agent's ADMM updates; configurable agent ID and neighbor list.
- **`docs/derivations/consensus_admm_derivation.tex`** — derives consensus ADMM from the augmented Lagrangian step by step.
- **`docs/derivations/convergence_proof.tex`** — Boyd's convergence proof in your own notation, with explicit identification of the assumptions that fail under switching topology (motivation for your AHTD object).
- **`docs/COMPARISON_VS_DUAL_DECOMP.md`** — when ADMM beats dual decomposition and vice versa.
## Demo deliverables
 
- 4-agent formation GIF, switching-topology GIF, convergence-curves plot.
## CI
 
- pytest + colcon build.
## Citation hook
 
The ADMM kernel is the distributed solver inside `transition-viable-swarm`. The convergence proof gap motivates your AHTD object: "ADMM convergence assumes synchronous updates; our AHTD object characterizes when *partially-ordered* asynchronous transitions still yield consensus."
 
---