# distributed-mpc-admm

Distributed MPC via consensus ADMM for 4-to-8 double-integrator agents in 2D, with formation control and time-varying graphs.

<!-- TODO(deepseek §13.5): add badges once CI is green -- test workflow status, Python
     versions, ROS 2 Lyrical Luth, licence. Do not add a badge before the corresponding job
     passes; a red badge on the first screen is worse than no badge. -->

<!-- TODO(deepseek §13.5): hero GIF here (media/4_agent_formation.gif). The first thing a reader sees must
     be the system working, not a paragraph. -->

## What this is

<!-- TODO(deepseek §13.5): three sentences.
     1. The problem: N agents each solving their own MPC, coupled only through formation
        costs, with no central solver anywhere.
     2. The method: general-form consensus ADMM, one neighbor-communication round per
        iteration.
     3. What is here: a readable Python reference, a real-time C++/ROS 2 kernel, the
        derivations, and the experiments that show where the convergence theory stops
        applying. -->

## Results

<!-- TODO(deepseek §13.5): the numbers table, filled from notebook 05 and analysis/. Suggested columns:
     configuration | ADMM iterations to 1e-4 | closed-loop settling (s) | final formation
     error (cm) | per-step wall time (ms). Rows for 4/8 agents across cycle/complete/path.
     Quote the hardware and the solver versions underneath -- timings without them are
     not a claim. -->

<!-- TODO(deepseek §13.5): media/convergence_curves.png -->

## Switching topology

<!-- TODO(deepseek §13.5): media/topology_switch.gif plus two sentences on what the split/merge event does
     to the formation, and a pointer to docs/derivations/convergence_proof.tex section 7
     for why the standard guarantee does not cover it. -->

## Repository layout

```
python/distributed_mpc_admm/   reference implementation (readable, not fast)
python/notebooks/              01-05, from ADMM intuition to convergence analysis
cpp_admm/                      OSQP + ROS 2 kernel, one process per agent
analysis/                      standalone studies: iterations, bandwidth, robustness
docs/derivations/              augmented Lagrangian, consensus ADMM, convergence proof
docs/README_math.md            notation reference -- read this before the code
media/                         generated figures and animations
```

## Quick start

### Python

```bash
cd python
pip install -e ".[dev]"
pytest -m "not slow"
```

<!-- TODO(deepseek §13.5): a 10-line runnable snippet that builds a 4-agent square formation and plots the
     result. It must run verbatim on a clean install -- copy it out of notebook 02 and
     actually execute it before committing. -->

### C++ / ROS 2

```bash
colcon build --packages-select cpp_admm
ros2 launch cpp_admm 4_agent_admm.launch.py
```

<!-- TODO(deepseek §13.5): note the ROS 2 Lyrical Luth + OSQP + Eigen prerequisites and how to get osqp_vendor if
     rosdep cannot resolve it. -->

## Method

<!-- TODO(deepseek §13.5): the algorithm block (x-update / z-update / dual update) in three lines of
     display math, then one sentence naming the only quantity that crosses the network:
     a (T x 2) trajectory block per neighbor per iteration. Link to
     docs/README_math.md and the derivations for everything else. -->

## What this does not do

<!-- TODO(deepseek §13.5): be explicit and put it above the fold-fold, not in a footnote.
     - no obstacle or inter-agent collision avoidance (the coupling is formation cost only)
     - no terminal set or terminal cost, so no recursive-feasibility or stability guarantee
       -- stability is demonstrated numerically
     - second-order integrator dynamics only; no attitude, no aerodynamics
     - the switching schedule is published centrally in the ROS demo (it stands in for
       physical link availability); the *control* is fully distributed
     A reader who finds one of these limitations themselves discounts everything else in
     the repo. A reader who is told them up front trusts the rest. -->

## Citation hook

<!-- TODO(deepseek §13.5): one paragraph. The ADMM kernel is the distributed solver inside
     `transition-viable-swarm`. The gap identified in convergence_proof.tex section 7 --
     that the guarantee assumes a fixed graph and synchronous updates -- is what motivates
     the AHTD object in the thesis proposal. Keep it factual: this repo demonstrates the
     gap and measures its consequences; it does not close it. -->

## References

<!-- TODO(deepseek §13.5): Boyd et al. 2011, Stellato et al. 2020 (OSQP), plus the switching-topology
     consensus reference. Full citations. -->

## License

BSD-3-Clause — see [LICENSE](LICENSE).
