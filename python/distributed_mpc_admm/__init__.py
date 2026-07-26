"""Distributed MPC via consensus ADMM for double-integrator agents in 2D.

This package is the *pedagogy-first* reference implementation. It is deliberately
readable rather than fast; the production kernel lives in ``cpp_admm/`` (OSQP +
ROS 2 / ZeroMQ transport).

Module map
----------
``communication_graph``
    Static and time-varying interaction topologies, plus a lossy/delayed channel
    model used to break the synchronous-update assumption on purpose.
``per_agent_solver``
    Double-integrator prediction model and the per-agent local QP (CVXPY backend).
``formation_constraints``
    Rigid / leader-follower / relative-position formation specifications, expressed
    so that they enter the ADMM local problems as separable costs.
``consensus_admm``
    The general-form consensus ADMM loop (x-update, z-update, dual update,
    residuals, adaptive rho) and the closed-loop ``DistributedMPC`` driver.
``plotting``
    Trajectory / convergence / topology figures and the animation helpers that
    produce the README media.

Notation used throughout (matches ``docs/derivations/consensus_admm_derivation.tex``)
-------------------------------------------------------------------------------
``N``       number of agents
``T``       MPC prediction horizon (number of steps)
``n``       state dimension per agent (4: px, py, vx, vy)
``m``       input dimension per agent (2: ax, ay)
``y_i^j``   agent ``i``'s *local copy* of agent ``j``'s predicted position trajectory,
            shape ``(T, 2)``, defined for ``j`` in the closed neighborhood of ``i``
``z^j``     the consensus (agreed) predicted position trajectory of agent ``j``
``lam_i^j`` scaled dual variable associated with ``y_i^j - z^j = 0``
``rho``     augmented-Lagrangian penalty parameter
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "ADMMHistory",
    "ADMMOptions",
    "ADMMResult",
    "AgentCostWeights",
    "AgentLimits",
    "CommunicationGraph",
    "ConsensusADMM",
    "CvxpyAgentSolver",
    "DistributedMPC",
    "DoubleIntegrator",
    "FormationSpec",
    "LeaderFollowerSpec",
    "LocalProblemData",
    "LocalSolution",
    "LossyChannel",
    "PerAgentSolver",
    "SimulationLog",
    "TimeVaryingGraph",
    "__version__",
]

# TODO [GUIDE 2.1]: re-export the public names above once the modules are implemented, e.g.
#     from .communication_graph import CommunicationGraph, LossyChannel, TimeVaryingGraph
#     from .consensus_admm import ADMMHistory, ADMMOptions, ADMMResult, ConsensusADMM, ...
#     from .formation_constraints import FormationSpec, LeaderFollowerSpec
#     from .per_agent_solver import AgentCostWeights, AgentLimits, CvxpyAgentSolver, ...
# Keep the imports lazy-free (plain top-level imports); the package must stay importable
# without a solver installed only if you guard the CVXPY import inside per_agent_solver.
