"""Formation specifications expressed in an ADMM-compatible (separable) form.

Three formation modes are supported, all reducing to *relative-position* terms so that
agent ``i``'s local problem only ever references ``y_i^i`` and ``y_i^j`` for ``j`` in its
own neighborhood:

``rigid``
    Every edge ``(i, j)`` carries a desired offset ``d_ij = o_i - o_j``. The formation is
    defined up to translation; if the edge set is infinitesimally rigid it is also defined
    up to rotation only in the sense of the *specified* offsets (this encoding fixes
    orientation, unlike a distance-only encoding).

``leader_follower``
    A designated subset of agents carries a tracking cost to an exogenous reference;
    everyone else carries only edge terms. The leader set must be able to reach every
    other agent in the graph or the formation drifts.

``consensus``
    All ``d_ij = 0`` — pure rendezvous. Useful as the degenerate sanity check in
    ``tests/test_admm_convergence.py``.

Costs, not hard constraints
---------------------------
Formation targets enter as quadratic *costs*. Hard equality on relative positions makes
the coupled problem infeasible the moment the initial condition is inconsistent with the
actuator limits, and it destroys the strong convexity that the ADMM convergence argument
leans on. If a hard version is ever needed, add it as a slacked constraint with a large
linear penalty rather than a bare equality.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .communication_graph import CommunicationGraph

__all__ = [
    "FormationError",
    "FormationSpec",
    "LeaderFollowerSpec",
    "formation_error",
    "formation_error_trajectory",
    "interpolate_formations",
    "offsets_from_positions",
    "settling_step",
]


@dataclass
class FormationSpec:
    """Desired geometry, stored as anchor offsets from the formation centroid.

    Parameters
    ----------
    offsets:
        ``(N, dim)`` array; row ``i`` is agent ``i``'s desired position relative to the
        formation anchor. Should be mean-centred so that the anchor is the centroid.
    graph:
        The formation (edge) graph. This need not equal the *communication* graph, but
        every formation edge must be a communication edge or the corresponding cost term
        cannot be evaluated locally. :meth:`validate_against` enforces that.
    scale:
        Uniform scale applied to ``offsets``; sweeping this is how the notebooks show
        formation expansion/contraction without rebuilding the spec.
    rotation:
        Formation heading in radians, applied to ``offsets`` about the anchor.
    """

    offsets: NDArray[np.float64]
    graph: CommunicationGraph
    scale: float = 1.0
    rotation: float = 0.0
    name: str = "custom"

    # ------------------------------------------------------------------ factories

    @classmethod
    def regular_polygon(
        cls, n_agents: int, radius: float, graph: CommunicationGraph | None = None
    ) -> FormationSpec:
        """Agents evenly spaced on a circle. Defaults to a cycle communication graph."""
        raise NotImplementedError

    @classmethod
    def line(
        cls,
        n_agents: int,
        spacing: float,
        heading: float = 0.0,
        graph: CommunicationGraph | None = None,
    ) -> FormationSpec:
        """Agents in a straight line. Defaults to a path graph (the hardest topology)."""
        raise NotImplementedError

    @classmethod
    def v_shape(
        cls,
        n_agents: int,
        spacing: float,
        half_angle: float = np.pi / 6,
        graph: CommunicationGraph | None = None,
    ) -> FormationSpec:
        """Two trailing arms behind agent 0; the classic aerial formation demo."""
        raise NotImplementedError

    @classmethod
    def grid(
        cls,
        rows: int,
        cols: int,
        spacing: float,
        graph: CommunicationGraph | None = None,
    ) -> FormationSpec:
        """Rectangular lattice; defaults to the 4-neighbour lattice graph."""
        raise NotImplementedError

    @classmethod
    def rendezvous(cls, n_agents: int, graph: CommunicationGraph) -> FormationSpec:
        """All offsets zero — pure consensus."""
        raise NotImplementedError

    # ------------------------------------------------------------------ geometry

    @property
    def n_agents(self) -> int:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def anchor_offsets(self) -> NDArray[np.float64]:
        """``offsets`` after applying ``scale`` and ``rotation``; shape ``(N, dim)``."""
        raise NotImplementedError

    def relative_offset(self, i: int, j: int) -> NDArray[np.float64]:
        """``d_ij = o_i - o_j``, shape ``(dim,)``. Antisymmetric: ``d_ji == -d_ij``."""
        raise NotImplementedError

    def edge_offsets(self, agent: int) -> dict[int, NDArray[np.float64]]:
        """``{j: d_ij}`` for every formation neighbor ``j`` of ``agent``.

        This is exactly what goes into ``LocalProblemData.offsets``.
        """
        raise NotImplementedError

    def target_positions(self, anchor: NDArray[np.float64]) -> NDArray[np.float64]:
        """Absolute desired positions given an anchor point; shape ``(N, dim)``."""
        raise NotImplementedError

    # ------------------------------------------------------------------ rigidity

    def rigidity_matrix(self, positions: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """Distance rigidity matrix ``R(p)``, shape ``(|E|, N*dim)``.

        Row for edge ``(i, j)`` has ``(p_i - p_j)`` in block ``i`` and ``(p_j - p_i)`` in
        block ``j``, zeros elsewhere. Evaluated at ``positions`` (default: the nominal
        formation from :meth:`anchor_offsets`).
        """
        raise NotImplementedError

    def is_infinitesimally_rigid(
        self, positions: NDArray[np.float64] | None = None, tol: float = 1e-8
    ) -> bool:
        """``rank R(p) == dim*N - dim*(dim+1)/2`` (3 trivial motions in 2D).

        A non-rigid formation still converges under this *offset* encoding, but reporting
        rigidity is the right diagnostic when comparing against distance-based encodings.
        """
        raise NotImplementedError

    def rigidity_eigenvalue(self, positions: NDArray[np.float64] | None = None) -> float:
        """Smallest non-trivial eigenvalue of ``R^T R``; a continuous rigidity margin."""
        raise NotImplementedError

    # ------------------------------------------------------------------ validation

    def validate_against(self, comm_graph: CommunicationGraph) -> None:
        """Raise ``ValueError`` if a formation edge is missing from ``comm_graph``.

        Under a switching topology this is the check that fails first: an edge that
        disappears takes its formation cost term with it. Callers should catch it and
        decide whether to freeze the last cost or drop the term.
        """
        raise NotImplementedError


@dataclass
class LeaderFollowerSpec:
    """Which agents track an exogenous reference, and what that reference is.

    Attributes
    ----------
    leaders:
        Agent ids that carry a nonzero ``q_position`` tracking weight.
    reference:
        ``(K, dim)`` global position reference over the whole mission, or a callable
        ``(t: float) -> (dim,)``. Sliced per control step by
        :func:`per_agent_solver.build_reference_trajectory`.
    follower_position_weight:
        Tracking weight applied to non-leaders. Normally 0.0; a small positive value
        regularises the problem when the leader set cannot reach every follower.
    """

    leaders: tuple[int, ...]
    reference: NDArray[np.float64]
    follower_position_weight: float = 0.0

    def is_leader(self, agent: int) -> bool:
        raise NotImplementedError

    def weight_for(self, agent: int, base_weight: float) -> float:
        """``base_weight`` for leaders, ``follower_position_weight`` otherwise."""
        raise NotImplementedError

    def validate_against(self, graph: CommunicationGraph) -> None:
        """Raise if the leader set cannot reach every agent over ``graph``.

        Unreachable followers have no term anchoring their absolute position, so the
        formation converges in *shape* but drifts as a rigid body — a real failure mode
        worth asserting on rather than debugging in a plot.
        """
        raise NotImplementedError


@dataclass
class FormationError:
    """Decomposition of formation error into the parts that matter separately."""

    edge_rms: float
    """RMS over formation edges of ``|| (p_i - p_j) - d_ij ||``."""

    edge_max: float
    """Worst single edge error."""

    centroid_error: float
    """``|| mean(p) - anchor_reference ||``; nonzero means rigid-body drift."""

    per_edge: dict[tuple[int, int], float] = field(default_factory=dict)


def formation_error(
    positions: NDArray[np.float64],
    spec: FormationSpec,
    anchor_reference: NDArray[np.float64] | None = None,
) -> FormationError:
    """Evaluate formation error for a single time instant.

    ``positions`` is ``(N, dim)``. Reported in every closed-loop figure; the edge RMS is
    the headline number quoted in the README.
    """
    raise NotImplementedError


def formation_error_trajectory(
    position_log: NDArray[np.float64],
    spec: FormationSpec,
    anchor_reference: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Vectorised :func:`formation_error` over time. ``position_log`` is ``(K, N, dim)``;
    returns ``(K,)`` of edge-RMS values."""
    raise NotImplementedError


def settling_step(
    errors: NDArray[np.float64], tolerance: float, hold: int = 5
) -> int | None:
    """First index after which ``errors`` stays below ``tolerance`` for ``hold`` steps.

    Returns ``None`` if the formation never settles. This is the metric plotted against
    algebraic connectivity in ``analysis/topology_robustness.ipynb``.
    """
    raise NotImplementedError


def offsets_from_positions(
    positions: NDArray[np.float64], graph: CommunicationGraph
) -> FormationSpec:
    """Build a :class:`FormationSpec` that holds an observed configuration.

    Used to freeze the current shape at a topology-switch instant, which is how a
    "merge" event is turned into a well-posed formation target.
    """
    raise NotImplementedError


def interpolate_formations(
    start: FormationSpec, end: FormationSpec, alpha: float
) -> FormationSpec:
    """Convex blend of two formations, ``alpha`` in ``[0, 1]``.

    A morph event is a time-parametrised sweep of ``alpha``; keeping it here (rather than
    inside the ADMM loop) keeps the solver oblivious to the event structure.
    """
    raise NotImplementedError
