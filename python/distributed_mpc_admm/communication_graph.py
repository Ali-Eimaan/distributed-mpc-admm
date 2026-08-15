# Copyright (c) 2026, Ali-Eimaan. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Interaction topologies for consensus ADMM: static, time-varying, and lossy.

The ADMM iteration needs three things from this module:

1. ``neighbors(i)`` / ``closed_neighborhood(i)`` — which local copies agent ``i`` carries.
2. ``laplacian()`` / ``algebraic_connectivity()`` — the spectral quantities that appear in
   the convergence rate bound (see ``docs/derivations/convergence_proof.tex``).
3. A channel model that can *drop* and *delay* messages, so that the synchronous-update
   assumption underlying Boyd's proof can be violated on purpose and measured.

All graphs here are undirected and simple (no self-loops in ``adjacency``); the self-term
is added explicitly by :meth:`CommunicationGraph.closed_neighborhood`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

__all__ = [
    "ChannelStats",
    "CommunicationGraph",
    "LossyChannel",
    "Message",
    "TimeVaryingGraph",
]


def _resolve_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Normalise the various random-source spellings into a ``Generator``."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, int):
        return np.random.default_rng(rng)
    return rng


class CommunicationGraph:
    """An undirected, simple communication topology over ``n_agents`` agents.

    Parameters
    ----------
    n_agents:
        Number of agents ``N``. Agent ids are ``0 .. N-1``.
    edges:
        Iterable of ``(i, j)`` pairs. Order within a pair is irrelevant; duplicates and
        reversed duplicates are collapsed. Self-loops raise ``ValueError``.
    weights:
        Optional edge weights ``{(i, j): w}``. Defaults to 1.0 for every edge. Weights are
        used by :meth:`laplacian` and by the weighted z-update variant.
    """

    def __init__(
        self,
        n_agents: int,
        edges: Iterable[tuple[int, int]] = (),
        weights: dict[tuple[int, int], float] | None = None,
    ) -> None:
        if n_agents < 1:
            raise ValueError(f"n_agents must be at least 1, got {n_agents}")
        self._n = int(n_agents)
        self._edges: frozenset[tuple[int, int]] = frozenset()
        self._weights: dict[tuple[int, int], float] = {}
        self._adj_cache: NDArray[np.float64] | None = None
        self._lap_cache: dict[bool, NDArray[np.float64]] = {}

        for i, j in edges:
            key = self._canonical(i, j)
            if key not in self._weights:
                self._edges = self._edges | {key}
                self._weights[key] = 1.0

        if weights is not None:
            for (i, j), w in weights.items():
                key = self._canonical(i, j)
                if key in self._weights:
                    self._weights[key] = float(w)

    def _canonical(self, i: int, j: int) -> tuple[int, int]:
        """Normalise and validate an edge, returning the canonical ``(min, max)`` key."""
        i = int(i)
        j = int(j)
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise ValueError(f"agent id out of range [0, {self._n}): ({i}, {j})")
        if i == j:
            raise ValueError(f"self-loops are not allowed: ({i}, {j})")
        return (min(i, j), max(i, j))

    def _invalidate(self) -> None:
        """Drop the cached adjacency and Laplacian matrices."""
        self._adj_cache = None
        self._lap_cache = {}

    # ------------------------------------------------------------------ factories

    @classmethod
    def from_adjacency(cls, adjacency: NDArray[np.floating]) -> CommunicationGraph:
        """Build a graph from a symmetric 0/1 (or weighted) adjacency matrix."""
        A = np.asarray(adjacency, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"adjacency must be square, got shape {A.shape}")
        if not np.allclose(A, A.T, atol=1e-12):
            raise ValueError("adjacency matrix must be symmetric")
        if np.any(np.abs(np.diag(A)) > 1e-12):
            raise ValueError("adjacency matrix must have a zero diagonal")
        n = A.shape[0]
        edges: list[tuple[int, int]] = []
        weights: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] != 0.0:
                    edges.append((i, j))
                    weights[(i, j)] = float(A[i, j])
        return cls(n, edges, weights)

    @classmethod
    def complete(cls, n_agents: int) -> CommunicationGraph:
        """All-to-all topology. Fastest consensus, worst communication load."""
        edges = [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]
        return cls(n_agents, edges)

    @classmethod
    def cycle(cls, n_agents: int) -> CommunicationGraph:
        """Ring topology ``0-1-...-(N-1)-0``."""
        if n_agents == 1:
            return cls(1)
        edges = [(i, (i + 1) % n_agents) for i in range(n_agents)]
        return cls(n_agents, edges)

    @classmethod
    def path(cls, n_agents: int) -> CommunicationGraph:
        """Chain topology. Worst-case algebraic connectivity for a connected graph."""
        edges = [(i, i + 1) for i in range(n_agents - 1)]
        return cls(n_agents, edges)

    @classmethod
    def star(cls, n_agents: int, center: int = 0) -> CommunicationGraph:
        """Star topology; the natural leader-follower communication pattern."""
        if center < 0 or center >= n_agents:
            raise ValueError(f"center {center} out of range [0, {n_agents})")
        edges = [(center, i) for i in range(n_agents) if i != center]
        return cls(n_agents, edges)

    @classmethod
    def random_connected(
        cls,
        n_agents: int,
        edge_prob: float = 0.4,
        rng: np.random.Generator | int | None = None,
        max_tries: int = 100,
    ) -> CommunicationGraph:
        """Erdos-Renyi graph, resampled until connected (or ``RuntimeError``)."""
        rng = _resolve_rng(rng)
        for _ in range(max_tries):
            edges = [
                (i, j)
                for i in range(n_agents)
                for j in range(i + 1, n_agents)
                if rng.random() < edge_prob
            ]
            graph = cls(n_agents, edges)
            if graph.is_connected():
                return graph
        raise RuntimeError(f"failed to sample a connected graph after {max_tries} tries")

    # ------------------------------------------------------------------ properties

    @property
    def n_agents(self) -> int:
        """Number of agents ``N``."""
        return self._n

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Canonical ``(i, j)`` pairs with ``i < j``, sorted."""
        return tuple(sorted(self._edges))

    @property
    def n_edges(self) -> int:
        """``|E|`` — used by :func:`communication_load` in the analysis notebooks."""
        return len(self._edges)

    @property
    def adjacency(self) -> NDArray[np.float64]:
        """Symmetric ``(N, N)`` weighted adjacency matrix with zero diagonal."""
        if self._adj_cache is None:
            A = np.zeros((self._n, self._n), dtype=np.float64)
            for (i, j), w in self._weights.items():
                A[i, j] = w
                A[j, i] = w
            self._adj_cache = A
        return self._adj_cache

    # ------------------------------------------------------------------ queries

    def neighbors(self, agent: int) -> tuple[int, ...]:
        """Open neighborhood ``N_i`` (sorted, excludes ``agent``)."""
        if agent < 0 or agent >= self._n:
            raise ValueError(f"agent id out of range [0, {self._n}): {agent}")
        return tuple(sorted(j for j in range(self._n) if self.has_edge(agent, j)))

    def closed_neighborhood(self, agent: int) -> tuple[int, ...]:
        """``N_i union {i}``, sorted.

        This is exactly the set of local copies ``y_i^j`` that agent ``i`` maintains,
        so the ordering here defines the ordering of the per-agent decision vector.
        """
        return tuple(sorted(self.neighbors(agent) + (agent,)))

    def degree(self, agent: int) -> int:
        """``|N_i|``."""
        return len(self.neighbors(agent))

    def contributors(self, agent: int) -> tuple[int, ...]:
        """Agents that hold a local copy of ``agent``'s trajectory.

        For an undirected graph this equals ``closed_neighborhood(agent)``; it is a
        separate method because the z-update averages over *this* set and the two
        coincide only in the undirected case.
        """
        return self.closed_neighborhood(agent)

    def has_edge(self, i: int, j: int) -> bool:
        return (min(int(i), int(j)), max(int(i), int(j))) in self._edges

    # ------------------------------------------------------------------ spectral

    def degree_matrix(self) -> NDArray[np.float64]:
        """Diagonal ``(N, N)`` matrix of weighted degrees."""
        return np.diag(self.adjacency.sum(axis=1))

    def laplacian(self, normalized: bool = False) -> NDArray[np.float64]:
        """Graph Laplacian ``L = D - A`` (or ``I - D^{-1/2} A D^{-1/2}`` if normalized)."""
        if normalized in self._lap_cache:
            return self._lap_cache[normalized]
        A = self.adjacency
        if not normalized:
            L = np.diag(A.sum(axis=1)) - A
        else:
            deg = A.sum(axis=1)
            d_inv_sqrt = np.zeros(self._n, dtype=np.float64)
            mask = deg > 0
            d_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
            L = np.eye(self._n) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
            L[~mask, :] = 0.0
        self._lap_cache[normalized] = L
        return L

    def algebraic_connectivity(self) -> float:
        """Fiedler value ``lambda_2(L)``.

        Zero iff the graph is disconnected. Larger values give faster consensus; report
        this alongside iterations-to-consensus in ``analysis/topology_robustness.ipynb``.
        """
        eigvals = scipy.linalg.eigh(self.laplacian(), eigvals_only=True)
        return max(0.0, float(eigvals[1]))

    def spectral_gap_ratio(self) -> float:
        """``lambda_2(L) / lambda_max(L)`` — the condition-number-like quantity that
        bounds the linear convergence rate of the consensus averaging step."""
        eigvals = scipy.linalg.eigh(self.laplacian(), eigvals_only=True)
        lam2 = max(0.0, float(eigvals[1]))
        lam_max = float(eigvals[-1])
        if lam_max <= 0.0:
            return 0.0
        return lam2 / lam_max

    def is_connected(self) -> bool:
        return self.algebraic_connectivity() > 1e-10

    # ------------------------------------------------------------------ mutation

    def add_edge(self, i: int, j: int, weight: float = 1.0) -> None:
        """Insert an edge in place; no-op if already present (weight is overwritten)."""
        key = self._canonical(i, j)
        self._edges = self._edges | {key}
        self._weights[key] = float(weight)
        self._invalidate()

    def remove_edge(self, i: int, j: int) -> None:
        """Delete an edge in place; no-op if absent."""
        key = self._canonical(i, j)
        self._edges = self._edges - {key}
        self._weights.pop(key, None)
        self._invalidate()

    def copy(self) -> CommunicationGraph:
        return CommunicationGraph(self._n, self._edges, dict(self._weights))

    # ------------------------------------------------------------------ interop

    def to_networkx(self) -> Any:
        """Return a ``networkx.Graph`` view (used only for layout in plotting)."""
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(self._n))
        for (i, j), w in self._weights.items():
            graph.add_edge(i, j, weight=w)
        return graph

    def __repr__(self) -> str:
        return (
            f"CommunicationGraph(n_agents={self._n}, n_edges={self.n_edges}, "
            f"connected={self.is_connected()})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CommunicationGraph):
            return NotImplemented
        return (self._n, self._edges, self._weights) == (
            other._n,
            other._edges,
            other._weights,
        )


class TimeVaryingGraph:
    """A schedule ``k -> CommunicationGraph`` for switching topologies.

    Two construction modes:

    * **Sequence mode** — a finite list of graphs, indexed by ``k`` and held/cycled past
      the end according to ``mode``.
    * **Callable mode** — an arbitrary ``Callable[[int], CommunicationGraph]``, which is
      what the split/merge event experiments use.
    """

    def __init__(
        self,
        schedule: Sequence[CommunicationGraph] | Callable[[int], CommunicationGraph],
        mode: str = "hold",
    ) -> None:
        """``mode`` is one of ``"hold"`` (repeat last) or ``"cycle"`` (wrap around)."""
        self._cache: dict[int, CommunicationGraph] = {}
        if callable(schedule):
            self._fn = schedule
            return

        graphs = tuple(schedule)
        if not graphs:
            raise ValueError("schedule must contain at least one graph")
        n_agents = graphs[0].n_agents
        for graph in graphs:
            if graph.n_agents != n_agents:
                raise ValueError("all graphs in a schedule must have the same n_agents")

        if mode == "hold":
            self._fn = lambda k: graphs[min(k, len(graphs) - 1)]
        elif mode == "cycle":
            self._fn = lambda k: graphs[k % len(graphs)]
        else:
            raise ValueError(f"mode must be 'hold' or 'cycle', got {mode!r}")

    @classmethod
    def switching(
        cls,
        graphs: Sequence[CommunicationGraph],
        dwell_time: int,
        mode: str = "cycle",
    ) -> TimeVaryingGraph:
        """Hold each graph for ``dwell_time`` control steps before switching.

        ``dwell_time`` is the object that shows up as the average-dwell-time condition in
        switched-systems stability arguments; expose it so the notebooks can sweep it.
        """
        graphs = tuple(graphs)
        if not graphs:
            raise ValueError("graphs must not be empty")
        if dwell_time <= 0:
            raise ValueError(f"dwell_time must be positive, got {dwell_time}")
        n_agents = graphs[0].n_agents
        for graph in graphs:
            if graph.n_agents != n_agents:
                raise ValueError("all graphs must have the same n_agents")
        if mode not in ("hold", "cycle"):
            raise ValueError(f"mode must be 'hold' or 'cycle', got {mode!r}")

        def fn(k: int) -> CommunicationGraph:
            idx = k // dwell_time
            if mode == "hold":
                return graphs[min(idx, len(graphs) - 1)]
            return graphs[idx % len(graphs)]

        return cls(fn)

    def at(self, k: int) -> CommunicationGraph:
        """Topology active at control step ``k``."""
        if k not in self._cache:
            self._cache[k] = self._fn(k)
        return self._cache[k]

    def union_over(self, k_start: int, k_end: int) -> CommunicationGraph:
        """Union of the topologies active over the half-open window ``[k_start, k_end)``."""
        if k_end < k_start:
            raise ValueError("k_end must be >= k_start")
        if k_end == k_start:
            return CommunicationGraph(self.at(k_start).n_agents)

        edges: set[tuple[int, int]] = set()
        weights: dict[tuple[int, int], float] = {}
        for k in range(k_start, k_end):
            graph = self.at(k)
            for edge in graph.edges:
                if edge not in edges:
                    edges.add(edge)
                    weights[edge] = graph._weights.get(edge, 1.0)
        return CommunicationGraph(self.at(k_start).n_agents, edges, weights)

    def is_jointly_connected(self, k_start: int, k_end: int) -> bool:
        """Whether the union graph over ``[k_start, k_end)`` is connected.

        Joint connectivity over bounded windows is the standard weakening of
        "connected at every instant" that switching-topology consensus proofs need.
        """
        return self.union_over(k_start, k_end).is_connected()

    def switch_steps(self, k_start: int, k_end: int) -> tuple[int, ...]:
        """Control steps in ``[k_start, k_end)`` at which the topology changes."""
        steps = []
        for k in range(k_start + 1, k_end):
            if self.at(k) != self.at(k - 1):
                steps.append(k)
        return tuple(steps)


@dataclass(frozen=True)
class Message:
    """One inter-agent packet.

    Attributes
    ----------
    sender, receiver:
        Agent ids.
    subject:
        The agent whose trajectory this message is *about*. In the y-exchange phase
        ``subject`` is the neighbor whose copy is being reported; in the z-broadcast phase
        ``subject == sender``.
    admm_iteration:
        ADMM iteration index at which the payload was produced. Receivers use this to
        detect and quantify staleness.
    payload:
        ``(T, 2)`` trajectory block.
    """

    sender: int
    receiver: int
    subject: int
    admm_iteration: int
    payload: NDArray[np.float64]


@dataclass
class ChannelStats:
    """Bookkeeping for a :class:`LossyChannel`, reported in the communication-load study."""

    sent: int = 0
    delivered: int = 0
    dropped: int = 0
    bytes_sent: int = 0
    staleness_histogram: dict[int, int] = field(default_factory=dict)

    @property
    def loss_rate(self) -> float:
        """Empirical ``dropped / sent``."""
        return self.dropped / self.sent if self.sent > 0 else 0.0

    @property
    def mean_staleness(self) -> float:
        """Mean number of ADMM iterations by which received payloads lagged."""
        if not self.staleness_histogram:
            return 0.0
        total = sum(k * v for k, v in self.staleness_histogram.items())
        count = sum(self.staleness_histogram.values())
        return total / count


class LossyChannel:
    """Bernoulli packet loss + bounded integer delay on top of a graph.

    Semantics
    ---------
    * ``send`` enqueues a message with an arrival iteration drawn from the delay model.
    * ``receive`` returns the *freshest already-arrived* payload for ``(receiver, subject)``,
      or ``None`` if nothing has ever arrived. Callers must decide what to do with ``None``
      — the reference ADMM loop falls back to the last known value, which is precisely the
      asynchronous behaviour the thesis characterises.
    * Losses are drawn per (message, edge), independently, from ``self._rng``.
    """

    def __init__(
        self,
        graph: CommunicationGraph,
        loss_prob: float = 0.0,
        max_delay: int = 0,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        self._graph = graph
        self._loss_prob = float(loss_prob)
        self._max_delay = int(max_delay)
        self._seed = rng if isinstance(rng, int) else None
        self._rng = _resolve_rng(rng)
        self._mailbox: dict[tuple[int, int, int], Message] = {}
        self._inflight: list[tuple[int, Message]] = []
        self._stats = ChannelStats()

    def send(self, message: Message, iteration: int) -> bool:
        """Attempt delivery. Returns ``False`` if the packet was dropped."""
        self._stats.sent += 1
        self._stats.bytes_sent += int(message.payload.nbytes)
        if self._rng.random() < self._loss_prob:
            self._stats.dropped += 1
            return False
        delay = int(self._rng.integers(0, self._max_delay + 1))
        self._inflight.append((iteration + delay, message))
        return True

    def broadcast(
        self,
        sender: int,
        subject: int,
        payload: NDArray[np.float64],
        iteration: int,
        receivers: Iterable[int] | None = None,
    ) -> int:
        """Send the same payload to every receiver (default: ``graph.neighbors(sender)``).

        Returns the number of packets that were *not* dropped.
        """
        if receivers is None:
            receivers = self._graph.neighbors(sender)
        delivered = 0
        for receiver in receivers:
            message = Message(
                sender=sender,
                receiver=receiver,
                subject=subject,
                admm_iteration=iteration,
                payload=payload,
            )
            if self.send(message, iteration):
                delivered += 1
        return delivered

    def receive(self, receiver: int, subject: int, sender: int, iteration: int) -> Message | None:
        """Freshest message about ``subject`` from ``sender`` available to ``receiver``.

        The ``(receiver, subject, sender)`` key keeps the two per-iteration streams
        distinct: contribution messages ``i -> j`` (about ``j``) and consensus broadcasts
        ``j -> i`` (about ``j``) both flow on edge ``(i, j)`` and would otherwise collide
        on the ``(receiver, subject)`` pair.
        """
        message = self._mailbox.get((receiver, subject, sender))
        if message is None:
            return None
        staleness = iteration - message.admm_iteration
        self._stats.staleness_histogram[staleness] = (
            self._stats.staleness_histogram.get(staleness, 0) + 1
        )
        return message

    def advance(self, iteration: int) -> None:
        """Move all in-flight packets whose arrival time has come into the mailboxes."""
        arriving = [entry for entry in self._inflight if entry[0] <= iteration]
        self._inflight = [entry for entry in self._inflight if entry[0] > iteration]
        for _, message in arriving:
            self._stats.delivered += 1
            key = (message.receiver, message.subject, message.sender)
            existing = self._mailbox.get(key)
            if existing is None or message.admm_iteration > existing.admm_iteration:
                self._mailbox[key] = message

    def clear_messages(self) -> None:
        """Drop delivered and in-flight messages, keeping the RNG stream and stats.

        A fresh ADMM solve restarts its iteration counter at zero, so messages from a
        previous solve carry larger ``admm_iteration`` values and would shadow the fresh
        ones forever. Clearing the message buffers between solves (while *keeping* the
        accumulated ``stats`` for the load study, and *not* reseeding the RNG so each
        control step sees a different loss/delay draw) restores the intended per-solve
        semantics.
        """
        self._mailbox.clear()
        self._inflight.clear()

    def set_graph(self, graph: CommunicationGraph) -> None:
        """Swap the topology mid-run (used by the time-varying experiments).

        Mailboxes are *not* cleared: an agent keeps the last value it heard from a node
        that has since disconnected, which is the realistic behaviour.
        """
        self._graph = graph

    def reset(self) -> None:
        """Clear mailboxes, in-flight queue and statistics; reseed if a seed was given."""
        self._mailbox.clear()
        self._inflight.clear()
        self._stats = ChannelStats()
        if self._seed is not None:
            self._rng = np.random.default_rng(self._seed)

    @property
    def stats(self) -> ChannelStats:
        return self._stats


def communication_load(
    graph: CommunicationGraph,
    horizon: int,
    admm_iterations: int,
    float_bytes: int = 8,
) -> dict[str, float]:
    """Bytes and packet counts for one full ADMM solve on ``graph``.

    Returns a dict with keys ``packets``, ``bytes``, ``bytes_per_agent``,
    ``packets_per_iteration``. Used directly by
    ``analysis/communication_load_study.ipynb`` to plot load versus ``N`` and versus
    topology density.
    """
    dim = 2  # the repository is fixed to 2D double-integrator agents
    packets_per_iteration = 4 * graph.n_edges
    packets = packets_per_iteration * admm_iterations
    bytes_total = packets * horizon * dim * float_bytes
    return {
        "packets": float(packets),
        "bytes": float(bytes_total),
        "bytes_per_agent": float(bytes_total) / graph.n_agents,
        "packets_per_iteration": float(packets_per_iteration),
    }
