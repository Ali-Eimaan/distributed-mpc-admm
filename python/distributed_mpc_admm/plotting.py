# Copyright (c) 2026, Ali-Eimaan. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Figures and animations for the README, notebooks, and analysis studies.

Every function takes an optional ``ax`` and returns the ``Axes`` (or ``Figure``) it drew
on, so notebooks can compose subplots. Nothing here calls ``plt.show()`` or writes files
except the explicit ``save_*`` helpers.

Deliverables this module has to produce (see ``media/``):

* ``4_agent_formation.gif``  -> :func:`animate_formation`
* ``topology_switch.gif``    -> :func:`animate_formation` with ``show_graph=True``
* ``convergence_curves.png`` -> :func:`plot_convergence`
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .communication_graph import CommunicationGraph
from .consensus_admm import ADMMHistory, SimulationLog
from .formation_constraints import FormationSpec

__all__ = [
    "AGENT_COLORS",
    "animate_formation",
    "apply_style",
    "make_readme_media",
    "plot_communication_load",
    "plot_convergence",
    "plot_formation_error",
    "plot_graph",
    "plot_inputs",
    "plot_iterations_to_consensus",
    "plot_rho_sweep",
    "plot_states",
    "plot_topology_timeline",
    "plot_trajectories",
    "save_animation",
]

AGENT_COLORS: tuple[str, ...] = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
)
"""Fixed per-agent colours. Keep the mapping stable across every figure in the repo —
a reader should be able to identify agent 2 by colour alone in any plot."""


def apply_style(context: str = "paper") -> None:
    """Set rcParams for a consistent look.

    ``context`` is ``"paper"`` (serif, small, for the LaTeX docs), ``"notebook"``
    (default sizes) or ``"readme"`` (larger fonts, transparent background so the figures
    read on both GitHub light and dark themes — this matters, dark-mode readers see a
    black box otherwise).
    """
    import matplotlib as mpl

    if context == "paper":
        mpl.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 9,
                "axes.titlesize": 9,
                "axes.labelsize": 9,
                "axes.linewidth": 0.6,
                "lines.linewidth": 1.0,
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
            }
        )
    elif context == "notebook":
        mpl.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 11,
                "axes.titlesize": 11,
                "axes.labelsize": 11,
                "figure.dpi": 100,
                "savefig.dpi": 150,
                "savefig.bbox": "tight",
            }
        )
    elif context == "readme":
        mpl.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 13,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "figure.dpi": 100,
                "savefig.dpi": 110,
                "savefig.bbox": "tight",
                "savefig.transparent": True,
                "figure.facecolor": "none",
                "axes.facecolor": "none",
            }
        )
    else:
        raise ValueError(f"unknown style context {context!r}")


# ---------------------------------------------------------------------- trajectories


def plot_trajectories(
    log: SimulationLog,
    ax: Any = None,
    show_start: bool = True,
    show_final_formation: bool = True,
    formation: FormationSpec | None = None,
    every: int = 1,
) -> Any:
    """XY plot of all agent paths, one colour per agent.

    Markers: circle at the start, filled dot at the end, dashed grey lines connecting the
    final formation edges.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    positions = log.positions
    n_agents = positions.shape[1]
    for i in range(n_agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax.plot(
            positions[::every, i, 0],
            positions[::every, i, 1],
            color=color,
            linewidth=1.2,
            label=f"agent {i}",
        )
        if show_start:
            ax.scatter(
                positions[0, i, 0],
                positions[0, i, 1],
                marker="o",
                facecolors="none",
                edgecolors=color,
                s=60,
                zorder=3,
            )
        ax.scatter(
            positions[-1, i, 0],
            positions[-1, i, 1],
            marker="o",
            color=color,
            s=40,
            zorder=3,
        )

    if show_final_formation and formation is not None:
        for i, j in formation.graph.edges:
            ax.plot(
                [positions[-1, i, 0], positions[-1, j, 0]],
                [positions[-1, i, 1], positions[-1, j, 1]],
                color="grey",
                linewidth=0.7,
                linestyle="--",
                alpha=0.6,
            )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize="small")
    ax.set_title("Agent trajectories")
    return ax


def plot_inputs(log: SimulationLog, axes: Any = None, u_max: float | None = None) -> Any:
    """Per-agent acceleration components versus time, with the saturation band shaded.

    Whether the limits are ever active is the first question about any MPC result; make it
    visible rather than asserting it in text.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    time = log.time
    inputs = log.inputs
    n_agents = inputs.shape[1]
    for comp, axis in zip((0, 1), axes, strict=True):
        for i in range(n_agents):
            axis.plot(
                time,
                inputs[:, i, comp],
                color=AGENT_COLORS[i % len(AGENT_COLORS)],
                linewidth=1.0,
                label=f"agent {i}",
            )
        if u_max is not None:
            axis.axhspan(-u_max, u_max, color="grey", alpha=0.12, zorder=0)
            axis.set_ylim(
                min(-u_max * 1.3, inputs[:, :, comp].min()),
                max(u_max * 1.3, inputs[:, :, comp].max()),
            )
        axis.set_ylabel(f"u{['x','y'][comp]} [m/s²]")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[0].set_title("Control inputs")
    axes[0].legend(loc="best", fontsize="x-small", ncol=2)
    return axes


def plot_states(log: SimulationLog, axes: Any = None) -> Any:
    """Position and velocity components versus time in a 2x2 grid."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)

    time = log.time
    states = log.states
    n_agents = states.shape[1]
    names = ["px [m]", "py [m]", "vx [m/s]", "vy [m/s]"]
    for comp, (axis, name) in enumerate(zip(axes.ravel(), names, strict=True)):
        for i in range(n_agents):
            axis.plot(
                np.concatenate([[0.0], time]),
                states[:, i, comp],
                color=AGENT_COLORS[i % len(AGENT_COLORS)],
                linewidth=1.0,
            )
        axis.set_ylabel(name)
        axis.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 1].set_xlabel("time [s]")
    axes[0, 0].set_title("States")
    return axes


# ---------------------------------------------------------------------- convergence


def plot_convergence(
    histories: ADMMHistory | Sequence[ADMMHistory],
    ax: Any = None,
    labels: Sequence[str] | None = None,
    show_tolerances: bool = True,
    show_rho: bool = False,
) -> Any:
    """Semilog-y primal and dual residuals versus ADMM iteration.

    Primal solid, dual dashed, tolerance thresholds dotted in grey. When ``show_rho`` is
    set, add a twin axis with the rho trace — required whenever adaptive rho is on, since
    the residual kinks are otherwise unexplained.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if isinstance(histories, ADMMHistory):
        histories = [histories]

    if labels is None:
        labels = [f"run {k}" for k in range(len(histories))]

    for history, label in zip(histories, labels, strict=False):
        primal = np.asarray(history.primal_residual, dtype=float)
        dual = np.asarray(history.dual_residual, dtype=float)
        iterations = np.arange(len(primal))
        (line,) = ax.semilogy(iterations, primal, label=f"{label} (primal)", linewidth=1.2)
        ax.semilogy(iterations, dual, linestyle="--", color=line.get_color(), linewidth=1.0)

    if show_tolerances and histories:
        eps = np.asarray(histories[0].eps_primal, dtype=float)
        if eps.size:
            ax.semilogy(
                np.arange(len(eps)),
                eps,
                linestyle=":",
                color="grey",
                linewidth=1.0,
                label="tolerance",
            )

    if show_rho and histories:
        ax2 = ax.twinx()
        rho = np.asarray(histories[0].rho, dtype=float)
        ax2.plot(np.arange(len(rho)), rho, color="tab:purple", alpha=0.6, linewidth=1.0)
        ax2.set_ylabel("rho", color="tab:purple")
        ax2.tick_params(axis="y", labelcolor="tab:purple")

    ax.set_xlabel("ADMM iteration")
    ax.set_ylabel("residual (log scale)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    ax.set_title("ADMM convergence")
    return ax


def plot_iterations_to_consensus(
    x_values: NDArray[np.float64],
    iterations: NDArray[np.float64],
    ax: Any = None,
    xlabel: str = "",
    show_median: bool = True,
) -> Any:
    """Iteration count versus a swept parameter, with per-trial scatter plus median line.

    Used for the ``N``, ``rho``, horizon, and ``lambda_2`` sweeps. Always plot the
    individual trials: the spread across random initial conditions is large and a bare
    mean curve overstates how clean the result is.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    x_values = np.asarray(x_values, dtype=float)
    iterations = np.asarray(iterations, dtype=float)

    if iterations.ndim == 1:
        iterations = iterations[:, None]

    for trial in range(iterations.shape[1]):
        ax.scatter(
            x_values,
            iterations[:, trial],
            color="grey",
            alpha=0.5,
            s=20,
            zorder=1,
            label="trial" if trial == 0 else None,
        )

    if show_median and iterations.shape[1] > 0:
        median = np.median(iterations, axis=1)
        ax.plot(x_values, median, color="black", linewidth=1.6, label="median", zorder=2)

    ax.set_xlabel(xlabel or "sweep parameter")
    ax.set_ylabel("ADMM iterations")
    ax.grid(True, alpha=0.3)
    if iterations.shape[1] > 0:
        ax.legend(loc="best", fontsize="small")
    ax.set_title("Iterations to consensus")
    return ax


def plot_rho_sweep(
    rho_values: NDArray[np.float64],
    iterations: NDArray[np.float64],
    ax: Any = None,
    mark_optimum: bool = True,
) -> Any:
    """Log-x U-curve of iterations versus rho, annotating the empirical minimum."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    rho_values = np.asarray(rho_values, dtype=float)
    iterations = np.asarray(iterations, dtype=float)
    if iterations.ndim == 1:
        iterations = iterations[:, None]

    for trial in range(iterations.shape[1]):
        ax.plot(
            rho_values,
            iterations[:, trial],
            color="grey",
            alpha=0.5,
            linewidth=1.0,
            marker="o",
            markersize=3,
        )

    median = np.median(iterations, axis=1)
    ax.plot(rho_values, median, color="black", linewidth=1.6, label="median")
    ax.set_xscale("log")

    if mark_optimum:
        best = int(np.argmin(median))
        ax.scatter(
            rho_values[best],
            median[best],
            color="tab:red",
            marker="*",
            s=160,
            zorder=3,
            label=f"optimum rho={rho_values[best]:.3g}",
        )

    ax.set_xlabel("rho (log scale)")
    ax.set_ylabel("ADMM iterations")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    ax.set_title("Rho sweep")
    return ax


def plot_formation_error(
    log: SimulationLog | Sequence[SimulationLog],
    ax: Any = None,
    labels: Sequence[str] | None = None,
    tolerance: float | None = None,
    mark_switches: bool = True,
) -> Any:
    """Edge-RMS formation error versus time, with vertical lines at topology switches."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    logs = [log] if isinstance(log, SimulationLog) else list(log)

    if labels is None:
        labels = [f"run {k}" for k in range(len(logs))]

    for entry, label in zip(logs, labels, strict=False):
        time = entry.time
        ax.plot(time, entry.formation_error, linewidth=1.3, label=label)

        if mark_switches and "switch_steps" in entry.metadata:
            for step in cast(Sequence[int], entry.metadata["switch_steps"]):
                if step < len(time):
                    ax.axvline(time[int(step)], color="grey", linestyle="--", alpha=0.5)

    if tolerance is not None:
        ax.axhline(tolerance, color="tab:red", linestyle=":", linewidth=1.0, label="tolerance")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("formation error (edge RMS) [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    ax.set_title("Formation error")
    return ax


# ---------------------------------------------------------------------- graphs


def plot_graph(
    graph: CommunicationGraph,
    positions: NDArray[np.float64] | None = None,
    ax: Any = None,
    node_labels: bool = True,
    highlight_edges: Sequence[tuple[int, int]] | None = None,
) -> Any:
    """Draw the topology. ``positions`` defaults to a circular layout.

    Pass the agents' actual positions to overlay the graph on a trajectory plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    n = graph.n_agents
    if positions is None:
        angles = 2 * np.pi * np.arange(n) / n
        positions = np.column_stack([np.cos(angles), np.sin(angles)])
    positions = np.asarray(positions, dtype=float)

    highlight = (
        set() if highlight_edges is None else {(min(i, j), max(i, j)) for i, j in highlight_edges}
    )

    for i, j in graph.edges:
        color = "tab:red" if (i, j) in highlight else "grey"
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            color=color,
            linewidth=1.4 if (i, j) in highlight else 0.8,
            alpha=0.8,
            zorder=1,
        )

    for i in range(n):
        ax.scatter(
            positions[i, 0],
            positions[i, 1],
            color=AGENT_COLORS[i % len(AGENT_COLORS)],
            s=140,
            zorder=2,
        )
        if node_labels:
            ax.annotate(
                str(i),
                (positions[i, 0], positions[i, 1]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
            )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Communication topology")
    return ax


def plot_topology_timeline(
    graphs: Sequence[CommunicationGraph], ax: Any = None, dt: float = 1.0
) -> Any:
    """One row per edge, shaded where that edge is active — a switching-signal raster.

    Add the algebraic connectivity as an overlaid line so the reader can see connectivity
    drop at each switch.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    graphs = list(graphs)
    all_edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for graph in graphs:
        for edge in graph.edges:
            if edge not in seen:
                seen.add(edge)
                all_edges.append(edge)

    edge_index = {edge: row for row, edge in enumerate(all_edges)}
    steps = np.arange(len(graphs))
    active = np.zeros((len(all_edges), len(graphs)))
    lambda2 = np.zeros(len(graphs))
    for step, graph in enumerate(graphs):
        lambda2[step] = graph.algebraic_connectivity()
        for edge in graph.edges:
            active[edge_index[edge], step] = 1.0

    ax.imshow(
        active,
        aspect="auto",
        cmap="Greys",
        interpolation="none",
        extent=[0, len(graphs), -0.5, len(all_edges) - 0.5],
        origin="lower",
    )
    ax.set_yticks(np.arange(len(all_edges)))
    ax.set_yticklabels([f"{i}-{j}" for i, j in all_edges])
    ax.set_xlabel("control step")

    ax2 = ax.twinx()
    ax2.plot(steps, lambda2, color="tab:blue", linewidth=1.2)
    ax2.set_ylabel("algebraic connectivity", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_title("Topology switching timeline")
    return ax


def plot_communication_load(
    loads: dict[str, NDArray[np.float64]],
    x_values: NDArray[np.float64],
    ax: Any = None,
    xlabel: str = "number of agents",
) -> Any:
    """Bytes (or packets) per solve versus a swept parameter, one line per topology.

    Annotate the ``O(N^2)`` complete-graph curve against the ``O(N)`` sparse curves; this
    plot is the argument for why distribution is worth the trouble.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    x_values = np.asarray(x_values, dtype=float)
    for name, values in loads.items():
        ax.plot(x_values, np.asarray(values, dtype=float), marker="o", linewidth=1.5, label=name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("bytes per solve")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    ax.set_title("Communication load")
    return ax


# ---------------------------------------------------------------------- animation


def animate_formation(
    log: SimulationLog,
    formation: FormationSpec | None = None,
    show_graph: bool = False,
    show_predictions: bool = True,
    trail_length: int = 40,
    interval_ms: int = 50,
    figsize: tuple[float, float] = (7.0, 7.0),
    title: str | None = None,
    events: Mapping[int, str] | None = None,
) -> Any:
    """Build a ``matplotlib.animation.FuncAnimation`` of the closed-loop run.

    Frame contents: agent markers, fading position trails, faint predicted horizons,
    optional communication edges that appear/disappear as the topology switches, and a
    corner text box with ``t``, ADMM iterations at that step, and formation error.

    Use ``blit=False``. The edge collection changes cardinality between frames when the
    topology switches, which blitting cannot handle.

    ``events`` maps a frame index to a short label (e.g. ``"SPLIT"`` / ``"MERGE"``) that
    is stamped onto that frame, so a switching-topology GIF can mark the instants where
    the graph changes rather than leaving the viewer to infer them.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=figsize)
    positions = log.positions
    n_agents = positions.shape[1]

    if title is None:
        title = "Formation control"
    fig.suptitle(title)

    (path_lines,) = ax.plot([], [], alpha=0.35, linewidth=0.8, color="grey")
    scatter = ax.scatter(
        positions[0, :, 0],
        positions[0, :, 1],
        c=[AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(n_agents)],
        s=90,
        zorder=3,
    )
    prediction_lines = [
        ax.plot([], [], alpha=0.25, linewidth=0.8, color=AGENT_COLORS[i % len(AGENT_COLORS)])[0]
        for i in range(n_agents)
    ]
    edge_lines: list[Any] = []
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize="small")
    event_text = ax.text(
        0.98,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize="large",
        fontweight="bold",
        color="tab:red",
    )

    def _draw_edges(graph: CommunicationGraph) -> None:
        for line in edge_lines:
            line.remove()
        edge_lines.clear()
        for i, j in graph.edges:
            (line,) = ax.plot(
                [positions[0, i, 0], positions[0, j, 0]],
                [positions[0, i, 1], positions[0, j, 1]],
                color="grey",
                linewidth=0.6,
                alpha=0.5,
                zorder=1,
            )
            edge_lines.append(line)

    def update(frame: int) -> list[Any]:
        x = positions[frame, :, 0]
        y = positions[frame, :, 1]
        scatter.set_offsets(np.column_stack([x, y]))

        lo = max(0, frame - trail_length)
        path_lines.set_data(positions[lo : frame + 1, :, 0].T, positions[lo : frame + 1, :, 1].T)

        if show_predictions and frame < len(log.predictions):
            pred = log.predictions[frame]
            for i, line in enumerate(prediction_lines):
                line.set_data(pred[i, :, 0], pred[i, :, 1])
        else:
            for line in prediction_lines:
                line.set_data([], [])

        if show_graph and frame < len(log.graphs):
            _draw_edges(log.graphs[frame])

        if frame < len(log.admm_iterations):
            info = (
                f"t = {log.time[frame]:.1f} s\n"
                f"iterations = {log.admm_iterations[frame]}\n"
                f"formation error = {log.formation_error[frame]:.3f} m"
            )
        else:
            info = f"t = {frame * (log.time[1] - log.time[0]) if len(log.time) > 1 else 0:.1f} s"
        info_text.set_text(info)
        event_text.set_text(events.get(frame, "") if events else "")

        artists: list[Any] = [
            scatter,
            path_lines,
            info_text,
            event_text,
            *prediction_lines,
            *edge_lines,
        ]
        return artists

    ax.set_xlim(positions[:, :, 0].min() - 1.0, positions[:, :, 0].max() + 1.0)
    ax.set_ylim(positions[:, :, 1].min() - 1.0, positions[:, :, 1].max() + 1.0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    return FuncAnimation(fig, update, frames=positions.shape[0], interval=interval_ms, blit=False)


def save_animation(
    anim: Any, path: str, fps: int = 20, dpi: int = 110, writer: str | None = None
) -> None:
    """Write a GIF (Pillow) or MP4 (ffmpeg), inferred from the extension.

    Keep GIFs under about 5 MB or GitHub will not autoplay them in the README: cap the
    figure at ~7 inches, use ``dpi<=110``, and subsample frames rather than lowering fps
    below ~15 (a choppy GIF reads as a broken demo).
    """
    if path.endswith(".gif"):
        anim.save(path, writer="pillow", fps=fps, dpi=dpi)
    elif path.endswith(".mp4"):
        anim.save(path, writer=writer or "ffmpeg", fps=fps, dpi=dpi)
    else:
        raise ValueError("path must end in .gif or .mp4")


def _switch_events(log: SimulationLog) -> dict[int, str]:
    """Label each topology switch as ``SPLIT`` or ``MERGE`` from edge-count changes.

    A step whose graph has fewer edges than the previous step is a split (an edge was
    removed); more edges is a merge (an edge was added). Steps with an unchanged edge
    count are not labelled, even if the topology differs.
    """
    events: dict[int, str] = {}
    graphs = log.graphs
    if len(graphs) < 2:
        return events
    prev_n = len(graphs[0].edges)
    for step in range(1, len(graphs)):
        cur_n = len(graphs[step].edges)
        if cur_n < prev_n:
            events[step] = "SPLIT"
        elif cur_n > prev_n:
            events[step] = "MERGE"
        prev_n = cur_n
    return events


def make_readme_media(
    log_4agent: SimulationLog,
    log_switching: SimulationLog,
    out_dir: str,
    switch_events: Mapping[int, str] | None = None,
) -> None:
    """Regenerate every file in ``media/`` from two logs.

    Called by ``python -m distributed_mpc_admm.plotting`` and by the release checklist, so
    the committed media can always be reproduced from committed code.

    ``switch_events`` annotates the switching GIF's split/merge instants; when omitted it
    is derived from the switching log's graph sequence.
    """
    import os

    import matplotlib.pyplot as plt

    apply_style("readme")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_convergence(log_4agent.histories, ax=ax)
    fig.savefig(os.path.join(out_dir, "convergence_curves.png"))
    plt.close(fig)

    anim = animate_formation(log_4agent, show_graph=True, show_predictions=True)
    save_animation(anim, os.path.join(out_dir, "4_agent_formation.gif"))
    plt.close(anim._fig)

    if switch_events is None:
        switch_events = _switch_events(log_switching)
    anim = animate_formation(
        log_switching,
        show_graph=True,
        show_predictions=True,
        title="Topology switching",
        events=switch_events,
    )
    save_animation(anim, os.path.join(out_dir, "topology_switch.gif"))
    plt.close(anim._fig)


if __name__ == "__main__":
    # Convenience entry point: regenerate the README media from committed logs.
    # Usage: python -m distributed_mpc_admm.plotting
    raise SystemExit("make_readme_media requires two SimulationLog instances; call it directly.")
