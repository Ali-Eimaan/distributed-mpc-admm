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

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .communication_graph import CommunicationGraph
from .consensus_admm import ADMMHistory, SimulationLog
from .formation_constraints import FormationSpec

__all__ = [
    "AGENT_COLORS",
    "animate_formation",
    "apply_style",
    "plot_communication_load",
    "plot_convergence",
    "plot_formation_error",
    "plot_graph",
    "plot_inputs",
    "plot_rho_sweep",
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
    raise NotImplementedError


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
    raise NotImplementedError


def plot_inputs(log: SimulationLog, axes: Any = None, u_max: float | None = None) -> Any:
    """Per-agent acceleration components versus time, with the saturation band shaded.

    Whether the limits are ever active is the first question about any MPC result; make it
    visible rather than asserting it in text.
    """
    raise NotImplementedError


def plot_states(log: SimulationLog, axes: Any = None) -> Any:
    """Position and velocity components versus time in a 2x2 grid."""
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def plot_rho_sweep(
    rho_values: NDArray[np.float64],
    iterations: NDArray[np.float64],
    ax: Any = None,
    mark_optimum: bool = True,
) -> Any:
    """Log-x U-curve of iterations versus rho, annotating the empirical minimum."""
    raise NotImplementedError


def plot_formation_error(
    log: SimulationLog | Sequence[SimulationLog],
    ax: Any = None,
    labels: Sequence[str] | None = None,
    tolerance: float | None = None,
    mark_switches: bool = True,
) -> Any:
    """Edge-RMS formation error versus time, with vertical lines at topology switches."""
    raise NotImplementedError


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
    raise NotImplementedError


def plot_topology_timeline(
    graphs: Sequence[CommunicationGraph], ax: Any = None, dt: float = 1.0
) -> Any:
    """One row per edge, shaded where that edge is active — a switching-signal raster.

    Add the algebraic connectivity as an overlaid line so the reader can see connectivity
    drop at each switch.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
) -> Any:
    """Build a ``matplotlib.animation.FuncAnimation`` of the closed-loop run.

    Frame contents: agent markers, fading position trails, faint predicted horizons,
    optional communication edges that appear/disappear as the topology switches, and a
    corner text box with ``t``, ADMM iterations at that step, and formation error.

    Use ``blit=False``. The edge collection changes cardinality between frames when the
    topology switches, which blitting cannot handle.
    """
    raise NotImplementedError


def save_animation(
    anim: Any, path: str, fps: int = 20, dpi: int = 110, writer: str | None = None
) -> None:
    """Write a GIF (Pillow) or MP4 (ffmpeg), inferred from the extension.

    Keep GIFs under about 5 MB or GitHub will not autoplay them in the README: cap the
    figure at ~7 inches, use ``dpi<=110``, and subsample frames rather than lowering fps
    below ~15 (a choppy GIF reads as a broken demo).
    """
    raise NotImplementedError


def make_readme_media(log_4agent: SimulationLog, log_switching: SimulationLog, out_dir: str) -> None:
    """Regenerate every file in ``media/`` from two logs.

    Called by ``python -m distributed_mpc_admm.plotting`` and by the release checklist, so
    the committed media can always be reproduced from committed code.
    """
    raise NotImplementedError
