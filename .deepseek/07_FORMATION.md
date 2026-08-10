# §7 · Formation geometry

**Governs:** `python/distributed_mpc_admm/formation_constraints.py`
**Milestone:** M3
**Done when:** every geometry and rigidity test in [§10.2](10_TESTS.md) passes.

Formation targets are the only thing that couples the agents. This module decides *what* the
formation is; [06_ADMM.md](06_ADMM.md) decides how the agents agree on it.

---

## §7.1 Two decisions that are already made

**Costs, not hard constraints.** Formation targets enter the local QP as quadratic costs. A hard
equality on relative positions goes infeasible the moment the initial condition is inconsistent with
the actuator limits, and it destroys the strong convexity the convergence argument leans on. If a
hard version is ever needed, add it as a *slacked* constraint with a large linear penalty — never a
bare equality.

**Relative offsets, not distances.** Every edge carries `d_ij = o_i − o_j ∈ R^dim`. A
distance-only encoding (`‖p_i − p_j‖ = ℓ_ij`) is non-convex and would put a different problem class
in the local QP. The offset encoding also fixes orientation, which is why the repository converges
on frameworks that distance-based rigidity theory would call flexible — say that plainly wherever
rigidity is discussed (§7.3).

Three modes, all reducing to the same edge terms:

| Mode | `d_ij` | Tracking |
| --- | --- | --- |
| `rigid` | from the offsets | every agent, or none |
| `leader_follower` | from the offsets | leaders only |
| `consensus` (rendezvous) | all zero | usually none |

## §7.2 `FormationSpec`

Offsets are stored **mean-centred**, so the anchor is the centroid. Every factory MUST subtract the
mean before returning; `test_offsets_are_mean_centred` checks it. Do it even where the construction
already produces a zero mean — the invariant should hold by construction, not by luck.

`anchor_offsets()` applies scale then rotation:

```python
R = np.array([[cos, -sin], [sin, cos]])
return self.scale * (self.offsets @ R.T)
```

`relative_offset(i, j)` is `anchor_offsets()[i] - anchor_offsets()[j]`, computed from the
**transformed** offsets so `scale` and `rotation` propagate. Antisymmetric by construction — do not
add a special case for `i == j`.

`edge_offsets(agent)` returns `{j: relative_offset(agent, j) for j in graph.neighbors(agent)}`. This
is exactly what goes into `LocalProblemData.offsets`.

Factories:

| Factory | Geometry | Default graph |
| --- | --- | --- |
| `regular_polygon(n, radius)` | angles `2πk/n` | `cycle(n)` |
| `line(n, spacing, heading)` | `(k − (n−1)/2)·spacing` along `heading` | `path(n)` — the hardest topology |
| `v_shape(n, spacing, half_angle)` | agent 0 at the apex, alternating arms | `path`-like, apex-rooted |
| `grid(rows, cols, spacing)` | rectangular lattice | 4-neighbour lattice, **not** complete |
| `rendezvous(n, graph)` | all zero | supplied |

## §7.3 Rigidity

Rigidity matrix `R(p)`, one row per edge `(i, j)`, `dim·N` columns:

```
row[(i,j)][i·dim : (i+1)·dim] =  (p_i − p_j)
row[(i,j)][j·dim : (j+1)·dim] = −(p_i − p_j)
```

`is_infinitesimally_rigid`: `matrix_rank(R, tol) == dim·N − dim(dim+1)/2`, i.e. `2N − 3` in 2D
(two translations and one infinitesimal rotation are the trivial motions).

Use `np.linalg.matrix_rank` with an **explicit `tol`**. The default tolerance is scale-dependent, so
the same formation scaled by 10 changes its answer — which is exactly the kind of result that
survives into a notebook and then into a claim.

`rigidity_eigenvalue`: eigenvalues of `RᵀR` ascending, return index `dim(dim+1)/2`, clipped at 0.
A continuous margin is more useful than a boolean when sweeping.

**Be honest about what this measures.** Under the offset encoding of §7.1 a non-rigid framework
still converges to a unique configuration. Rigidity is reported as a diagnostic and to make the
comparison against distance-based encodings meaningful. Do not write, in a docstring or a notebook,
that rigidity is required for convergence here — it is not, and a reader who checks will find it
out.

## §7.4 Leader-follower

`LeaderFollowerSpec.validate_against(graph)` runs a BFS from the leader set and raises `ValueError`
**naming the unreachable agent**.

An unreachable follower has nothing anchoring its absolute position, so the formation converges in
shape and drifts as a rigid body. That failure looks like a tuning problem in a trajectory plot and
costs an afternoon; raising at construction costs nothing.

`weight_for(agent, base_weight)` returns `base_weight` for leaders and
`follower_position_weight` (default `0.0`) otherwise. A small positive follower weight is the
documented escape hatch when the leader set genuinely cannot reach everyone — it regularises the
drift rather than fixing it, and the docstring should say so.

`validate_against(comm_graph)` on `FormationSpec` raises when a formation edge has no communication
link. Under a switching topology this is the check that fails first: an edge that disappears takes
its cost term with it. The caller — `DistributedMPC._rebuild_solvers`
([08_CLOSED_LOOP.md §8.2](08_CLOSED_LOOP.md)) — catches it and decides whether to freeze the last
cost or drop the term. **Silently dropping it inside this module is not an option**: the formation
then converges to a different shape with no error anywhere.

## §7.5 Error metrics

`formation_error(positions, spec, anchor_reference)` at one instant:

- `per_edge[(i,j)] = ‖(p_i − p_j) − d_ij‖` for each edge in `spec.graph`
- `edge_rms = sqrt(mean(per_edge²))`, `edge_max = max(per_edge)`
- `centroid_error = ‖mean(positions, axis=0) − anchor_reference‖`, or `0.0` when
  `anchor_reference is None`

The split between `edge_rms` and `centroid_error` is not cosmetic: shape error and rigid-body drift
are different failures with different causes (§7.4), and a single scalar hides which one you have.
`test_centroid_error_separates_shape_from_drift` pins it.

`settling_step(errors, tolerance, hold)` returns the smallest `k` such that
`np.all(errors[k:k+hold] < tolerance)` **and** the condition holds for the whole tail. `None`
otherwise. A transient dip below tolerance is not settling, and an oscillating formation must not
pass.

## §7.6 Morphing and events

`offsets_from_positions(positions, graph)` builds a spec that holds an observed configuration. This
is how a **merge** event is turned into a well-posed target: freeze the current shape at the switch
instant rather than snapping to a nominal one.

`interpolate_formations(start, end, alpha)` is a convex blend. A **morph** event is a
time-parametrised sweep of `alpha`, driven from the closed-loop layer.

Both live here, not in the ADMM loop. The solver stays oblivious to event structure — that
separation is what lets [§12.4](12_ANALYSIS.md) claim the events are a property of the *problem*
rather than of the solver.

## §7.7 Tests owned by this section

In `python/tests/test_formation_consensus.py` ([§10.2](10_TESTS.md)):

- `d_ij == −d_ji` for every pair in every built-in formation
- offsets are mean-centred
- relative offsets are invariant to translating all offsets by a constant — this is *why* the
  encoding is implementable from relative measurements only
- setting `rotation` preserves all pairwise distances
- polygon side lengths are equal for `n ∈ {4, 5, 6, 8}`
- `formation_error(spec.target_positions(anchor), spec)` is zero for **any** anchor
- displacing one agent raises `edge_max` and shows up on that agent's edges only
- `complete(4)` is infinitesimally rigid; `path(4)` is not
- `R(p)` has shape `(|E|, N·dim)` and its null space contains the three trivial motions
- a formation edge missing from the communication graph raises
- a follower unreachable from every leader raises
